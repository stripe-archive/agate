package com.stripe.agate.tensor

import cats.data.Chain
import com.stripe.dagon.HMap
import fastparse.all._
import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context
import scala.util.{Failure, Success, Try}

trait ScalarParser[E] {
  def parser: P[E]
}

object ScalarParser {

  def instance[E](pe: P[E]): ScalarParser[E] =
    new ScalarParser[E] { val parser = pe }

  val integralParser: P[String] = {
    val digits = CharIn("0123456789")
    val integerString = digits.rep(min = 1)
    val maybeSign = P("+" | "-").?

    P(maybeSign ~ integerString).!
  }

  private def make[A](p: P[String])(f: String => A): P[A] =
    p.flatMap(
      s =>
        Try(f(s)) match {
          case Success(a) => PassWith(a)
          case Failure(_) => Fail
        }
    )

  val floatingParser: P[String] = {
    val digits = CharIn("0123456789")
    val integerString = digits.rep(min = 1)
    val maybeSign = P("+" | "-").?
    val exponent = (P("e" | "E") ~ maybeSign ~ integerString)
    P(maybeSign ~ integerString ~ ("." ~ integerString.?).? ~ exponent.?).!
  }

  val byteParser: ScalarParser[Byte] =
    instance(make(integralParser)(java.lang.Byte.valueOf(_)))
  val shortParser: ScalarParser[Short] =
    instance(make(integralParser)(java.lang.Short.valueOf(_)))
  val intParser: ScalarParser[Int] =
    instance(make(integralParser)(java.lang.Integer.valueOf(_)))
  val longParser: ScalarParser[Long] =
    instance(make(integralParser)(java.lang.Long.valueOf(_)))
  val floatParser: ScalarParser[Float] =
    instance(make(floatingParser)(java.lang.Float.valueOf(_)))
  val doubleParser: ScalarParser[Double] =
    instance(make(floatingParser)(java.lang.Double.valueOf(_)))

  val dataTypeMap: HMap[DataType.Aux, ScalarParser] =
    HMap
      .empty[DataType.Aux, ScalarParser]
      .updated(DataType.Int8, byteParser)
      .updated(DataType.Int16, shortParser)
      .updated(DataType.Int32, intParser)
      .updated(DataType.Int64, longParser)
      .updated(DataType.Float32, floatParser)
      .updated(DataType.Float64, doubleParser)

  def forDataType[E](dt: DataType.Aux[E]): ScalarParser[E] =
    dataTypeMap(dt)
}

/**
 * Parser tensors from numpy output strings
 * useful for testing
 */
class TensorParser[D <: DataType](val dt: D) {

  type E = dt.Elem

  val scalarParser: P[E] = ScalarParser.forDataType(dt).parser

  val spaces: P[Unit] = CharIn(" \t\n").rep(min = 1)
  val maybeSpaces: P[Unit] = spaces.?

  val parseShapeValues: P[(Shape.Axes, Chain[E])] = {
    val rawScalar = scalarParser.map { f =>
      (Shape.axes(), Chain.one(f))
    }
    val recurse = P(parseShapeValues)

    val shaped = P(
      "[" ~ maybeSpaces ~ recurse.rep(sep = (spaces | (P(",") ~ maybeSpaces))) ~ maybeSpaces ~ "]"
    ).flatMap {
      case inners =>
        if (inners.isEmpty) {
          PassWith((Shape.axes(0), Chain.empty[E]))
        } else {
          val shape = inners.head._1
          if (inners.forall(_._1 != shape)) Fail
          else {
            val newShape = Shape.NonEmpty(inners.size.toLong, Shape.Axis, shape)
            // one extra outer dim:
            PassWith((newShape, inners.foldLeft(Chain.empty[E])(_ ++ _._2)))
          }
        }
    }

    (rawScalar | shaped)
  }

  val alloc = StorageAllocator.forDataType(dt)
  implicit val classTag = dt.classTag

  val parser: P[Tensor[dt.type]] =
    (maybeSpaces ~ parseShapeValues ~ maybeSpaces).map {
      case (s, chain) =>
        val st = alloc.toArrayStorage(chain.toList.toArray, 0)
        Tensor(dt, s.asRowMajorDims)(st)
    }

  def unsafeFromString(str: String): Tensor[dt.type] =
    parser.parse(str) match {
      case Parsed.Success(t, idx) =>
        require(idx == str.length, s"only parsed $idx characters, expected: ${str.length}")
        t
      case failure => sys.error(failure.toString)
    }
}

object TensorParser {

  val float32: TensorParser[DataType.Float32.type] =
    new TensorParser(DataType.Float32)

  val int64: TensorParser[DataType.Int64.type] =
    new TensorParser(DataType.Int64)

  implicit class Interpolation(val sc: StringContext) {
    def int64(args: Any*): Tensor[DataType.Int64.type] = macro Interpolation.parseInt64
    def tensor(args: Any*): Tensor[DataType.Float32.type] = macro Interpolation.parse
  }

  object Interpolation {

    def parseInt64(c: Context)(args: c.Tree*): c.Expr[Tensor[DataType.Int64.type]] = null

    def parse(c: Context)(args: c.Tree*): c.Expr[Tensor[DataType.Float32.type]] = {
      import c.universe._

      implicit val liftableDims: Liftable[Shape.Dims] =
        new Liftable[Shape.Dims] {
          def apply(d: Shape.Dims): c.Tree =
            d match {
              case Shape.NonEmpty(len, Shape.Dim(o, n), rest) =>
                val tail = apply(rest)
                q"com.stripe.agate.tensor.Shape.NonEmpty($len, com.stripe.agate.tensor.Shape.Dim($o, $n), $tail)"
              case Shape.Empty =>
                q"com.stripe.agate.tensor.Shape.Empty"
            }
        }

      implicit def liftableStorage[F](implicit laf: Liftable[Array[F]]): Liftable[Storage[F]] =
        new Liftable[Storage[F]] {
          def apply(d: Storage[F]): c.Tree =
            d match {
              case Storage.ArrayStorage(data, off) =>
                q"""com.stripe.agate.tensor.Storage.ArrayStorage($data, $off)"""
              case fbs @ Storage.FloatBufferStorage(_, _) =>
                liftableStorage[Float](laf)(fbs.copyToArrayStorage)
              case Storage.Chunked(_, _, _) =>
                c.abort(c.enclosingPosition, "chunked tensors are too large to use this macro")
            }
        }

      implicit val liftableDataType: Liftable[TensorParser.float32.dt.type] =
        new Liftable[TensorParser.float32.dt.type] {
          def apply(t: TensorParser.float32.dt.type): c.Tree =
            q"""com.stripe.agate.tensor.DataType.Float32"""
        }

      val Apply(_, List(Apply(_, List(Literal(Constant(s: String)))))) = c.prefix.tree
      val t = TensorParser.float32.unsafeFromString(s)
      c.Expr(
        q"""com.stripe.agate.tensor.Tensor(${t.dataType}, ${t.dims})(${t.storage})"""
      )
    }
  }
}
