package com.stripe.agate
package laws

import cats.implicits._
import com.stripe.agate.tensor.{DataType, Shape, Tensor}
import com.stripe.dagon.HMap
import org.scalacheck.{Arbitrary, Cogen, Gen}

import onnx.onnx.TensorProto

import Shape.{Axes, Axis, Coord, Coords, Dims, Empty, NonEmpty}

object Check {
  val genDataType: Gen[DataType] =
    Gen.oneOf(
      DataType.Uint8,
      DataType.Uint16,
      DataType.Int8,
      DataType.Int16,
      DataType.Int32,
      DataType.Int64,
      DataType.BFloat16,
      DataType.Float16,
      DataType.Float32,
      DataType.Float64
    )

  val elemForTypeMap: HMap[DataType.Aux, Gen] =
    HMap
      .empty[DataType.Aux, Gen]
      .updated(DataType.Uint8, Gen.choose(Byte.MinValue, Byte.MaxValue))
      .updated(DataType.Uint16, Gen.choose(Short.MinValue, Short.MaxValue))
      .updated(DataType.Int8, Gen.choose(Byte.MinValue, Byte.MaxValue))
      .updated(DataType.Int16, Gen.choose(Short.MinValue, Short.MaxValue))
      .updated(DataType.Int32, Gen.choose(Int.MinValue, Int.MaxValue))
      .updated(DataType.Int64, Gen.choose(Long.MinValue, Long.MaxValue))
      .updated(DataType.BFloat16, Gen.choose(-10f, 10f).map(x => tensor.BFloat16.fromFloat(x).raw))
      .updated(DataType.Float16, Gen.choose(-10f, 10f).map(x => tensor.Float16.fromFloat(x).raw))
      .updated(DataType.Float32, Gen.choose(-10f, 10f))
      .updated(DataType.Float64, Gen.choose(-10.0, 10.0))

  def genElemForType(dt: DataType): Gen[dt.Elem] =
    elemForTypeMap[dt.Elem](dt)

  val genScalar: Gen[Float] =
    genElemForType(DataType.Float32)

  val genDim: Gen[Long] =
    Gen.choose(1L, 4L)

  val genAxes0: Gen[Axes] = {
    val g1 = genDim.map(d => Shape.axes(d))
    val g2 = for { d1 <- genDim; d2 <- genDim } yield Shape.axes(d1, d2)
    val g3 = for { d1 <- genDim; d2 <- genDim; d3 <- genDim } yield Shape.axes(d1, d2, d3)
    Gen.oneOf(g1, g2, g3)
  }

  val genAxesTransform: Gen[Axes => Gen[Axes]] = {
    def genAxis(a: Axes): Gen[Long] =
      if (a.rank == 0) Gen.const(0L) else Gen.choose(0L, (a.rank - 1).toLong)

    val tp = { axes: Axes =>
      val ga = genAxis(axes)
      Gen.zip(ga, ga).map {
        case (a1, a2) =>
          axes.transpose(a1, a2).getOrElse(axes)
      }
    }

    val splitGen = { axes: Axes =>
      genAxis(axes).flatMap { ga =>
        axes
          .split(ga)
          .map {
            case (s1, _, _, s2) =>
              (s1, s2)
          }
          .fold(Gen.const(axes)) { case (a1, a2) => Gen.oneOf(a1, a2) }
      }
    }

    val atGen = { axes: Axes =>
      genAxis(axes).map { ga =>
        axes
          .at(ga)
          .map { case (_, _, s2) => s2 }
          .getOrElse(axes)
      }
    }

    Gen.oneOf(tp, splitGen, atGen)
  }

  val genAxes: Gen[Axes] = {
    val simple = genAxes0
    val complex =
      for {
        fn <- genAxesTransform
        in <- genAxes
        a <- fn(in)
      } yield a

    Gen.oneOf(simple, complex)
  }

  implicit val arbitraryAxes: Arbitrary[Axes] =
    Arbitrary(genAxes)

  def axisFromShape[A](shape: Shape[A]): Gen[Long] = {
    val n = shape.rank
    if (n <= 1) Gen.const(0L) else Gen.choose(0L, (shape.rank - 1).toLong)
  }

  def coordsFromDims(dims: Dims): Gen[Coords] =
    dims match {
      case Empty =>
        Gen.const(Empty)
      case NonEmpty(len, _, ds) =>
        for {
          i <- Gen.choose(0L, len - 1L)
          rest <- coordsFromDims(ds)
        } yield NonEmpty(i, Coord, rest)
    }

  def tensorForTypeAndAxes(dt: DataType, axes: Axes): Gen[Tensor[dt.type]] =
    for {
      lst <- Gen.listOfN(axes.totalSize.toInt, genElemForType(dt))
    } yield Tensor(dt)(lst.toArray(dt.classTag), 0, axes.asRowMajorDims)

  def tensorFromAxes(axes: Axes): Gen[Tensor.F] =
    tensorForTypeAndAxes(DataType.Float32, axes)

  val SmallFactors: Set[Long] =
    Set(1L, 2L, 3L, 5L, 7L, 11L, 13L, 17L, 19L, 23L, 29L, 31L)

  def factors(n: Long): List[Long] = {
    @annotation.tailrec
    def smallest(f: Long, n: Long): Option[Long] =
      if (f * f > n) None
      else if (n % f == 0L) Some(f)
      else smallest(f + 1, n)

    smallest(2L, n) match {
      case None    => Nil
      case Some(f) => f :: factors(n / f)
    }
  }

  def genTransform1(dt: DataType): Gen[Tensor[dt.type] => Gen[Tensor[dt.type]]] = {
    val f1: Tensor[dt.type] => Gen[Tensor[dt.type]] = { (t: Tensor[dt.type]) =>
      val g = axisFromShape(t.dims.as(Axis))
      Gen.zip(g, g).map { case (a1, a2) => t.transpose(a1, a2).getOrElse(t) }
    }
    val f2: Tensor[dt.type] => Gen[Tensor[dt.type]] = { (t: Tensor[dt.type]) =>
      val ga = axisFromShape(t.dims.as(Axis))
      val gc = coordsFromDims(t.dims)
      for {
        axis <- ga
        coords <- gc
      } yield coords.at(axis) match {
        case Some((index, _, _)) => t.slice(axis, index)
        case None                => t
      }
    }
    val f3: Tensor[dt.type] => Gen[Tensor[dt.type]] = { (t: Tensor[dt.type]) =>
      def sizeFromDim(len: Long): Gen[Long] = {
        val factors = (SmallFactors.filter(len % _ == 0L) + len).toList
        Gen.oneOf(factors)
      }
      if (t.dims.rank == 0) Gen.const(t)
      else
        for {
          axis <- axisFromShape(t.dims.as(Axis))
          triple = t.dims.at(axis).get
          (len, _, _) = triple
          size <- sizeFromDim(len)
          i <- Gen.choose(0L, (len / size) - 1)
        } yield {
          val seq = t.chunk(axis, size).get
          seq(i.toInt)
        }
    }
    val f4: Tensor[dt.type] => Gen[Tensor[dt.type]] = { (t: Tensor[dt.type]) =>
      // select a random axis and choose one factor
      if (t.rank == 0) Gen.const(t)
      else {
        Gen
          .choose(0, t.rank - 1)
          .flatMap { axis =>
            val axisSize = t.axes.at(axis).get._1
            val fs0 = factors(axisSize)
            val fs = if (fs0.isEmpty) List(axisSize) else fs0
            Gen.choose(0, fs.size - 1).flatMap { idx =>
              val chunkSize = fs(idx)
              val items = t.chunk(axis, chunkSize).get
              val sz = items.size
              Gen.choose(0, sz - 1).map { idx =>
                items(idx)
              }
            }
          }
      }
    }
    Gen.oneOf[Tensor[dt.type] => Gen[Tensor[dt.type]]](f1, f2, f3, f4)
  }

  def genTensor(dt: DataType): Gen[Tensor[dt.type]] = {
    val g1 = genAxes.flatMap(tensorForTypeAndAxes(dt, _))
    val g2 = for {
      f <- genTransform1(dt)
      t0 <- genTensor(dt)
      t1 <- f(t0)
    } yield t1
    Gen.frequency(1 -> g1, 1 -> g2)
  }

  val genTensorU: Gen[Tensor.Unknown] = {
    val f: DataType => Gen[Tensor.Unknown] = (dt: DataType) => genTensor(dt)
    genDataType.flatMap(f)
  }

  implicit val arbitraryTensorUnknown: Arbitrary[Tensor.Unknown] =
    Arbitrary(genTensorU)
  implicit val arbitraryTensorFloat32: Arbitrary[Tensor[DataType.Float32.type]] =
    Arbitrary(genTensor(DataType.Float32))

  val genMatrix: Gen[Tensor.F] =
    for {
      x <- Gen.choose(1L, 4L)
      y <- Gen.choose(1L, 4L)
      t <- tensorFromAxes(Shape.axes(y, x))
    } yield t

  case class Matrix(tensor: Tensor.F)

  object Matrix {
    implicit val arbitraryMatrix: Arbitrary[Matrix] =
      Arbitrary(genMatrix.map(Matrix(_)))
  }

  implicit val cogenForCoords: Cogen[Coords] =
    Cogen[List[Long]].contramap((c: Coords) => c.components)

  case class SmallFloat(value: Float)

  object SmallFloat {
    implicit val arbitrarySmallFloat: Arbitrary[SmallFloat] =
      Arbitrary(Gen.choose(-1000f, 1000f).map(SmallFloat(_)))
  }

  /**
   * Generate a random permutation
   */
  def genPerm(size: Long): Gen[List[Long]] = {
    // to make a random permutation, we can unsort a list
    // randomly select one of an array to take as the first or last item
    // and repeat
    type State = (Array[Long], Int)
    def loop(state: State): Gen[State] = {
      val ary = state._1
      val sz = ary.length
      val idx = state._2
      if (idx >= sz) Gen.const(state)
      else {
        Gen
          .choose(idx, sz - 1)
          .flatMap { swapIdx =>
            val ss = ary(swapIdx)
            ary(swapIdx) = ary(idx)
            ary(idx) = ss
            loop((ary, idx + 1))
          }
      }
    }
    val init: State = ((0L until size).toArray, 0)
    loop(init).map { case (ary, _) => ary.toList }
  }
}
