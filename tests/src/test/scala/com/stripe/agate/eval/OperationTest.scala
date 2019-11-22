package com.stripe.agate
package eval

import com.stripe.agate.eval.Model.OpData
import com.stripe.agate.laws.Check._
import com.stripe.agate.tensor.{DataType, OnnxNumber, Shape, Tensor, TensorParser}
import java.io.{PrintWriter, StringWriter}
import onnx.onnx.AttributeProto
import org.scalacheck.{Gen, Prop, Properties}
import org.typelevel.claimant.Claim
import scala.util.{Failure, Success, Try}

import TensorParser.Interpolation
import TestImplicits._
import Shape.AxisRange

object OperationTest extends Properties("OperationTest") {
  def stackTrace(t: Throwable): String = {
    val sw = new StringWriter
    val pw = new PrintWriter(sw)
    t.printStackTrace(pw)
    pw.close
    sw.toString
  }

  def test(
      opName: String,
      inputs: List[Tensor.U],
      outputs: List[Tensor.U],
      attrs: Map[String, AttributeProto]
  ): Prop =
    Model.supportedOps.get(opName) match {
      case None => Prop(false) :| s"operator $opName not supported"
      case Some(bldr) =>
        val ins = inputs.zipWithIndex.map { case (n, i) => (Register(i.toString), n) }
        val off = ins.size
        val outs = outputs.zipWithIndex.map { case (o, i) => (Register((i + off).toString), o) }
        val opData = OpData(s"test of $opName", ins.map(_._1), outs.map(_._1), attrs)
        val registers = Registers(ins.toMap)
        bldr.build(opData).flatMap(_(registers)) match {
          case Success(outReg) =>
            outs.foldLeft(Prop(true)) {
              case (p, (reg, expect0)) =>
                val ex = expect0.asDataTyped
                val t: Try[Tensor[ex.dataType.type]] =
                  outReg.getAndUnify(reg, ex.dataType)
                t match {
                  case Success(got) => p && (got =~= ex)
                  case Failure(err) => p && (Prop(false) :| err.toString)
                }
            }
          case Failure(err) =>
            Prop(false) :| s"failed to run: $err: ${stackTrace(err)}"
        }
    }

  val tensorFPair: Gen[(Tensor.F, Tensor.F)] =
    genTensor(DataType.Float32)
      .flatMap { t: Tensor.F =>
        val res: Gen[(Tensor.F, Tensor.F)] = tensorForTypeAndAxes(t.dataType, t.axes).map((t, _))
        res
      }

  property("Add works as expected") = {
    Prop.forAll(tensorFPair) {
      case (left, right) =>
        test(
          "Add",
          List(left, right),
          List(Tensor.map2(DataType.Float32)(left, right)(OnnxNumber.Float32.plus(_, _)).get),
          Map.empty
        )
    }
  }

  property("Sub works as expected") = {
    Prop.forAll(tensorFPair) {
      case (left, right) =>
        test(
          "Sub",
          List(left, right),
          List(Tensor.map2(DataType.Float32)(left, right)(OnnxNumber.Float32.minus(_, _)).get),
          Map.empty
        )
    }
  }

  property("Mul works as expected") = {
    Prop.forAll(tensorFPair) {
      case (left, right) =>
        test(
          "Mul",
          List(left, right),
          List(Tensor.map2(DataType.Float32)(left, right)(OnnxNumber.Float32.times(_, _)).get),
          Map.empty
        )
    }
  }

  property("Relu works") = Prop.forAll(genTensor(DataType.Float32)) { arg =>
    test("Relu", List(arg), List(arg.map { x =>
      if (x > 0) x else 0.0f
    }), Map.empty)
  }

  property("LeakyRelu works") = Prop.forAll(genTensor(DataType.Float32), Gen.choose(0.0f, 0.5f)) {
    (arg, alpha) =>
      test("LeakyRelu", List(arg), List(arg.map { x =>
        if (x > 0) x else x * alpha
      }), Map("alpha" -> AttributeProto(floats = List(alpha))))
  }

  property("Exp works") = Prop.forAll(genTensor(DataType.Float32)) { arg =>
    test("Exp", List(arg), List(arg.map { x =>
      java.lang.Math.exp(x.toDouble).toFloat
    }), Map.empty)
  }

  property("Log works") = Prop.forAll(genTensor(DataType.Float32)) { arg =>
    test("Log", List(arg), List(arg.map { x =>
      java.lang.Math.log(x.toDouble).toFloat
    }), Map.empty)
  }

  property("Identity works") = Prop.forAll(genTensorU) { arg =>
    test("Identity", List(arg), List(arg), Map.empty)
  }

  property("Transpose works") = Prop.forAll(genTensorU.flatMap { t =>
    Gen.option(genPerm(t.rank)).map((t, _))
  }) {
    case (tensor, perm0) =>
      val perm = perm0.getOrElse((0L until tensor.rank.toLong).reverse.toList)
      val attr = perm0
        .map { p =>
          Map("perm" -> AttributeProto(ints = p))
        }
        .getOrElse(Map.empty)
      test("Transpose", List(tensor), List(tensor.transpose(perm).get), attr)
  }

  property("Slice works") = {
    def longsToTensor(l: List[Long]): Tensor[DataType.Int64.type] =
      Tensor.vector(DataType.Int64)(l.toArray)

    case class Args(
        t: Tensor.U,
        startsEnds: List[(Long, Long)],
        axes: Option[List[Long]],
        steps: Option[List[Long]]
    ) {
      def inputs: List[Tensor.U] =
        t :: longsToTensor(startsEnds.map(_._1)) :: longsToTensor(startsEnds.map(_._2)) ::
          axes.fold(List.empty[Tensor.U]) { a =>
            longsToTensor(a) :: Nil
          } :::
          steps.fold(List.empty[Tensor.U]) { a =>
            longsToTensor(a) :: Nil
          }

      def expectedOutput: Tensor.U =
        t.select(
          Operation.SliceOp
            .buildSelection(
              t.axes,
              startsEnds.map(_._1).toArray,
              startsEnds.map(_._2).toArray,
              axes.map(_.toArray),
              steps.map(_.toArray)
            )
            .get
        )
    }

    lazy val genArgs: Gen[Args] =
      genTensorU.flatMap { ten =>
        if (ten.rank > 0) {
          val selectAxis = Gen.choose(0, ten.rank.toInt - 1)
          val long = Gen.choose(Long.MinValue, Long.MaxValue)
          val maybeSane = Gen.oneOf(selectAxis.map(_.toLong), long)
          for {
            indices <- selectAxis
            startEnds <- Gen.listOfN(indices, Gen.zip(maybeSane, maybeSane))
            axes <- Gen.option(genPerm(ten.rank).map(_.take(indices)))
            steps <- if (axes.isEmpty) Gen.const(None)
            else Gen.option(Gen.listOfN(indices, Gen.choose(1L, Long.MaxValue)))
          } yield Args(ten, startEnds, axes, steps)
        } else genArgs
      }

    val c0 = Prop.forAll(genArgs) { args =>
      test("Slice", args.inputs, List(args.expectedOutput), Map.empty)
    } :| "Slice consistency test"

    val c1 = {
      val t0 = tensor"[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]"
      val expected = tensor"[[[1, 2], [3, 4]]]"
      test("Slice", t0 :: tensor"[0]" :: tensor"[1]" :: Nil, List(expected), Map.empty)
    }

    val c2 = {
      val t0 = tensor"[[1, 2, 3], [4, 5, 6]]"
      val expected = tensor"[[3, 2], [6, 5]]"
      test(
        "Slice",
        t0 :: tensor"[0, 2]" :: tensor"[2, 0]" :: tensor"[0, 1]" :: tensor"[1, -1]" :: Nil,
        List(expected),
        Map.empty
      )
    }

    // Example 1: data = [ [1, 2, 3, 4], [5, 6, 7, 8], ] axes = [0, 1] starts = [1, 0] ends = [2, 3] steps = [1, 2] result = [ [5, 7], ]
    val ex1 = {
      val data = tensor"[ [1, 2, 3, 4], [5, 6, 7, 8] ]"
      val axes = tensor"[0, 1]"
      val starts = tensor"[1, 0]"
      val ends = tensor"[2, 3]"
      val steps = tensor"[1, 2]"
      val result = tensor"[[5, 7]]"

      (test("Slice", data :: starts :: ends :: axes :: steps :: Nil, List(result), Map.empty) :| "Example 1") && {
        val built: List[AxisRange] = Operation.SliceOp
          .buildSelection(
            data.axes,
            Array(1L, 0L),
            Array(2L, 3L),
            Some(Array(0L, 1L)),
            Some(Array(1L, 2L))
          )
          .get

        Prop(built == (AxisRange(1, 2, 1) :: AxisRange(0, 3, 2) :: Nil)) :| s"Example1 built = $built"
      }
    }
    // Example 2: data = [ [1, 2, 3, 4], [5, 6, 7, 8], ] starts = [0, 1] ends = [-1, 1000] result = [ [2, 3, 4], ]
    val ex2 = {
      val data = tensor"[ [1, 2, 3, 4], [5, 6, 7, 8] ]"
      val starts = tensor"[0, 1]"
      val ends = tensor"[-1, 1000]"
      val result = tensor"[[2, 3, 4]]"

      (test("Slice", data :: starts :: ends :: Nil, List(result), Map.empty) :| "Example 2") && {
        val built: List[AxisRange] = Operation.SliceOp
          .buildSelection(data.axes, Array(0L, 1L), Array(-1L, 1000L), None, None)
          .get

        Prop(built == (AxisRange(0, 1, 1) :: AxisRange(1, 4, 1) :: Nil)) :| s"Example1 built = $built"
      }
    }

    c0 && c1 && c2 && ex1 && ex2
  }

  property("NonZero works") = {
    // Check that the NonZero operator works. Note that more detailed tests are in the TensorTest file
    // NOTE: Here we expect the result to be the transposed version!
    val t0 = tensor"[[3, 4, 0, 6], [0, 0, 0, 7]]"
    val expected = tensor"[[0, 0], [0, 1], [0, 3], [1, 3]]".cast(DataType.Int64)
    test("NonZero", t0 :: Nil, List(expected), Map.empty)
  }
}
