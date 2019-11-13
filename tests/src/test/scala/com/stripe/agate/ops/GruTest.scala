package com.stripe.agate
package ops

import com.stripe.agate.laws.Check._
import com.stripe.agate.tensor.{DataType, Shape, Storage, Tensor, TensorParser}
import org.scalacheck.{Prop, Properties}
import org.typelevel.claimant.Claim
import scala.util.Success

import DataType.{Float32, Int32}
import TensorParser.Interpolation
import TestImplicits._

object GruTest extends Properties("GruTest") {

  // see https://github.com/Microsoft/onnxruntime/blob/master/onnxruntime/test/providers/cpu/rnn/deep_cpu_gru_op_test.cc

  def runGruTest(
      x: Tensor.F,
      w: Tensor.F,
      r: Tensor.F,
      y: Tensor.F,
      yH: Tensor.F,
      inputSize: Long,
      batchSize: Int,
      hiddenSize: Int,
      seqLength: Long,
      b: Option[Tensor.F],
      initialH: Option[Tensor.F],
      sequenceLens: Option[Tensor[Int32.type]],
      direction: Gru.Direction,
      clip: Float,
      outputSequence: Boolean,
      linearBeforeReset: Boolean,
      activations: List[Gru.ActivationFn],
      activationAlphas: List[Float],
      activationBetas: List[Float]
  ): Prop = {

    val got = try {
      Gru(Float32)(
        x,
        w,
        r,
        b,
        sequenceLens,
        initialH,
        activationAlphas,
        activationBetas,
        activations,
        clip,
        direction,
        hiddenSize,
        linearBeforeReset
      ).get
    } catch {
      case (e: Exception) =>
        e.printStackTrace()
        throw e
    }

    val expected = Gru.Output(y, yH)

    val ok = (got.y closeTo expected.y) && (got.yH closeTo expected.yH)

    if (!ok) {
      println(s"got(axes) = ${got.y.axesString}, ${got.yH.axesString}")
      println(s"expected(axes) = ${expected.y.axesString}, ${expected.yH.axesString}")
      println("")
      println(s"got(y) = ${got.y}")
      println(s"expected(y) = ${expected.y}")
      println("")
      println(s"got(yH) = ${got.yH}")
      println(s"expected(yH) = ${expected.yH}")
    }

    Claim(ok)
  }

  def runDefaultActivationsSimpleWeightsNoBias(
      direction: Gru.Direction,
      y: Tensor.F,
      yH: Tensor.F
  ): Prop = {

    val inputSize = 1L
    val batchSize = 2
    val hiddenSize = 3
    val seqLength = 2L

    val DefaultFG = List(Gru.ActivationFn.Sigmoid, Gru.ActivationFn.Tanh)

    val (numDirections, activations) =
      direction match {
        case Gru.Direction.Bidirectional => (2, DefaultFG ::: DefaultFG)
        case _                           => (1, DefaultFG)
      }

    val x: Tensor.F = tensor"[[[1] [2]] [[10] [11]]]"

    val w: Tensor.F = {
      val xs = Vector(0.1f, 0.2f, 0.3f, 1f, 2f, 3f, 10f, 11f, 12f)
      val array = (if (numDirections == 2) xs ++ xs else xs).toArray
      val dims = Shape.rowMajorDims(numDirections, 3 * hiddenSize, inputSize)
      val data = Storage.ArrayStorage(array, 0)
      Tensor(Float32, dims)(data)
    }

    val r: Tensor.F = {
      val axes = Shape.axes(numDirections, 3 * hiddenSize, hiddenSize)
      Tensor.const(Float32)(0.1f, axes)
    }

    runGruTest(
      x = x,
      w = w,
      r = r,
      y = y,
      yH = yH,
      inputSize = inputSize,
      batchSize = batchSize,
      hiddenSize = hiddenSize,
      seqLength = seqLength,
      b = None,
      initialH = None,
      sequenceLens = None,
      direction = direction,
      clip = 9999f,
      outputSequence = true,
      linearBeforeReset = false,
      activations = activations,
      activationAlphas = Nil,
      activationBetas = Nil
    )
  }

  def runDefaultActivationsSimpleWeightsWithBias(
      direction: Gru.Direction,
      y: Tensor.F,
      yH: Tensor.F,
      linearBeforeReset: Boolean,
      oneRow: Boolean
  ): Prop = {

    val seqLength = 2L
    val batchSize = if (oneRow) 1 else 2
    val inputSize = 1L
    val hiddenSize = 3

    val DefaultFG = List(Gru.ActivationFn.Sigmoid, Gru.ActivationFn.Tanh)

    val (numDirections, activations) =
      direction match {
        case Gru.Direction.Bidirectional => (2, DefaultFG ::: DefaultFG)
        case _                           => (1, DefaultFG)
      }

    val x: Tensor.F =
      if (batchSize == 2) tensor"[ [ [-0.1] [0.2] ] [ [-0.3] [0.4] ] ]"
      else tensor"[ [ [-0.1] ] [ [-0.3] ] ]"

    val w: Tensor.F = {
      val xs = Vector(0.1f, 0.2f, 0.3f, 0.2f, 0.3f, 0.1f, 0.3f, 0.1f, 0.2f)
      val array = (if (numDirections == 2) xs ++ xs else xs).toArray
      val dims = Shape.rowMajorDims(numDirections, 3 * hiddenSize, inputSize)
      val data = Storage.ArrayStorage(array, 0)
      Tensor(Float32, dims)(data)
    }

    val b: Tensor.F = {
      val xs = Vector(-0.01f, 0.1f, 0.01f, -0.2f, -0.02f, 0.02f, 0.3f, -0.3f, -0.3f, -0.03f, 0.5f,
        -0.7f, 0.05f, -0.7f, 0.3f, 0.07f, -0.03f, 0.5f)
      val array = (if (numDirections == 2) xs ++ xs else xs).toArray
      val dims = Shape.rowMajorDims(numDirections, 6 * hiddenSize)
      val data = Storage.ArrayStorage(array, 0)
      Tensor(Float32, dims)(data)
    }

    val r: Tensor.F = {
      val axes = Shape.axes(numDirections, 3 * hiddenSize, hiddenSize)
      Tensor.const(Float32)(0.1f, axes)
    }

    runGruTest(
      x = x,
      w = w,
      r = r,
      y = y,
      yH = yH,
      inputSize = inputSize,
      batchSize = batchSize,
      hiddenSize = hiddenSize,
      seqLength = seqLength,
      b = Some(b),
      initialH = None,
      sequenceLens = None,
      direction = direction,
      clip = 9999f,
      outputSequence = true,
      linearBeforeReset = linearBeforeReset,
      activations = activations,
      activationAlphas = Nil,
      activationBetas = Nil
    )
  }

  property("gru forward, no bias") = {
    val y: Tensor.F = tensor"""
[ [ [0.4750208  0.450166   0.4255575 ]
    [0.45016602 0.40131235 0.35434368] ]
  [ [0.6027093  0.5083023  0.44950223]
    [0.5754369  0.45485455 0.3747841 ] ] ]"""

    val yH: Tensor.F = tensor"""
[ [0.6027093 0.5083023  0.44950223]
  [0.5754369 0.45485455 0.3747841 ] ]"""

    runDefaultActivationsSimpleWeightsNoBias(Gru.Direction.Forward, y, yH)
  }

  property("gru reverse, no bias") = {
    val y: Tensor.F = tensor"""
   [[[0.6082785 0.50623393 0.4426924]
     [0.5803454 0.4527356 0.36886263] ]

    [[0.26894143 0.11920292 0.04742587]
     [0.24973989 0.09975048 0.03557118]]]"""

    val yH: Tensor.F = tensor"""
    [[0.6082785 0.50623393 0.4426924]
     [0.5803454 0.4527356 0.36886263]]"""

    runDefaultActivationsSimpleWeightsNoBias(Gru.Direction.Reverse, y, yH)
  }

  property("gru bidirectional, no bias") = {
    val y: Tensor.F = tensor"""
   [ [ [0.4750208  0.450166   0.4255575 ]
       [0.45016602 0.40131235 0.35434368] ]
     [ [0.6082785, 0.50623393, 0.4426924]
       [0.5803454, 0.4527356, 0.36886263] ]
     [ [0.6027093  0.5083023  0.44950223]
       [0.5754369  0.45485455 0.3747841 ] ]
     [ [0.26894143, 0.11920292, 0.04742587]
       [0.24973989, 0.09975048, 0.03557118] ] ]"""

    val yH: Tensor.F = tensor"""
    [ [0.6027093 0.5083023  0.44950223 ]
      [0.5754369 0.45485455 0.3747841  ]
      [0.6082785, 0.50623393, 0.4426924]
      [0.5803454, 0.4527356, 0.36886263] ]"""

    runDefaultActivationsSimpleWeightsNoBias(Gru.Direction.Bidirectional, y, yH)
  }

  property("gru forward, bias, linear after reset") = {
    val y: Tensor.F = tensor"""
      [ [ [0.16783132 -0.11754231 0.11977843]
          [0.2046872 -0.10372487 0.15365849] ]
        [ [0.22688604 -0.19698407 0.14017843]
          [0.33386092 -0.15799662 0.2381169] ] ]"""
    val yH: Tensor.F = tensor"""
        [ [0.22688604 -0.19698407 0.14017843]
          [0.33386092 -0.15799662 0.2381169] ]"""
    runDefaultActivationsSimpleWeightsWithBias(
      Gru.Direction.Forward,
      y,
      yH,
      linearBeforeReset = false,
      oneRow = false
    )
  }

  property("gru forward, bias, linear before reset") = {

    val y: Tensor.F = tensor"""
      [ [ [0.15024948 -0.11097029 -0.02121867]
          [0.18887489 -0.09747667 0.02093463] ]
        [ [0.19538902 -0.19016478 -0.05644283]
          [0.30856851 -0.15190377 0.05999807] ] ]"""

    val yH: Tensor.F = tensor"""
        [ [0.19538902 -0.19016478 -0.05644283]
          [0.30856851 -0.15190377 0.05999807] ]"""

    runDefaultActivationsSimpleWeightsWithBias(
      Gru.Direction.Forward,
      y,
      yH,
      linearBeforeReset = true,
      oneRow = false
    )
  }

  property("gru reverse, bias, linear before reset") = {
    val y: Tensor.F = tensor"""
      [ [ [0.20910699 -0.18880953 -0.04005555]
          [0.29700265 -0.15308119 0.04537245] ]
        [ [0.12252139 -0.12032216 -0.05064924]
          [0.21249877 -0.08884402 0.04751285] ] ]"""

    val yH: Tensor.F = tensor"""
      [ [0.20910699 -0.18880953 -0.04005555]
        [0.29700265 -0.15308119 0.04537245] ]"""

    runDefaultActivationsSimpleWeightsWithBias(
      Gru.Direction.Reverse,
      y,
      yH,
      linearBeforeReset = true,
      oneRow = false
    )
  }

  property("gru forward, bias, linear before reset, non-batching") = {
    val y: Tensor.F = tensor"""
      [ [ [0.15024948 -0.11097029 -0.02121867] ]
        [ [0.19538902 -0.19016478 -0.05644283] ] ]"""

    val yH: Tensor.F = tensor"""
      [ [0.19538902 -0.19016478 -0.05644283] ]"""

    runDefaultActivationsSimpleWeightsWithBias(
      Gru.Direction.Forward,
      y,
      yH,
      linearBeforeReset = true,
      oneRow = true
    )
  }

  property("gru reverse, bias, linear before reset, non-batching") = {
    val y: Tensor.F = tensor"""
       [ [ [0.20910699 -0.18880953 -0.04005555] ]
         [ [0.12252139 -0.12032216 -0.05064924] ] ]"""

    val yH: Tensor.F = tensor"""
       [ [0.20910699 -0.18880953 -0.04005555] ]"""

    runDefaultActivationsSimpleWeightsWithBias(
      Gru.Direction.Reverse,
      y,
      yH,
      linearBeforeReset = true,
      oneRow = true
    )
  }

  // future tests should resume from line 306 of:
  //
  // https://github.com/Microsoft/onnxruntime/blob/master/onnxruntime/test/providers/cpu/rnn/deep_cpu_gru_op_test.cc#L306

}
