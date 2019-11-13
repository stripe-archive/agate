package com.stripe.agate
package ops

import com.stripe.agate.tensor.{DataType, OnnxNumber, Shape, TensorParser}
import com.stripe.agate.tensor.Tensor

import org.scalacheck.{Arbitrary, Gen, Prop, Properties}
import org.typelevel.claimant.Claim
import scala.util.Try
import shapeless.HNil

import com.stripe.agate.laws.Check._
import Prop.{forAllNoShrink => forAll}
import TestImplicits._

import TensorParser.Interpolation

object BatchNormalizationTest extends Properties("BatchNormalizationTest") {

  trait Args[D <: DataType] {
    val dataType: D

    val data: Tensor[dataType.type]
    val scale: Tensor[dataType.type]
    val bias: Tensor[dataType.type]
    val mean: Tensor[dataType.type]
    val variance: Tensor[dataType.type]
    val epsilon: Float

    def run: Tensor[dataType.type] =
      BatchNormalization(dataType)(data, scale, bias, mean, variance, epsilon).get
  }

  object Args {

    def apply(dt: DataType)(
        data0: Tensor[dt.type],
        scale0: Tensor[dt.type],
        bias0: Tensor[dt.type],
        mean0: Tensor[dt.type],
        variance0: Tensor[dt.type],
        epsilon0: Float
    ): Args[dt.type] = new Args[dt.type] {
      val dataType: dt.type = dt
      val data = data0
      val scale = scale0
      val bias = bias0
      val mean = mean0
      val variance = variance0
      val epsilon = epsilon0
    }

    implicit val arbitraryArgs: Arbitrary[Args[DataType.Float32.type]] =
      Arbitrary(for {
        c <- genDim
        n <- genDim
        data <- tensorFromAxes(Shape.axes(n, c))
        cax = Shape.axes(c)
        g = tensorFromAxes(cax)
        scale <- g
        bias <- g
        mean <- g
        variance <- g
        epsilon <- Gen.choose(1e-8f, 1e-4f)
      } yield Args(DataType.Float32)(data, scale, bias, mean, variance, epsilon))
  }

  property("does not crash") = forAll { (args: Args[DataType.Float32.type]) =>
    val got = Try(args.run).get
    Claim(got.axes == args.data.axes)
  }

  property("scala function seems to work") = forAll(genScalar) { x =>
    val scale = 0.5f
    val bias = 0.125f
    val mean = 1.0f
    val variance = 4.0f
    val epsilon = 0.0f

    def calc0(x: Float): Float =
      (((x.toDouble - mean.toDouble) / Math.sqrt(variance.toDouble + epsilon.toDouble)) * scale.toDouble + bias.toDouble).toFloat

    def calc1(x: Float): Float = {
      val sc = scale.toDouble / Math.sqrt(variance.toDouble + epsilon.toDouble)
      val ub = bias.toDouble - mean.toDouble * sc
      (x.toDouble * sc + ub).toFloat
    }

    val f = OnnxNumber.Float32.batchNormalize(mean, variance, scale, bias, epsilon)
    (calc0(x) =~= calc1(x)) && (calc0(x) =~= (f(x)))
  }

  property("normalization example seems to work") = forAll(genScalar, genScalar, genScalar) {
    (x0: Float, x1: Float, x2: Float) =>
      val data = Tensor((x0 :: HNil, x1 :: HNil, x2 :: HNil)) // 3x1

      val mean = 1.0f
      val variance = 4.0f
      val scale = 0.5f
      val bias = 0.125f
      val epsilon = 0.0f

      val got = BatchNormalization(DataType.Float32)(
        data,
        Tensor(scale.toFloat :: HNil),
        Tensor(bias.toFloat :: HNil),
        Tensor(mean.toFloat :: HNil),
        Tensor(variance.toFloat :: HNil),
        epsilon.toFloat
      ).get

      val f = OnnxNumber.Float32.batchNormalize(mean, variance, scale, bias, epsilon)
      val expected = Tensor((f(x0) :: HNil, f(x1) :: HNil, f(x2) :: HNil))
      got =~= expected
  }

  property("test example 1 from onnx repo") = {
    val res = Args(
      DataType.Float32
    )(
      data0 = tensor"[[[[-1, 0, 1]], [[2, 3, 4]]]]",
      scale0 = tensor"[1.0, 1.5]",
      bias0 = tensor"[0, 1]",
      mean0 = tensor"[0, 3]",
      variance0 = tensor"[1.0, 1.5]",
      epsilon0 = 1e-5f
    ).run

    val expected =
      tensor"""
  [[[[-0.999995    0.          0.999995  ]]
  [[-0.22474074  1.          2.2247407 ]]]]"""
    res =~= expected
  }
}
