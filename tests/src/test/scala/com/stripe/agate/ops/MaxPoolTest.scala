package com.stripe.agate
package ops

import com.stripe.agate.tensor.{DataType, Shape, Storage, Tensor, TensorParser}
import org.scalacheck.{Prop, Properties}
import org.typelevel.claimant.Claim
import scala.util.Success

import com.stripe.agate.laws.Check._
import DataType.{Float32, Int32}
import Shape.Axes
import TensorParser.Interpolation

object MaxPoolTest extends Properties("MaxPoolTest") {
  def build(axes: Axes, xs: Iterable[Float]): Tensor[Float32.type] =
    Tensor(Float32, axes.asRowMajorDims)(Storage.ArrayStorage(xs.toArray, 0))

  property("maxpool_1d_default") = {
    val (n, c, d) = (1L, 3L, 4L)
    val total = n * c * d

    val axes = Shape.axes(n, c, d)
    val input = tensor"[[[1 2 3 4] [5 6 7 8] [9 10 11 12]]]"

    val expected = tensor"[[[2 3 4] [6 7 8] [10 11 12]]]"

    val got =
      try {
        MaxPool(Float32)(
          input = input,
          autoPad = MaxPool.AutoPad.NotSet,
          ceilMode = false,
          dilations = None,
          kernelShape = List(2),
          pads = List(0, 0),
          storageOrder = MaxPool.StorageOrder.RowMajor,
          strides = List(1)
        )
      } catch {
        case (e: Exception) =>
          e.printStackTrace()
          throw e
      }

    Claim(got == Success(MaxPool.Output(expected)))
  }

  property("maxpool_2d_ceil") = {
    val input = tensor"[[[ [1 2 3 4] [5 6 7 8] [9 10 11 12] [13 14 15 16] ]]]"

    val expected = tensor"[[[ [11 12] [15 16] ]]]"

    val got =
      try {
        MaxPool(Float32)(
          input = input,
          autoPad = MaxPool.AutoPad.NotSet,
          ceilMode = true,
          dilations = None,
          kernelShape = List(3, 3),
          pads = List(0, 0, 0, 0),
          storageOrder = MaxPool.StorageOrder.RowMajor,
          strides = List(2, 2)
        )
      } catch {
        case (e: Exception) =>
          e.printStackTrace()
          throw e
      }

    Claim(got == Success(MaxPool.Output(expected)))
  }

  property("maxpool_2d_floor") = {
    val input = tensor"[[[ [1 2 3 4] [5 6 7 8] [9 10 11 12] [13 14 15 16] ]]]"

    val expected = tensor"[[[ [11] ]]]"

    val got =
      try {
        MaxPool(Float32)(
          input = input,
          autoPad = MaxPool.AutoPad.NotSet,
          ceilMode = false,
          dilations = None,
          kernelShape = List(3, 3),
          pads = List(0, 0, 0, 0),
          storageOrder = MaxPool.StorageOrder.RowMajor,
          strides = List(2, 2)
        )
      } catch {
        case (e: Exception) =>
          e.printStackTrace()
          throw e
      }

    Claim(got == Success(MaxPool.Output(expected)))
  }

  property("1d, width = 4, kernel = 3, stride = 2, cielMode = false") = {
    val input = tensor"[[[1 2 3 4]]]"

    val expected = tensor"[[[3]]]"

    val got =
      try {
        MaxPool(Float32)(
          input = input,
          autoPad = MaxPool.AutoPad.NotSet,
          ceilMode = false,
          dilations = None,
          kernelShape = List(3),
          pads = List(0, 0),
          storageOrder = MaxPool.StorageOrder.RowMajor,
          strides = List(2)
        )
      } catch {
        case (e: Exception) =>
          e.printStackTrace()
          throw e
      }

    Claim(got == Success(MaxPool.Output(expected)))
  }

  property("1d, width = 4, kernel = 4, stride = 2, cielMode = false") = {
    val input = tensor"[[[1 2 3 4]]]"

    val expected = tensor"[[[4]]]"

    val got =
      try {
        MaxPool(Float32)(
          input = input,
          autoPad = MaxPool.AutoPad.NotSet,
          ceilMode = false,
          dilations = None,
          kernelShape = List(4),
          pads = List(0, 0),
          storageOrder = MaxPool.StorageOrder.RowMajor,
          strides = List(2)
        )
      } catch {
        case (e: Exception) =>
          e.printStackTrace()
          throw e
      }

    Claim(got == Success(MaxPool.Output(expected)))
  }
}
