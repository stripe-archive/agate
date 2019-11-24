package com.stripe.agate.bench

import com.stripe.agate.ops.Gemm
import com.stripe.agate.tensor.{DataType, OnnxNumber, Shape, Tensor, WritableStorage}
import org.openjdk.jmh.annotations._
import java.util.concurrent.TimeUnit

@State(Scope.Benchmark)
class MatrixMultBench {
  // left:   H x C matrix
  // right:  C x W matrix
  // output: H x W matrix

  // final val height = 100
  // final val common = 100
  // final val width = 100

  // GEMM #1
  final val height = 1
  final val common = 200
  final val width = 100

  // // GEMM #2
  // final val height = 1
  // final val common = 100
  // final val width = 1

  final val left: Array[Float] =
    new Array[Float](height * common) // (height x common) matrix

  final val right: Array[Float] =
    new Array[Float](common * width) // (common x width) matrix

  import Shape.{Dim, Empty, NonEmpty}

  val a0: Tensor.F =
    Tensor(DataType.Float32)(
      left,
      0,
      NonEmpty(height, Dim(0L, common), NonEmpty(common, Shape.SingleDim, Empty))
    )
  val a1: Tensor.F =
    Tensor(DataType.Float32)(
      left,
      0,
      NonEmpty(height, Shape.SingleDim, NonEmpty(common, Dim(0L, height), Empty))
    )

  val b0: Tensor.F =
    Tensor(DataType.Float32)(
      right,
      0,
      NonEmpty(common, Dim(0L, width), NonEmpty(width, Shape.SingleDim, Empty))
    )
  val b1: Tensor.F =
    Tensor(DataType.Float32)(
      left,
      0,
      NonEmpty(common, Shape.SingleDim, NonEmpty(width, Dim(0L, common), Empty))
    )

  val c: Tensor.F =
    Tensor.zero(Shape.axes(height, width))

  def init(xs: Array[Float]): Unit =
    (0 until xs.length).foreach(i => xs(i) = scala.util.Random.nextFloat)

  init(left)
  init(right)

  @Benchmark
  @BenchmarkMode(Array(Mode.AverageTime))
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  def rowMajorTimesColMajor(): Array[Float] =
    rowTimesCol(left, right)

  def rowTimesCol(left: Array[Float], right: Array[Float]): Array[Float] = {
    val output = new Array[Float](height * width) // (height x width) matrix
    var y = 0
    while (y < height) {
      val yoffset = y * common
      val zoffset = y * width
      var x = 0
      while (x < width) {
        val xoffset = x * common
        var i = 0
        var out = 0f
        while (i < common) {
          out += left(yoffset + i) * right(xoffset + i)
          i += 1
        }
        output(zoffset + x) = out
        x += 1
      }
      y += 1
    }
    output
  }

  @Benchmark
  @BenchmarkMode(Array(Mode.AverageTime))
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  def rowMajorTimesRowMajor(): Array[Float] = {
    val dt = DataType.Float32
    val on = OnnxNumber.forDataType(dt)
    rowTimesRow[dt.Elem](left, right, on)
  }

  def rowTimesRow[@specialized A](left: Array[A], right: Array[A], on: OnnxNumber[A]): Array[A] = {
    implicit val ct: scala.reflect.ClassTag[A] = on.dataType.classTag
    val output = new Array[A](height * width) // (height x width) matrix
    var y = 0
    while (y < height) {
      val yoffset = y * common
      val zoffset = y * width
      var x = 0
      while (x < width) {
        var rightStep = x
        var i = 0
        var out = on.zero
        while (i < common) {
          out = on.plus(out, on.times(left(yoffset + i), right(rightStep)))
          i += 1
          rightStep += width
        }
        output(zoffset + x) = out
        x += 1
      }
      y += 1
    }
    output
  }

  @Benchmark
  @BenchmarkMode(Array(Mode.AverageTime))
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  def colMajorTimesColMajor(): Array[Float] =
    colTimesCol(left, right)

  def colTimesCol(left: Array[Float], right: Array[Float]): Array[Float] = {
    val output = new Array[Float](height * width) // (height x width) matrix
    var y = 0
    while (y < height) {
      var x = 0
      val zoffset = y * width
      while (x < width) {
        var leftStep = y
        var rightStep = x
        var i = 0
        var out = 0f
        while (i < common) {
          out += left(leftStep) * right(rightStep)
          i += 1
          rightStep += width
          leftStep += height
        }
        output(zoffset + x) = out
        x += 1
      }
      y += 1
    }
    output
  }

  @Benchmark
  @BenchmarkMode(Array(Mode.AverageTime))
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  def gemm00(): Tensor.F =
    Gemm(DataType.Float32)(a0, b0, c, 1f, 1f, false, false).get

  @Benchmark
  @BenchmarkMode(Array(Mode.AverageTime))
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  def gemm10(): Tensor.F =
    Gemm(DataType.Float32)(a1, b0, c, 1f, 1f, false, false).get

  @Benchmark
  @BenchmarkMode(Array(Mode.AverageTime))
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  def gemm01(): Tensor.F =
    Gemm(DataType.Float32)(a0, b1, c, 1f, 1f, false, false).get

  @Benchmark
  @BenchmarkMode(Array(Mode.AverageTime))
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  def gemm11(): Tensor.F =
    Gemm(DataType.Float32)(a1, b1, c, 1f, 1f, false, false).get
}
