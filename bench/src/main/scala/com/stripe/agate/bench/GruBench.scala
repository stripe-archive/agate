package com.stripe.agate.bench

import com.stripe.agate.ops.Gru
import com.stripe.agate.tensor.{DataType, Shape, Storage, Tensor}
import org.openjdk.jmh.annotations._
import java.util.concurrent.TimeUnit
import scala.util.Try

import DataType.{Float32, Int32}
import Shape.Axes

@State(Scope.Benchmark)
class GruBench {
  def init(axes: Shape.Axes): Tensor.F = {
    val data = new Array[Float](axes.totalSize.toInt)
    var i = 0
    while (i < data.length) {
      // generate scalars between -10 and 10
      data(i) = scala.util.Random.nextFloat * 20f - 10f
      i += 1
    }
    Tensor(Float32, axes.asRowMajorDims)(Storage.ArrayStorage(data, 0))
  }

  val seqLength = 18L
  val batchSize = 32L
  val inputSize = 13L
  val hiddenSize = 8
  val linearBeforeReset = true
  val direction = Gru.Direction.Forward
  val numDirections = 1L
  val input = init(Shape.axes(seqLength, batchSize, inputSize))
  val weight = init(Shape.axes(numDirections, (3L * hiddenSize), inputSize))
  val recurrence = init(Shape.axes(numDirections, (3L * hiddenSize), hiddenSize))
  val bias = Some(init(Shape.axes(numDirections, 6L * hiddenSize)))
  val sequenceLens: Option[Tensor[Int32.type]] = None
  val initialH: Option[Tensor.F] = None
  val clip = 0f

  @Benchmark
  @BenchmarkMode(Array(Mode.AverageTime))
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  def runGru(): Try[Gru.Output[Float32.type]] =
    Gru(Float32)(
      input,
      weight,
      recurrence,
      bias,
      sequenceLens,
      initialH,
      Nil,
      Nil,
      Nil,
      clip,
      direction,
      hiddenSize,
      linearBeforeReset
    )
}
