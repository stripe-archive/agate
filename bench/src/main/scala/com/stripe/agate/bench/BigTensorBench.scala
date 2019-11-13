package com.stripe.agate.bench

import com.stripe.agate.ops.{Gather, Gemm}
import com.stripe.agate.tensor.Shape
import com.stripe.agate.tensor.Tensor
import com.stripe.agate.tensor.TensorParser.Interpolation
import com.stripe.agate.tensor.DataType
import org.openjdk.jmh.annotations._
import java.nio.file.Files
import java.util.concurrent.TimeUnit
import scala.util.Random.nextInt

@State(Scope.Benchmark)
class BigTensorBench {

  def makeBigTensor(x: Float, axes: Shape.Axes): Tensor.F = {
    val path = Files.createTempFile("tensor", "bench")
    path.toFile.deleteOnExit()
    Tensor.save(path, Tensor.const(DataType.Float32)(x, axes)).unsafeRunSync()
    val (t, _) =
      Tensor.loadMappedRowMajorTensor(DataType.Float32, path, axes).allocated.unsafeRunSync()
    t
  }

  val a = makeBigTensor(3f, Shape.axes(4, 1000000))
  val b = makeBigTensor(3f, Shape.axes(1000000, 4))

  @Benchmark
  @BenchmarkMode(Array(Mode.AverageTime))
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  def multiplyTwoBigTensors(): Tensor.F =
    Gemm(DataType.Float32)(a, b, Tensor.Zero, 1f, 1f, false, false).get

  @Benchmark
  @BenchmarkMode(Array(Mode.AverageTime))
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  def gatherFromBigTensor(): Tensor.F = {
    val x = nextInt(1000000)
    val y = nextInt(1000000)
    val z = nextInt(1000000)
    val indices1 = Tensor.vector(DataType.Int32)(Array(0, 1, 2))
    Gather(b, indices1, 0).get
  }
}
