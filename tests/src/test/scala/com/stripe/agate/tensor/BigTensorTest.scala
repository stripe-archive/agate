package com.stripe.agate
package tensor

import cats.effect.IO
import java.nio.file.{Files, Path}
import org.scalacheck.{Gen, Prop, Properties}
import org.typelevel.claimant.Claim
import shapeless.HNil
import scala.util.{Success, Try}

import com.stripe.agate.laws.Check._
import Prop.{forAllNoShrink => forAll}
import Shape.{Axes, Axis, Coord, Coords, Dim, Dims, Empty, NonEmpty}
import TensorParser.Interpolation

object BigTensorTest extends Properties("BigTensorTest") {

  final val Big = 100000
  final val Small = 20

  def tmp(t: Tensor.F): Path = {
    val path = Files.createTempFile("tensor", "test")
    path.toFile.deleteOnExit()
    Tensor.save(path, t).unsafeRunSync
    path
  }

  def ones(axes: Axes): Tensor.F =
    Tensor.const(DataType.Float32)(1f, axes)

  val bigAxes: Axes =
    Shape.axes(Big, Small)

  val bigGen: Gen[Tensor.F] =
    tensorForTypeAndAxes(DataType.Float32, bigAxes)

  // don't use forAll here since this is actually quite slow.
  property("large tensor equals broadcasted tensor") = {
    val Some(t0) = bigGen.sample
    val path = tmp(t0)
    Tensor
      .loadMappedRowMajorTensor(DataType.Float32, path, bigAxes)
      .use { t1 =>
        IO.pure(Claim(t0 == t1))
      }
      .unsafeRunSync()
  }

  property("gather a large tensor into a small tensor") = {
    val axes = Shape.axes(Big, Small)
    val path = tmp(ones(axes))
    Tensor
      .loadMappedRowMajorTensor(DataType.Float32, path, axes)
      .use { tensor =>
        val idata = Array(0, 1, (Big * 0.9).toInt)
        val indices = Tensor.vector(DataType.Int32)(idata)
        val axis = 0 // slice along Y, slices are length `Small` rows
        val got = ops.Gather(tensor, indices, axis)
        IO.pure(Claim(got == Success(ones(Shape.axes(idata.length, Small)))))
      }
      .unsafeRunSync()
  }
}
