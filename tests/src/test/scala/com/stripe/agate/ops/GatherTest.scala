package com.stripe.agate
package ops

import com.stripe.agate.tensor.{DataType, Shape}
import com.stripe.agate.tensor.Tensor.{F => Tensor}

import org.scalacheck.{Prop, Properties}
import org.typelevel.claimant.Claim
import shapeless.HNil
import scala.util.Success

import Prop.forAll
import com.stripe.agate.laws.Check._

object GatherTest extends Properties("GatherTest") {
  property("gather #1") = {
    val data = Tensor(((1.0f, 1.2f), (2.3f, 3.4f), (4.5f, 5.7f))) // 3x2
    val indices = Tensor.matrix(DataType.Int32)(Array(0, 1, 1, 2), 0, 2, 2)
    val axis = 0 // slice along Y, slices are length-2 rows
    val got = Gather(data, indices, axis) // 2x2x2
    val output = Tensor((((1.0f, 1.2f), (2.3f, 3.4f)), ((2.3f, 3.4f), (4.5f, 5.7f))))
    Claim(got == Success(output))
  }

  property("gather #2") = {
    val data = Tensor(((1.0f, 1.2f, 1.9f), (2.3f, 3.4f, 3.9f), (4.5f, 5.7f, 5.9f)))
    val indices = Tensor.matrix(DataType.Int32)(Array(0, 2), 0, 1, 2)
    val axis = 1
    val output = Tensor(((1.0f, 1.9f), (2.3f, 3.9f), (4.5f, 5.9f)) :: HNil)
    try {
      val got = Gather(data, indices, axis)
      Claim(got == Success(output))
    } catch {
      case e: Exception =>
        e.printStackTrace()
        throw e
    }
  }

  property("gather #3") = {
    val data = Tensor(
      (
        ((1f, 2f, 3f, 4f), (5f, 6f, 7f, 8f), (9f, 10f, 11f, 12f)),
        ((13f, 14f, 15f, 16f), (17f, 18f, 19f, 20f), (21f, 22f, 23f, 24f))
      )
    )

    val indices = Tensor.matrix(DataType.Int32)(Array(0, 2), 0, 1, 2)
    val axis = 1L

    val output = Tensor(
      (((1f, 2f, 3f, 4f), (9f, 10f, 11f, 12f)), ((13f, 14f, 15f, 16f), (21f, 22f, 23f, 24f))) :: HNil
    )

    val got = Gather(data, indices, axis)
    Claim(got == Success(output))
  }

  property("indices(axis) = slices(axis).map(_._1)") = forAll { (t: Tensor, n: Int) =>
    val axis = (n & 0xf).toLong
    val lhs = t.slices(axis).map(_._1).toList
    val rhs = t.indices(axis).toList
    Claim(lhs == rhs)
  }

  property("Gather works with slices(0)") = forAll { (t: Tensor) =>
    val axis = 0L

    val (indices, expected) =
      if (t.dims.rank > 0) {
        (Tensor.vector(DataType.Int32)(t.indices(axis).map(_.toInt).toArray), Some(t))
      } else {
        (Tensor.vector(DataType.Int32)(Array(0, 1)), None)
      }

    val got = Gather(t, indices, axis).toOption
    Claim(got == expected)
  }

  property("Gather works with scalar index") = {
    val data = Tensor(((1f, 0f, 5f), (0f, 2f, 4f)))

    val i0 = Tensor.scalar(DataType.Int64)(0)
    val i1 = Tensor.scalar(DataType.Int64)(1)
    val i2 = Tensor.scalar(DataType.Int64)(2)

    val expect1 = Success(Tensor((1f, 0f)))
    val expect2 = Success(Tensor((0f, 2f)))
    val expect3 = Success(Tensor((5f, 4f)))

    Claim(
      Gather(data, i0, 1L) == expect1 &&
        Gather(data, i1, 1L) == expect2 &&
        Gather(data, i2, 1L) == expect3
    )
  }
}
