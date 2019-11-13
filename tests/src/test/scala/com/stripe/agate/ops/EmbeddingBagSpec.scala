package com.stripe.agate
package ops

import com.stripe.agate.tensor.{DataType, Shape, Tensor}

import org.scalacheck.{Gen, Prop, Properties}
import org.typelevel.claimant.Claim
import Prop.forAll

import scala.util.Success

import com.stripe.agate.laws.Check._
import com.stripe.agate.tensor.TensorParser.Interpolation

object EmbeddingBagTest extends Properties("EmbeddingBagTest") {

  val data = Tensor(((10.0f, 10.0f, 100.0f), (2.0f, 20.0f, 400.0f), (4.0f, 30.0f, 300.0f)))

  val genMode: Gen[EmbeddingBag.Mode] =
    Gen.oneOf(EmbeddingBag.Mode.Sum, EmbeddingBag.Mode.Mean, EmbeddingBag.Mode.Max)

  case class Input(
      data: Tensor.F,
      mode: EmbeddingBag.Mode,
      input: Tensor[DataType.Int64.type],
      offsets: Option[Tensor[DataType.Int64.type]]
  )

  val gen2DInput: Gen[Input] =
    for {
      data <- genMatrix
      mode <- genMode
      rows = data.dims.lengthOf(0).get
      cols <- Gen.choose(1, 4)
      nInputs <- Gen.choose(1, 10)
      inputData <- Gen.listOfN(cols * nInputs, Gen.choose(0L, rows - 1))
    } yield {
      val input = Tensor(DataType.Int64)(inputData.toArray, 0, Shape.rowMajorDims(nInputs, cols))
      Input(data, mode, input, None)
    }

  private def flatten[D <: DataType](m: Tensor[D]): Tensor[m.dataType.type] = {
    val rows = m.dims.lengthOf(0).get
    val cols = m.dims.lengthOf(1).get
    m.reshape(Shape.axes(rows * cols)).get
  }

  property("2D is equivalent to 1D with offsets") = forAll(gen2DInput) {
    case Input(data, mode, input, _) =>
      val rows = input.dims.lengthOf(0).get
      val cols = input.dims.lengthOf(1).get
      val offsets =
        Tensor(DataType.Int64)(Array.tabulate(rows.toInt)(_ * cols), 0, Shape.rowMajorDims(rows))
      val actual = EmbeddingBag(data, mode, flatten(input), Some(offsets), None)
      val expected = EmbeddingBag(data, mode, input, None, None)
      Claim(actual == expected)
  }

  property("embedding bag sum") = {

    val c0 = {
      val input1 = tensor"[0]".cast(DataType.Int64)
      val got1 = EmbeddingBag(data, EmbeddingBag.Mode.Sum, input1, Some(input1), None)
      val input2 = tensor"[[0]]".cast(DataType.Int64)
      val got2 = EmbeddingBag(data, EmbeddingBag.Mode.Sum, input2, None, None)
      val output = tensor"[[10.0, 10.0, 100.0]]"
      Claim(got1 == Success(output))
      Claim(got2 == Success(output))
    }

    val c1 = {
      val input1 = tensor"[0, 2]".cast(DataType.Int64)
      val offsets1 = tensor"[0]".cast(DataType.Int64)
      val got1 = EmbeddingBag(data, EmbeddingBag.Mode.Sum, input1, Some(offsets1), None)
      val input2 = tensor"[[0, 2]]".cast(DataType.Int64)
      val got2 = EmbeddingBag(data, EmbeddingBag.Mode.Sum, input2, None, None)
      val output = tensor"[[14.0, 40.0, 400.0]]"
      Claim(got1 == Success(output))
      Claim(got2 == Success(output))
    }

    val c2 = {
      val input = tensor"[[0, 1]]".cast(DataType.Int64)
      val got = EmbeddingBag(data, EmbeddingBag.Mode.Sum, input, None, None)
      val output = tensor"[[12.0, 30.0, 500.0]]"
      Claim(got == Success(output))
    }

    val c3 = {
      val input = tensor"[[0, 1], [0, 1]]".cast(DataType.Int64)
      val got = EmbeddingBag(data, EmbeddingBag.Mode.Sum, input, None, None)
      val output = tensor"[[12.0, 30.0, 500.0], [12.0, 30.0, 500.0]]"
      Claim(got == Success(output))
    }

    c0 && c1 && c2 && c3
  }

  property("embedding bag max") = {

    val c0 = {
      val input = tensor"[[0]]".cast(DataType.Int64)
      val got = EmbeddingBag(data, EmbeddingBag.Mode.Max, input, None, None)
      val output = tensor"[[10, 10, 100]]"
      Claim(got == Success(output))
    }

    val c1 = {
      val input = tensor"[[0, 2]]".cast(DataType.Int64)
      val got = EmbeddingBag(data, EmbeddingBag.Mode.Max, input, None, None)
      val output = tensor"[[10.0, 30.0, 300.0]]"
      Claim(got == Success(output))
    }

    val c2 = {
      val input = tensor"[[0, 1]]".cast(DataType.Int64)
      val got = EmbeddingBag(data, EmbeddingBag.Mode.Max, input, None, None)
      val output = tensor"[[10.0, 20.0, 400.0]]"
      Claim(got == Success(output))
    }

    val c3 = {
      val input = tensor"[0, 1, 2, 0, 1, 0, 2, 1, 1, 2]".cast(DataType.Int64)
      val offsets = tensor"[0, 3, 4, 5, 7]".cast(DataType.Int64)
      val got = EmbeddingBag(data, EmbeddingBag.Mode.Max, input, Some(offsets), None)
      val output =
        tensor"[[10.0, 30.0, 400.0], [10, 10, 100], [2, 20, 400], [10, 30, 300], [4, 30, 400]]"
      Claim(got == Success(output))
    }

    c0 && c1 && c2 && c3
  }

  property("embedding bag mean") = {

    val c0 = {
      val input = tensor"[[0]]".cast(DataType.Int64)
      val got = EmbeddingBag(data, EmbeddingBag.Mode.Mean, input, None, None)
      val output = tensor"[[10, 10, 100]]"
      Claim(got == Success(output))
    }

    val c1 = {
      val input = tensor"[[0, 2]]".cast(DataType.Int64)
      val got = EmbeddingBag(data, EmbeddingBag.Mode.Mean, input, None, None)
      val output = tensor"[[7.0, 20.0, 200.0]]"
      Claim(got == Success(output))
    }

    val c2 = {
      val input = tensor"[[0, 1], [0, 2]]".cast(DataType.Int64)
      val got = EmbeddingBag(data, EmbeddingBag.Mode.Mean, input, None, None)
      val output = tensor"[[6.0, 15.0, 250.0], [7.0, 20.0, 200.0]]"
      Claim(got == Success(output))
    }

    val c3 = {
      val input = tensor"[0, 1, 0, 2, 2, 0, 1]".cast(DataType.Int64)
      val offsets = tensor"[0, 4, 5]".cast(DataType.Int64)
      val got = EmbeddingBag(data, EmbeddingBag.Mode.Mean, input, Some(offsets), None)
      val output = tensor"[[6.5, 17.5, 225], [4, 30, 300], [6, 15, 250]]"
      Claim(got == Success(output))
    }

    c0 && c1 && c2 && c3
  }

  property("perIndexWeights scale embeddings") = {
    val c1 = {
      val input = tensor"[[0, 1], [1, 2], [0, 2]]".cast(DataType.Int64)
      val weights = tensor"[[0.5, 0.25], [1.0, 2.0], [-1.0, -0.5]]"
      val got = EmbeddingBag(data, EmbeddingBag.Mode.Sum, input, None, Some(weights))
      val output = tensor"[[5.5, 10, 150], [10, 80, 1000], [-12, -25, -250]]"
      Claim(got == Success(output))
    }

    val c2 = {
      val input = tensor"[0, 1, 2, 0, 1, 2]".cast(DataType.Int64)
      val offsets = tensor"[0, 1, 4]".cast(DataType.Int64)
      val weights = tensor"[0.5, 0.25, -1.0, 2.0, -5.0, 0.0]"
      val got = EmbeddingBag(data, EmbeddingBag.Mode.Sum, input, Some(offsets), Some(weights))
      val output = tensor"[[5, 5, 50], [16.5, -5, 0], [-10, -100, -2000]]"
      Claim(got == Success(output))
    }

    c1 && c2
  }

  property("works with empty slices") = {
    val input = tensor"[0, 1, 0, 2, 2, 0, 1]".cast(DataType.Int64)
    val offsets = tensor"[0, 4, 4, 5, 7]".cast(DataType.Int64)
    val got = EmbeddingBag(data, EmbeddingBag.Mode.Mean, input, Some(offsets), None)
    val output = tensor"[[6.5, 17.5, 225], [0, 0, 0], [4, 30, 300], [6, 15, 250], [0, 0, 0]]"
    Claim(got == Success(output))
  }
}
