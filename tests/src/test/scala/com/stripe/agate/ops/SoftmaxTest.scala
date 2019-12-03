package com.stripe.agate
package ops

import com.stripe.agate.laws.Check._
import com.stripe.agate.tensor.{DataType, OnnxNumber, Shape, Storage}
import com.stripe.agate.tensor.Tensor
import com.stripe.agate.tensor.TensorParser.Interpolation
import org.scalacheck.{Gen, Prop, Properties}
import org.typelevel.claimant.Claim
import shapeless.HNil
import scala.util.Success

import Prop.{forAllNoShrink => forAll}
import TestImplicits._

object SoftmaxTest extends Properties("SoftmaxTest") {
  /*
x = np.array([[-1, 0, 1]]).astype(np.float32)
# expected output [[0.09003058, 0.24472848, 0.66524094]]
y = np.exp(x) / np.sum(np.exp(x), axis=1)
   */
  property("softmax #1") = {
    val t = tensor"[[-1 0 1]]"
    val got = Softmax(input = t, axis = 1).get
    val expected = tensor"[[0.09003058, 0.24472848, 0.66524094]]"
    got =~= expected
  }

  /*
def softmax_2d(x):  # type: (np.ndarray) -> np.ndarray
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
# expected output [[0.0320586, 0.08714432, 0.23688284, 0.64391428],
#                 [0.0320586, 0.08714432, 0.23688284, 0.64391428]]
y = softmax_2d(x)
   */
  property("softmax #2") = {
    val t = tensor"[[0 1 2 3] [10000 10001 10002 10003]]"
    val got = Softmax(input = t, axis = 1).get
    val expected = tensor"""
[[0.0320586, 0.08714432, 0.23688284, 0.64391428],
 [0.0320586, 0.08714432, 0.23688284, 0.64391428]]"""
    got =~= expected
  }

  property("we can stack any tensor, and get identical softmaxs") = {
    case class Arg(t: Tensor.F, axis: Int, cnt: Int) {
      def check = {
        val stacked = Tensor.stack(DataType.Float32, axis)(List.fill(cnt)(t)).get
        val sm = Softmax(input = stacked, axis = axis).get

        sm.slices(axis).iterator.map(_._2).toList match {
          case h :: tail =>
            val good = tail.forall(_ closeTo h)
            if (!good) {
              println((h :: tail).mkString("\n\n"))
            }
            Claim(good)
          case Nil => Claim(false)
        }
      }
    }

    val genArg: Gen[Arg] =
      for {
        t <- genTensor(DataType.Float32)
        axis <- if (t.axes.rank == 0) Gen.const(0) else Gen.choose(1, t.axes.rank)
        cnt <- Gen.choose(1, 10)
      } yield Arg(t, axis, cnt)

    forAll(genArg.filter(_.axis != 0))(_.check)
  }

  case class Arg(t: Tensor.F, axis: Int)
  val genArg: Gen[Arg] =
    for {
      t <- genTensor(DataType.Float32)
      axis <- Gen.choose(0, t.axes.rank)
    } yield Arg(t, axis)

  property("output of Softmax has the same shape") = {
    forAll(genArg) {
      case Arg(t, axis) =>
        val t1 = Softmax(input = t, axis = axis).get
        Claim(t1.axes == t.axes)
    }
  }

  // if you softmax each element individually they all become 1.
  property("Softmax(t, t.rank) = const(1)") = forAll { (t: Tensor.F) =>
    val got = Softmax(t, axis = t.rank).get
    val num = OnnxNumber.forDataTypeOrThrow(t.dataType)
    val expected = Tensor.const(t.dataType)(num.one, t.axes)
    Claim(got == expected)
  }

  // adding a constant to each element doesn't change softmax.
  property("Softmax(t, axis) = Softmax(t + const(1), axis)") = forAll(genArg) {
    case Arg(t0, axis) =>
      val num = OnnxNumber.forDataTypeOrThrow(t0.dataType)
      val t1 = t0.map(x => num.plus(x, num.one))
      val t2 = Softmax(t0, axis).get
      val t3 = Softmax(t1, axis).get
      t3 =~= t2
  }

  // softmax ensures values are in the unit interval.
  property("softmax values are within [0, 1]") = forAll(genArg) {
    case Arg(t0, axis) =>
      val t1 = Softmax(t0, axis).get
      Claim(t1.scalars.forall(x => 0.0 <= x && x <= 1.0)) :| s"$t1"
  }

  // generate a tensor along with 3 distinct axes ordered from
  // smallest to largest. i.e. the tensor must be of rank >= 3.
  val genCase: Gen[(Tensor.F, Int, Int, Int)] =
    for {
      a0 <- Gen.choose(1, 2)
      a1 <- Gen.choose(1, 2)
      a2 <- Gen.choose(1, 2)
      extra <- Gen.choose(0, 3)
      as <- Gen.listOfN(extra, Gen.choose(1L, 2L))
      // we end up with 3-6 axes, each of size 1-2
      // (max possible size is 2^6 = 64 values)
      axes = Shape.axes((a0.toLong :: a1.toLong :: a2.toLong :: as): _*)
      t <- tensorFromAxes(axes)
      n = as.size + 3
      k0 <- Gen.choose(0, n - 3)
      k1 <- Gen.choose(k0 + 1, n - 2)
      k2 <- Gen.choose(k1 + 1, n - 1)
    } yield (t, k0, k1, k2)

  // transpositions within axis' slices don't affect results.
  property(
    "Softmax(t.transpose(k1, k2), k0).transpose(k1, k2) = Softmax(t, k0) with 0 <= k0 < k1 < k2 < rank"
  ) = forAll(genCase) {
    case (t0, k0, k1, k2) =>
      val got = Softmax(t0.transpose(k1, k2).get, k0).get.transpose(k1, k2).get
      val expected = Softmax(t0, k0).get
      got =~= expected
  }

  // transpositions outside of axis' slices don't affect results.
  property(
    "Softmax(t.transpose(k0, k1), k2).transpose(k0, k1) = Softmax(t, k2) with 0 <= k0 < k0 < k1 < k2 < rank"
  ) = forAll(genCase) {
    case (t0, k0, k1, k2) =>
      val got = Softmax(t0.transpose(k0, k1).get, k2).get.transpose(k0, k1).get
      val expected = Softmax(t0, k2).get
      got =~= expected
  }

  // build a tensor with the given data + axes
  def build(data: Array[Float], xs: Long*): Tensor.F =
    Tensor(DataType.Float32, Shape.rowMajorDims(xs: _*))(Storage.ArrayStorage(data, 0))

  // extract data from a tensor. makes a lot of assumptions
  // currently -- if storage gets fancier we'll need to update this.
  def extract(t: Tensor.F): List[Float] =
    t.storage match {
      case Storage.ArrayStorage(data, 0) => data.toList
      case st                            => sys.error(s"couldn't extract from $st")
    }

  // 2 slices of size 2x2 are treated the same (data-wise) as 2 slices
  // of size 4.
  property("Softmax(2x2x2, 1).data = Softmax(2x4, 1).data") = forAll(Gen.listOfN(8, genScalar)) {
    lst =>
      val data = lst.toArray
      val st = Storage.ArrayStorage(data, 0)
      val t0 = build(data, 2, 2, 2)
      val t1 = build(data, 2, 4)
      val lhs = extract(Softmax(t0, 1).get)
      val rhs = extract(Softmax(t1, 1).get)
      Claim(lhs == rhs)
  }

  // 2x2 slices of size 2 are treated the same (data-wise) as 4 slices
  // of size 2.
  property("Softmax(2x2x2, 2).data = Softmax(4x2, 1).data") = forAll(Gen.listOfN(8, genScalar)) {
    lst =>
      val data = lst.toArray
      val st = Storage.ArrayStorage(data, 0)
      val t0 = build(data, 2, 2, 2)
      val t1 = build(data, 4, 2)
      val lhs = extract(Softmax(t0, 2).get)
      val rhs = extract(Softmax(t1, 1).get)
      Claim(lhs == rhs)
  }

  // 2x2 slices of size 2x2 are treated the same (data-wise) as 4
  // slices of size 4.
  property("Softmax(2x2x2x2, 2).data = Softmax(4x4, 1).data") = forAll(Gen.listOfN(16, genScalar)) {
    lst =>
      val data = lst.toArray
      val st = Storage.ArrayStorage(data, 0)
      val t0 = build(data, 2, 2, 2, 2)
      val t1 = build(data, 4, 4)
      val lhs = extract(Softmax(t0, 2).get)
      val rhs = extract(Softmax(t1, 1).get)
      Claim(lhs == rhs)
  }
}
