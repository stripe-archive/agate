package com.stripe.agate
package ops

import cats.implicits._
import com.stripe.agate.tensor.{DataType, Shape, TensorParser}
import com.stripe.agate.tensor.Tensor.{F => Tensor}
import org.scalacheck.{Arbitrary, Gen, Prop, Properties}
import org.typelevel.claimant.Claim
import scala.util.{Success, Try}
import shapeless._

import com.stripe.agate.laws.Check._
import Prop.{forAllNoShrink => forAll}
import Shape.Axis
import TensorParser.Interpolation
import TestImplicits._

object GemmTest extends Properties("GemmTest") {
  // we use a slightly more forgiving tolerance for these tests
  implicit val tolerance: Epsilon = Epsilon(1e-3f)

  case class GemmArgs(
      a: Tensor,
      b: Tensor,
      c: Tensor,
      alpha: Float,
      beta: Float,
      transA: Boolean,
      transB: Boolean
  ) {
    def withZeroA: GemmArgs =
      copy(a = Tensor.zero(a.dims.as(Axis)))

    def withZeroB: GemmArgs =
      copy(b = Tensor.zero(b.dims.as(Axis)))

    def withZeroC: GemmArgs =
      copy(c = Tensor.zero(c.dims.as(Axis)))

    def run: Option[Tensor] = Gemm(DataType.Float32)(a, b, c, alpha, beta, transA, transB).toOption
  }

  object GemmArgs {
    implicit val arbitraryGemmArgs: Arbitrary[GemmArgs] =
      Arbitrary(for {
        h <- Gen.choose(1, 5)
        d <- Gen.choose(1, 5)
        w <- Gen.choose(1, 5)
        ta <- Gen.oneOf(true, false)
        tb <- Gen.oneOf(true, false)
        axesA = if (ta) Shape.axes(d, h) else Shape.axes(h, d)
        axesB = if (tb) Shape.axes(w, d) else Shape.axes(d, w)
        axesC = Shape.axes(h, w)
        a <- tensorFromAxes(axesA)
        b <- tensorFromAxes(axesB)
        c <- tensorFromAxes(axesC)
        alpha <- Gen.choose(-1f, 1f)
        beta <- Gen.choose(-1f, 1f)
      } yield GemmArgs(a, b, c, alpha, beta, ta, tb))
  }

  property("Gemm doesn't crash") = forAll { (g: GemmArgs) =>
    val res = Try(g.run)
    Claim(res.isSuccess)
  }

  property("A x 0 + 0 = 0") = forAll { (g0: GemmArgs) =>
    val g1 = g0.withZeroB.withZeroC
    val res =
      try {
        g1.run
      } catch {
        case (e: Exception) =>
          e.printStackTrace()
          throw e
      }
    Claim(res == Some(g1.c))
  }

  property("A x 1 + 0 = A") = forAll { (ma: Matrix) =>
    val a = ma.tensor
    val List(h, w) = a.dims.components
    val one = Tensor.identityMatrix(DataType.Float32, w)
    val got = Gemm(DataType.Float32)(a, one, Tensor.Zero, 1f, 1f, false, false).get
    Claim(got == a)
  }

  property("1 x B + 0 = B") = forAll { (mb: Matrix) =>
    val b = mb.tensor
    val List(h, w) = b.dims.components
    val one = Tensor.identityMatrix(DataType.Float32, h)
    val got = Gemm(DataType.Float32)(one, b, Tensor.Zero, 1f, 1f, false, false).get
    Claim(got == b)
  }

  property("0 x B + 0 = 0") = forAll { (g0: GemmArgs) =>
    val g1 = g0.withZeroA.withZeroC
    val res = g1.run
    Claim(res == Some(g1.c))
  }

  property("((A x B) * 0) + 0 = 0") = forAll { (g0: GemmArgs) =>
    val g1 = g0.copy(alpha = 0f).withZeroC
    val res = g1.run
    Claim(res == Some(g1.c))
  }

  property("((A x B) * 0) + (C * 0) = 0") = forAll { (g0: GemmArgs) =>
    val g1 = g0.copy(alpha = 0f).copy(beta = 0f)
    val res = g1.run
    val z = Tensor.zero(g1.c.dims.as(Axis))
    Claim(res == Some(z))
  }

  property("A x B = (B' x A')'") = forAll { (g0: GemmArgs) =>
    val g1 = g0.withZeroC
    val lhs = g1.run

    val GemmArgs(a, b, c, alpha, beta, ta, tb) = g1
    val b2 = b.transpose(0, 1).get
    val a2 = a.transpose(0, 1).get
    val c2 = c.transpose(0, 1).get
    val g2 = GemmArgs(b2, a2, c2, alpha, beta, tb, ta)
    val got: Option[Tensor.F] = g2.run
    val f: Tensor.F => Option[Tensor.F] = _.transpose(0, 1)
    val rhs = got.flatMap(f)

    Claim(lhs == rhs)
  }

  type SmallFloat12 =
    Tuple12[
      SmallFloat,
      SmallFloat,
      SmallFloat,
      SmallFloat,
      SmallFloat,
      SmallFloat,
      SmallFloat,
      SmallFloat,
      SmallFloat,
      SmallFloat,
      SmallFloat,
      SmallFloat
    ]

  property("test case #1") = forAll { (t: SmallFloat12, alpha: SmallFloat, beta: SmallFloat) =>
    val (a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3) = t

    val g = GemmArgs(
      a = Tensor(((a0.value, a1.value), (a2.value, a3.value))),
      b = Tensor(((b0.value, b1.value), (b2.value, b3.value))),
      c = Tensor(((c0.value, c1.value), (c2.value, c3.value))),
      alpha = alpha.value,
      beta = beta.value,
      transA = false,
      transB = false
    )

    val got = g.run.get

    val expected = Tensor(
      (
        (
          (a0.value * b0.value + a1.value * b2.value) * alpha.value + c0.value * beta.value,
          (a0.value * b1.value + a1.value * b3.value) * alpha.value + c1.value * beta.value
        ),
        (
          (a2.value * b0.value + a3.value * b2.value) * alpha.value + c2.value * beta.value,
          (a2.value * b1.value + a3.value * b3.value) * alpha.value + c3.value * beta.value
        )
      )
    )

    got =~= expected
  }

  type SmallFloat7 =
    Tuple7[SmallFloat, SmallFloat, SmallFloat, SmallFloat, SmallFloat, SmallFloat, SmallFloat]

  property("test case #2") = forAll { (t: SmallFloat7, alpha: SmallFloat, beta: SmallFloat) =>
    val (a0, a1, a2, b0, b1, b2, c0) = t
    val g = GemmArgs(
      a = Tensor((a0.value, a1.value, a2.value) :: HNil),
      b = Tensor((Tuple1(b0.value), Tuple1(b1.value), Tuple1(b2.value))),
      c = Tensor(Tuple1(Tuple1(c0.value))),
      alpha = alpha.value,
      beta = beta.value,
      transA = false,
      transB = false
    )

    val got = g.run.get

    val scalar = (a0.value * b0.value + a1.value * b1.value + a2.value * b2.value) * alpha.value + c0.value * beta.value
    val expected = Tensor((scalar :: HNil) :: HNil)

    got =~= expected
  }

  // this is pretty complicated. we're trying to encode the
  // broadcasting rules. basically for c you can add and remove
  // dimensions of size 1 at will. this means that a vector of size 2
  // can become a 1x2 or 2x1 matrix. it also means (confusingly) that
  // a 2x1 matrix can become a 1x2 matrix as well.
  def validGemmAxes(a: Tensor, b: Tensor, c: Tensor): Boolean =
    (a.dims.components, b.dims.components, c.dims.components) match {
      case (List(ay, ax), List(by, bx), List(cy, cx)) =>
        // c is rank 2, so c's dimensions must match (or be 1).
        ax == by && (ay == cy || cy == 1) && (bx == cx || cx == 1)
      case (List(ay, ax), List(by, bx), List(c)) =>
        // c is rank 1, so we'll broadcast it across ay, and it must
        // match bx or be 1.
        ax == by && (bx == c || c == 1)
      case (List(ay, ax), List(by, bx), Nil) =>
        // c is rank 0, so we'll broadcast it to all dimension
        ax == by
      case _ =>
        false
    }

  // property("GEMM fails on invalid input #1") = forAll { (a: Tensor, b: Tensor, c: Tensor) =>
  //   val res = Try(Gemm(DataType.Float32)(a, b, c, 1f, 1f, false, false))
  //   val ok = validGemmAxes(a, b, c)
  //   val got = res.isSuccess
  //   if (got != ok) {
  //     println(a.dims.components)
  //     println(b.dims.components)
  //     println(c.dims.components)
  //   }
  //   Claim(got == ok)
  // }

  // property("GEMM fails on invalid input #2") = forAll { (ma: Matrix, mb: Matrix, mc: Matrix) =>
  //   val (a, b, c) = (ma.tensor, mb.tensor, mc.tensor)
  //   val res = Try(Gemm(DataType.Float32)(a, b, c, 1f, 1f, false, false))
  //   val ok = validGemmAxes(a, b, c)
  //   val got = res.isSuccess
  //   if (got != ok) {
  //     println(a.dims.components)
  //     println(b.dims.components)
  //     println(c.dims.components)
  //   }
  //   Claim(got == ok)
  // }

  def matrix1x1(x: Float): Tensor =
    Tensor.matrix(DataType.Float32)(Array(x), 0, 1, 1)

  val gf = Gen.choose(-1f, 1f)

  property("matrix and scalar multiplication agree") = forAll {
    (a: Float, b: Float, c: Float, alpha: Float, beta: Float) =>
      val (ma, mb, mc) = (matrix1x1(a), matrix1x1(b), matrix1x1(c))
      val got = Gemm(DataType.Float32)(ma, mb, mc, alpha, beta, false, false).get
      val expected = matrix1x1(((a * b) * alpha) + (beta * c))
      got =~= expected
  }

  property("unidirectional broadcasting works") = {
    val a1 = tensor"[[1 2 3]]" // 1x3
    val b1 = tensor"[[4 5] [6 7] [8 9]]" // 3x2
    val c1 = tensor"[10 11]" // 2 should broadcast to 1x2
    val res = Gemm(DataType.Float32)(a1, b1, c1, 1f, 1f, false, false)
    Claim(res.isSuccess)
  }
}
