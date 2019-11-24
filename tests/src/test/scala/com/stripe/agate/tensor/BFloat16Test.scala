package com.stripe.agate
package tensor

import java.lang.{Double => JDouble, Float => JFloat, Integer => JInteger, Math}
import org.typelevel.claimant.Claim
import org.scalacheck.{Gen, Prop, Properties}
import org.scalacheck.Prop.{forAllNoShrink => forAll}

import java.io._
import scala.math.pow

object BFloat16Test extends Properties("BFloat16Test") {
  val genShort: Gen[Short] =
    Gen.choose(Short.MinValue, Short.MaxValue)

  lazy val genNaN: Gen[BFloat16] =
    genShort.flatMap { n =>
      val x = new BFloat16((n | 0x7c00).toShort)
      if (x.isInfinite) genNaN else Gen.const(x)
    }

  val genInfinite: Gen[BFloat16] =
    Gen.oneOf(BFloat16.PositiveInfinity, BFloat16.NegativeInfinity)

  lazy val genFinite: Gen[BFloat16] =
    genShort.flatMap { n =>
      if ((n & 0x7c00) == 0x7c00) genFinite
      else Gen.const(new BFloat16(n))
    }

  val genNonNaN: Gen[BFloat16] =
    Gen.frequency(1 -> genInfinite, 30 -> genFinite)

  val genAny: Gen[BFloat16] =
    Gen.frequency(1 -> genInfinite, 1 -> genNaN, 30 -> genFinite)

  def floatEq(x: BFloat16, y: BFloat16): Prop = {
    val prop = Prop(if (x.isNaN) y.isNaN else !y.isNaN && x == y)
    prop :| s"floatEq($x, $y) was false"
  }

  property("finite/infinite/NaN are disjoint") = exhaust { x =>
    if (x.isInfinite) !x.isFinite && !x.isNaN
    else if (x.isFinite) !x.isInfinite && !x.isNaN
    else if (x.isNaN) !x.isInfinite && !x.isInfinite
    else false
  }

  property("x.isNaN -> x.toFloat.isNaN") = exhaust(x => x.isNaN == x.toFloat.isNaN)

  property("x.isInfinite -> x.toFloat.isInfinite") = exhaust(
    x => x.isInfinite == x.toFloat.isInfinite
  )

  property("x.signum -> x.toFloat.signum") = exhaust(
    x => JFloat.compare(x.signum, Math.signum(x.toFloat)) == 0
  )

  property("isInfinite -> == +/- inf") = exhaust { x =>
    if (x.isInfinite) {
      x == BFloat16.PositiveInfinity || x == BFloat16.NegativeInfinity
    } else {
      x != BFloat16.PositiveInfinity && x != BFloat16.NegativeInfinity
    }
  }

  property("-(x.signum) = (-x).signum") = exhaust { x =>
    JFloat.compare(-x.signum, (-x).signum) == 0
  }

  property("exhaustive: shortbits -> Float32 -> shortbits") = exhaust { f16 =>
    val f = f16.toFloat
    if (JFloat.isNaN(f)) {
      f16.isNaN && BFloat16.fromFloat(f).isNaN
    } else {
      BFloat16.fromFloat(f16.toFloat).raw == f16.raw
    }
  }

  property("BFloat16 -> Double -> BFloat16") = exhaustNonNaN { x =>
    BFloat16.fromDouble(x.toDouble) == x
  }

  property("x.toFloat.isNaN = x.toDouble.isNaN = x.isNaN") = exhaust { x =>
    x.isNaN == x.toFloat.isNaN && x.isNaN == x.toDouble.isNaN
  }

  property("x.toFloat.isInfinite = x.toDouble.isInfinite = x.isInfinite") = exhaust { x =>
    x.isInfinite == x.toFloat.isInfinite && x.isInfinite == x.toDouble.isInfinite
  }

  property("x.toFloat.isFinite = x.toDouble.isFinite = x.isFinite") = exhaust { x =>
    x.isFinite == JFloat.isFinite(x.toFloat) && x.isFinite == JDouble.isFinite(x.toDouble)
  }

  property("-x = BFloat16.fromFloat(-x.toFloat)") = exhaustNonNaN { x =>
    (-x) == BFloat16.fromFloat(-x.toFloat)
  }

  property("(x + y) = BFloat16.fromFloat(x.toFloat + y.toFloat)") = forAll(genNonNaN, genNonNaN) {
    (x, y) =>
      floatEq(x + y, BFloat16.fromFloat(x.toFloat + y.toFloat))
  }

  property("(x - y) = BFloat16.fromFloat(x.toFloat - y.toFloat)") = forAll(genNonNaN, genNonNaN) {
    (x, y) =>
      floatEq(x - y, BFloat16.fromFloat(x.toFloat - y.toFloat))
  }

  property("(x * y) = BFloat16.fromFloat(x.toFloat * y.toFloat)") = forAll(genNonNaN, genNonNaN) {
    (x, y) =>
      val prod = x * y
      if (prod.isNaN) {
        Claim((x.isZero && y.isInfinite) || (x.isInfinite && y.isZero))
      } else {
        Claim((x * y) == BFloat16.fromFloat(x.toFloat * y.toFloat))
      }
  }

  property("(x / y) = BFloat16.fromFloat(x.toFloat / y.toFloat)") = forAll(genNonNaN, genNonNaN) {
    (x, y) =>
      floatEq(x / y, BFloat16.fromFloat(x.toFloat / y.toFloat))
  }

  property("(x compare y) is consistent with JFloat.compare(x.toFloat, y.toFloat)") =
    forAll(genAny, genAny) { (x, y) =>
      val lhs = (x compare y)
      val rhs = JFloat.compare(x.toFloat, y.toFloat)
      Prop(Math.signum(lhs) == Math.signum(rhs)) :| s"$lhs ~ $rhs"
    }

  property("(x compare y) is consistent (x < y)") = forAll(genNonNaN, genNonNaN) { (x, y) =>
    if (x < y) {
      Claim(Math.signum(x.compare(y)) == -1)
    } else {
      Claim(Math.signum(x.compare(y)) != -1)
    }
  }

  property("(x < y) = (x.toFloat < y.toFloat)") = forAll(genAny, genAny) { (x, y) =>
    Claim((x < y) == (x.toFloat < y.toFloat))
  }

  property("(x <= y) = (x.toFloat <= y.toFloat)") = forAll(genAny, genAny) { (x, y) =>
    Claim((x <= y) == (x.toFloat <= y.toFloat))
  }

  property("(x > y) = (x.toFloat > y.toFloat)") = forAll(genAny, genAny) { (x, y) =>
    Claim((x > y) == (x.toFloat > y.toFloat))
  }

  property("(x >= y) = (x.toFloat >= y.toFloat)") = forAll(genAny, genAny) { (x, y) =>
    Claim((x >= y) == (x.toFloat >= y.toFloat))
  }

  // BFloat16 doesn't ensure NaN != NaN (if bits are equal)
  property("(x == y) = (x.toFloat == y.toFloat)") = forAll(genNonNaN, genNonNaN) { (x, y) =>
    Claim((x == y) == (x.toFloat == y.toFloat))
  }

  val allBFloat16 =
    (Short.MinValue to Short.MaxValue)
      .map { i =>
        new BFloat16(i.toShort)
      }
      .filterNot(_.isNaN)
      .toList
      .sorted(BFloat16.orderingForBFloat16)
      .toArray

  property("BFloat16.fromFloat returns the closest BFloat16") = {
    import scala.collection.Searching._

    def nextUp(f: BFloat16): BFloat16 = {
      val idx = allBFloat16.search(f).insertionPoint
      if (idx == allBFloat16.length - 1) BFloat16.PositiveInfinity
      else allBFloat16(idx + 1)
    }
    def nextDown(f: BFloat16): BFloat16 = {
      val idx = allBFloat16.search(f).insertionPoint
      if (idx == 0) BFloat16.NegativeInfinity
      else allBFloat16(idx - 1)
    }

    def law(floatBits: Int)(fn: Float => BFloat16): Prop = {
      val float = JFloat.intBitsToFloat(floatBits)
      val f16 = fn(float)
      if (JFloat.isFinite(float)) {
        if (f16.isInfinite) {
          Claim((BFloat16.MaxValue.toFloat < float) || (float < BFloat16.MinValue.toFloat))
        } else {
          val up = nextUp(f16)
          val down = nextDown(f16)
          Claim(
            ((float - f16.toFloat).abs <= (float - up.toFloat).abs) &&
              ((float - f16.toFloat).abs <= (float - down.toFloat).abs)
          ).label(
            s"int = ${JInteger.toBinaryString(floatBits)} float = $float, f16 = $f16, shortbits = ${f16.raw}, up = $up, ${up.raw}, down = $down, ${down.raw}"
          )
        }
      } else if (float.isInfinite) {
        Claim(f16.toFloat == float)
      } else {
        Claim(f16.isNaN)
      }
    }

    def checkFn(fn: Float => BFloat16): Prop = {
      val randomChecks = forAll(Gen.choose(Int.MinValue, Int.MaxValue))(law(_)(fn))
      val hardCases = List(-989858712, -1195880448, -2025051)

      hardCases.foldLeft(randomChecks)(_ && law(_)(fn))
    }

    checkFn(BFloat16.fromFloat(_)).label("fromFloat")
  }

  //see: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
  property("some known test vectors from wikipedia") = {
    val table = List(
      (0x3f80, 1.0),
      (0xc000, -2.0),
      (0x7f7f, 255.0 * pow(2.0, 120)),
      (0x0080, pow(2.0, -126)),
      (0, 0.0),
      (0x8000, -0.0),
      (0x3eab, 0.333984375),
      (0x7f80, Double.PositiveInfinity),
      (0xff80, Double.NegativeInfinity)
    )

    table.foldLeft(Prop(true)) {
      case (p, (s, d)) =>
        val f = d.toFloat
        val f1 = new BFloat16(s.toShort).toFloat
        p && Prop(f == f1).label(s"$f == $f1 for ${s.toHexString}")
    }
  }

  def exhaustNonNaN(p: BFloat16 => Boolean): Prop =
    exhaust(x => if (x.isNaN) true else p(x))

  def exhaust(p: BFloat16 => Boolean): Prop = {
    @annotation.tailrec
    def check(s: Short, acc: List[BFloat16]): List[BFloat16] = {
      val x = new BFloat16(s)
      val next = if (p(x)) acc else x :: acc
      if (s == Short.MaxValue) next
      else check((s + 1).toShort, next)
    }

    val fails = check(Short.MinValue, Nil)

    if (fails.isEmpty) Prop(true)
    else {
      val n = fails.size
      val lst = fails.take(10).map(x => s"$x (${x.raw})")
      val s =
        if (n < 10) lst.mkString(", ") else lst.mkString("", ", ", ", ...")

      Prop(false) :| s"failed on $n values: $s"
    }
  }

  property("MinusZero is correct") = Claim(BFloat16.MinusOne.toFloat == -1f)
  property("NegativeZero is correct") = Claim(
    BFloat16.NegativeZero.toFloat == -0f &&
      BFloat16.NegativeZero.isNegativeZero &&
      BFloat16.NegativeZero.isZero
  )
  property("Zero is correct") = Claim(
    BFloat16.Zero.toFloat == 0f &&
      BFloat16.Zero.isPositiveZero &&
      BFloat16.Zero.isZero
  )
  property("One is correct") = Claim(BFloat16.One.toFloat == 1f)
  property("NegativeInfinity is correct") = Claim(
    BFloat16.NegativeInfinity.isInfinite && BFloat16.NegativeInfinity.isNegativeInfinity
  )
  property("PositiveInfinity is correct") = Claim(
    BFloat16.PositiveInfinity.isInfinite && BFloat16.PositiveInfinity.isPositiveInfinity
  )

  property("MinValue is correct") = exhaustNonNaN(
    x => x >= BFloat16.MinValue || x == BFloat16.NegativeInfinity
  )
  property("MaxValue is true") = exhaustNonNaN(
    x => x <= BFloat16.MaxValue || x == BFloat16.PositiveInfinity
  )

  property("MinPositive is correct") = exhaustNonNaN(
    x => x >= BFloat16.MinPositive || x.signum < 1f
  )
  property("MaxNegative is true") = exhaustNonNaN(x => x <= BFloat16.MaxNegative || x.signum > -1f)

  property("MinPositiveNormal is correct") = exhaustNonNaN(
    x => x >= BFloat16.MinPositive || x.signum < 1f || x.isSubnormal
  )
  property("MaxNegativeNormal is true") = exhaustNonNaN(
    x => x <= BFloat16.MaxNegative || x.signum > -1f || x.isSubnormal
  )

  property("(x * y) = z, then z.isZero if x.isZero || y.isZero") = forAll(genNonNaN, genNonNaN) {
    (x, y) =>
      val z = x * y
      // two small numbers can multiply to be zero
      // but if x is zero or y is zero, then z is
      if (x.isZero) Claim(z.isZero || y.isInfinite)
      else if (y.isZero) Claim(z.isZero || x.isInfinite)
      else Prop(true)
  }
}
