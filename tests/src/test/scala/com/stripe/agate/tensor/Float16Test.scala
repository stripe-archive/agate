package com.stripe.agate
package tensor

import java.lang.{Double => JDouble, Float => JFloat, Integer => JInteger, Math}
import org.typelevel.claimant.Claim
import org.scalacheck.{Gen, Prop, Properties}
import org.scalacheck.Prop.{forAllNoShrink => forAll}

import java.io._
import scala.math.pow

object Float16Test extends Properties("Float16Test") {

  val genShort: Gen[Short] =
    Gen.choose(Short.MinValue, Short.MaxValue)

  lazy val genNaN: Gen[Float16] =
    genShort.flatMap { n =>
      val x = new Float16((n | 0x7c00).toShort)
      if (x.isInfinite) genNaN else Gen.const(x)
    }

  val genInfinite: Gen[Float16] =
    Gen.oneOf(Float16.PositiveInfinity, Float16.NegativeInfinity)

  lazy val genFinite: Gen[Float16] =
    genShort.flatMap { n =>
      if ((n & 0x7c00) == 0x7c00) genFinite
      else Gen.const(new Float16(n))
    }

  val genNonNaN: Gen[Float16] =
    Gen.frequency(1 -> genInfinite, 30 -> genFinite)

  val genAny: Gen[Float16] =
    Gen.frequency(1 -> genInfinite, 1 -> genNaN, 30 -> genFinite)

  def floatEq(x: Float16, y: Float16): Prop = {
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
      x == Float16.PositiveInfinity || x == Float16.NegativeInfinity
    } else {
      x != Float16.PositiveInfinity && x != Float16.NegativeInfinity
    }
  }

  property("-(x.signum) = (-x).signum") = exhaust { x =>
    JFloat.compare(-x.signum, (-x).signum) == 0
  }

  property("exhaustive: shortbits -> Float32 -> shortbits") = exhaust { f16 =>
    val f = f16.toFloat
    if (JFloat.isNaN(f)) {
      f16.isNaN && Float16.fromFloat(f).isNaN
    } else {
      Float16.fromFloat(f16.toFloat).raw == f16.raw
    }
  }

  property("Float16 -> Double -> Float16") = exhaustNonNaN { x =>
    Float16.fromDouble(x.toDouble) == x
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

  property("-x = Float16.fromFloat(-x.toFloat)") = exhaustNonNaN { x =>
    (-x) == Float16.fromFloat(-x.toFloat)
  }

  property("(x + y) = Float16.fromFloat(x.toFloat + y.toFloat)") = forAll(genNonNaN, genNonNaN) {
    (x, y) =>
      floatEq(x + y, Float16.fromFloat(x.toFloat + y.toFloat))
  }

  property("(x - y) = Float16.fromFloat(x.toFloat - y.toFloat)") = forAll(genNonNaN, genNonNaN) {
    (x, y) =>
      floatEq(x - y, Float16.fromFloat(x.toFloat - y.toFloat))
  }

  property("(x * y) = Float16.fromFloat(x.toFloat * y.toFloat)") = forAll(genNonNaN, genNonNaN) {
    (x, y) =>
      val prod = x * y
      if (prod.isNaN) {
        Claim((x.isZero && y.isInfinite) || (x.isInfinite && y.isZero))
      } else {
        Claim((x * y) == Float16.fromFloat(x.toFloat * y.toFloat))
      }
  }

  property("(x / y) = Float16.fromFloat(x.toFloat / y.toFloat)") = forAll(genNonNaN, genNonNaN) {
    (x, y) =>
      floatEq(x / y, Float16.fromFloat(x.toFloat / y.toFloat))
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

  property("(x == y) = (x.toFloat == y.toFloat)") = forAll(genAny, genAny) { (x, y) =>
    Claim((x == y) == (x.toFloat == y.toFloat))
  }

  property("Float16.NaN != Float16.NaN") = Claim(Float16.NaN != Float16.NaN)
  property("!(Float16.NaN == Float16.NaN)") = Claim(!(Float16.NaN == Float16.NaN))
  property("!(Float16.NaN < Float16.NaN)") = Claim(!(Float16.NaN < Float16.NaN))
  property("!(Float16.NaN <= Float16.NaN)") = Claim(!(Float16.NaN <= Float16.NaN))
  property("!(Float16.NaN > Float16.NaN)") = Claim(!(Float16.NaN > Float16.NaN))
  property("!(Float16.NaN >= Float16.NaN)") = Claim(!(Float16.NaN >= Float16.NaN))
  property("(Float16.NaN compare Float16.NaN) = 0") = Claim((Float16.NaN compare Float16.NaN) == 0)

  def shortBitsToFloat(raw: Short): Float = {
    import java.lang.Math.pow
    val s = (raw >>> 14) & 2 // sign*2
    val e = (raw >>> 10) & 31 // exponent
    val m = (raw & 1023) // mantissa
    if (e == 0) {
      // either zero or a subnormal number
      if (m != 0) (1f - s) * pow(2f, -14).toFloat * (m / 1024f)
      else if (s == 0) 0f
      else -0f
    } else if (e != 31) {
      // normal number
      (1f - s) * pow(2f, e - 15).toFloat * (1f + m / 1024f)
    } else if (m != 0) {
      Float.NaN
    } else if (raw < 0) {
      Float.NegativeInfinity
    } else {
      Float.PositiveInfinity
    }
  }

  property("toFloat works like shortBitsToFloat") = exhaust { x =>
    val lhs = x.toFloat
    val rhs = shortBitsToFloat(x.raw)
    JFloat.compare(lhs, rhs) == 0
  }

  val allFloat16 =
    (Short.MinValue to Short.MaxValue)
      .map { i =>
        new Float16(i.toShort)
      }
      .filterNot(_.isNaN)
      .toList
      .sorted(Float16.orderingForFloat16)
      .toArray

  property("Float16.fromFloat returns the closest Float16") = {
    import scala.collection.Searching._

    def nextUp(f: Float16): Float16 = {
      val idx = allFloat16.search(f).insertionPoint
      if (idx == allFloat16.length - 1) Float16.PositiveInfinity
      else allFloat16(idx + 1)
    }
    def nextDown(f: Float16): Float16 = {
      val idx = allFloat16.search(f).insertionPoint
      if (idx == 0) Float16.NegativeInfinity
      else allFloat16(idx - 1)
    }

    def law(floatBits: Int)(fn: Float => Float16): Prop = {
      val float = JFloat.intBitsToFloat(floatBits)
      val f16 = fn(float)
      if (JFloat.isFinite(float)) {
        if (f16.isInfinite) {
          Claim((Float16.MaxValue.toFloat < float) || (float < Float16.MinValue.toFloat))
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

    def checkFn(fn: Float => Float16): Prop = {
      val randomChecks = forAll(Gen.choose(Int.MinValue, Int.MaxValue))(law(_)(fn))
      val hardCases = List(-989858712, -1195880448, -2025051)

      hardCases.foldLeft(randomChecks)(_ && law(_)(fn))
    }

    checkFn(Float16.fromFloat(_)).label("fromFloat")
  }

  property("some known test vectors from wikipedia") = {
    val min = pow(2.0, -14.0)
    val table = List(
      (0x0001, min * 1.0 / 1024.0),
      (0x03ff, min * 1023.0 / 1024.0),
      (0x0400, min),
      (0x7bff, pow(2.0, 15.0) * (1.0 + 1023.0 / 1024.0)),
      (0x3bff, pow(2.0, -1.0) * (1.0 + 1023.0 / 1024.0)),
      (0x3c00, 1.0),
      (0x3c01, (1.0 + 1.0 / 1024.0)),
      (0x3555, pow(2.0, -2.0) * (1.0 + 341.0 / 1024.0)),
      (0xc000, -2.0),
      (0, 0.0),
      (0x8000, -0.0),
      (0x7c00, Double.PositiveInfinity),
      (0xfc00, Double.NegativeInfinity)
    )

    table.foldLeft(Prop(true)) {
      case (p, (s, d)) =>
        val f = d.toFloat
        val f1 = new Float16(s.toShort).toFloat
        p && Prop(f == f1).label(s"$f == $f1 for $s")
    }
  }

  def exhaustNonNaN(p: Float16 => Boolean): Prop =
    exhaust(x => if (x.isNaN) true else p(x))

  def exhaust(p: Float16 => Boolean): Prop = {

    @annotation.tailrec
    def check(s: Short, acc: List[Float16]): List[Float16] = {
      val x = new Float16(s)
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

  property("MinusZero is correct") = Claim(Float16.MinusOne.toFloat == -1f)
  property("NegativeZero is correct") = Claim(
    Float16.NegativeZero.toFloat == -0f &&
      Float16.NegativeZero.isNegativeZero &&
      Float16.NegativeZero.isZero
  )
  property("Zero is correct") = Claim(
    Float16.Zero.toFloat == 0f &&
      Float16.Zero.isPositiveZero &&
      Float16.Zero.isZero
  )
  property("One is correct") = Claim(Float16.One.toFloat == 1f)
  property("NegativeInfinity is correct") = Claim(
    Float16.NegativeInfinity.isInfinite && Float16.NegativeInfinity.isNegativeInfinity
  )
  property("PositiveInfinity is correct") = Claim(
    Float16.PositiveInfinity.isInfinite && Float16.PositiveInfinity.isPositiveInfinity
  )

  property("MinValue is correct") = exhaustNonNaN(
    x => x >= Float16.MinValue || x == Float16.NegativeInfinity
  )
  property("MaxValue is true") = exhaustNonNaN(
    x => x <= Float16.MaxValue || x == Float16.PositiveInfinity
  )

  property("MinPositive is correct") = exhaustNonNaN(x => x >= Float16.MinPositive || x.signum < 1f)
  property("MaxNegative is true") = exhaustNonNaN(x => x <= Float16.MaxNegative || x.signum > -1f)

  property("MinPositiveNormal is correct") = exhaustNonNaN(
    x => x >= Float16.MinPositive || x.signum < 1f || x.isSubnormal
  )
  property("MaxNegativeNormal is true") = exhaustNonNaN(
    x => x <= Float16.MaxNegative || x.signum > -1f || x.isSubnormal
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

  property("round") = forAll(Gen.choose(Int.MinValue, Int.MaxValue), Gen.choose(1, 30)) {
    (input, shifts) =>
      val mask = (1 << shifts) - 1
      val n = input & mask
      val res = Float16.round(n, shifts)
      Claim(res <= (mask + 1))
  }
}
