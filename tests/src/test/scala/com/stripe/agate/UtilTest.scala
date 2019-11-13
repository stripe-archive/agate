package com.stripe.agate

import org.scalacheck.{Prop, Properties}
import org.typelevel.claimant.Claim

import Float.NaN
import Prop.{forAllNoShrink => forAll}
import TestImplicits._
import Util.closeTo

object UtilTest extends Properties("UtilTest") {
  property("closeTo(NaN, NaN) = true") = {
    NaN =~= NaN
  }

  property("closeTo(x, y) = closeTo(y, x)") = forAll { (x: Float, y: Float) =>
    Claim(closeTo(x, y) == closeTo(y, x))
  }

  property("closeTo(NaN, x) = x.isNaN") = forAll { (x: Float) =>
    Claim(closeTo(NaN, x) == x.isNaN)
  }
}
