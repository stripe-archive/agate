package com.stripe.agate

import java.lang.Float.{isNaN, MIN_NORMAL}
import java.lang.Math

object Util {

  implicit class RequireIntOps(val n: Long) extends AnyVal {
    def requireInt: Int =
      if ((n & 0XFFFFFFFFL) == n) n.toInt
      else throw new RuntimeException(s"$n was not a valid integer")
  }

  /**
   * See if x is "close to" y using a relative error epsilon.
   *
   * NOTE: Be careful! For values close to 0 relative error can be
   *       quite large!
   */
  def closeToRelative(x: Float, y: Float, eps: Float = 1e-5f): Boolean =
    if (isNaN(x)) {
      isNaN(y)
    } else if (isNaN(y)) {
      false
    } else {
      val xa = Math.abs(x)
      val ya = Math.abs(y)
      val delta = Math.abs(xa - ya)
      if (x == y) {
        true
      } else if (x == 0 || y == 0 || delta < MIN_NORMAL) {
        delta < eps * MIN_NORMAL
      } else {
        delta / (xa + ya) < eps
      }
    }

  /**
   * If |x| >= 1 and |y| >= 1, this is identical to closeToRelative.
   * Otherwise, we only require that |x-y| < epsilon.
   */
  def closeTo(x: Float, y: Float, eps: Float = 1e-5f): Boolean =
    if (isNaN(x)) {
      isNaN(y)
    } else if (isNaN(y)) {
      false
    } else {
      val xa = Math.abs(x)
      val ya = Math.abs(y)
      val delta = Math.abs(xa - ya)
      if (x == y) {
        true
      } else if (xa < 1f && ya < 1f) {
        delta < eps
      } else {
        delta / (xa + ya) < eps
      }
    }
}
