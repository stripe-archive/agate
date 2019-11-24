package com.stripe.agate

import com.stripe.agate.tensor.{DataType, Tensor}
import org.scalacheck.Prop
import scala.util.Success

object TestImplicits {
  case class Epsilon(toFloat: Float)

  object Epsilon {
    implicit val defaultEpsilon: Epsilon = Epsilon(1e-5f)
  }

  implicit class TensorTestOps(val lhs: Tensor.Unknown) extends AnyVal {
    def =~=(rhs: Tensor.Unknown)(implicit e: Epsilon): Prop =
      rhs.assertDataType(lhs.dataType) match {
        case Success(t) =>
          Prop(lhs.closeTo(t, e.toFloat)) :| s"$lhs =~= $rhs (eps: ${e.toFloat})"
        case _ =>
          Prop(false) :| s"type mismatch: $lhs =~= $rhs"
      }
  }

  implicit class FloatTestOps(val lhs: Float) extends AnyVal {
    def =~=(rhs: Float)(implicit e: Epsilon): Prop =
      Prop(Util.closeTo(lhs, rhs, e.toFloat)) :| s"$lhs =~= $rhs (eps: ${e.toFloat})"
  }
}
