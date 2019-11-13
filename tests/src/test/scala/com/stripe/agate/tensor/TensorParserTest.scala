package com.stripe.agate
package tensor

import org.scalacheck.Properties
import org.scalacheck.Prop.forAll
import org.typelevel.claimant.Claim
import scala.util.{Failure, Success}

import cats.implicits._
import fastparse.all._
import com.stripe.agate.laws.Check._
import TestImplicits._

object TensorParserTest extends Properties("TensorParserTest") {
  property("we can parse all doubles converted to string") = forAll { d: Double =>
    val dstr = d.toString
    TensorParser.float32.scalarParser.parse(dstr) match {
      case Parsed.Success(res, idx) => Claim(dstr.take(idx) == dstr)
      case _                        => Claim(false)
    }
  }

  property("parse arrays") = forAll { ds: Vector[Double] =>
    val str = ds.mkString("[", " ", "]")
    TensorParser.float32.parseShapeValues.parse(str) match {
      case Parsed.Success((shp, _), idx) =>
        Claim(idx == str.length) && Claim(shp == Shape.axes(ds.size))
      case _ => Claim(false)
    }
  }

  property("we can round-trip tensors") = forAll { (t: Tensor.F) =>
    val tstr = t.toString
    TensorParser.float32.parser.parse(tstr) match {
      case Parsed.Success(tensor0, idx) =>
        tensor0.assertDataType(t.dataType) match {
          case Success(tensor) =>
            Claim(idx == tstr.length) &&
              Claim(tensor.dims.as(Shape.Axis) == t.dims.as(Shape.Axis)) &&
              (tensor =~= t)
          case Failure(e) =>
            Claim(false)
        }
      case _ => Claim(false)
    }
  }

  property("test some examples") = {
    import TensorParser.float32.{unsafeFromString => from}
    val t1 = Tensor(Tuple1(1.0f))
    val t2 = Tensor(Tuple2(1.0f, 2.0f))
    val t3 = Tensor(Tuple1(1.0f))
    val t4 = Tensor(Tuple2(1.0f, 2.0f))
    Claim(
      from("[1]") == t1 &&
        from("[1 2]") == t2 &&
        from("[1.]") == t3 &&
        from("[1.0 2.]") == t4
    )
  }
}
