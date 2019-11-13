package com.stripe.agate

import cats.effect.IO
import com.stripe.agate.eval.{Model, Register}
import com.stripe.agate.tensor.{DataType, Shape, Tensor}
import org.scalacheck.{Gen, Properties}
import org.scalacheck.Prop.{forAllNoShrink => forAll}
import org.typelevel.claimant.Claim
import onnx.onnx.TensorProto

import com.stripe.agate.laws.Check._
import Onnx._

import org.scalacheck.{Prop, Properties}

import cats.implicits._

object ModelTest extends Properties("ModelTest") {

  val gen: Gen[(String, Tensor.Unknown, TensorProto)] =
    for {
      name <- Gen.identifier
      t <- genTensorU
      tp <- genTensorProtoFromTensor(name, t)
    } yield (name, t, tp)

  property("can load arbritrary TensorProto") = forAll(gen) {
    case (name, t0, tp) =>
      Model
        .loadTensor(tp)(_ => IO.raiseError(new Exception("won't happen")))
        .use {
          case (reg, t1) =>
            IO.pure(Claim(t0 == t1) && Claim(Register(name) == reg))
        }
        .unsafeRunSync
  }

  property("can load TensorProto without name") = forAll(gen) {
    case (name, t0, tp) => {
      val tp0 = tp.copy(name = None)
      val tensorType = Model.tensorType(tp0).assertInternal
      val (reg, it) = Model.loadInternalTensor(tensorType).get
      Claim(reg == None) && Claim(t0 == it)
    }
  }
}
