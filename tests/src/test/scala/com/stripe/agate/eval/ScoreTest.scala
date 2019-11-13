package com.stripe.agate.eval

import com.stripe.agate.laws.Check
import com.stripe.agate.tensor.DataType
import org.scalacheck.{Gen, Prop, Properties}

object ScoreTest extends Properties("ScoreTest") {
  val argGen = Gen
    .choose(1, 10)
    .flatMap(Gen.listOfN(_, Gen.zip(Gen.identifier, Check.genTensor(DataType.Int64))))
  property("can parse args") = Prop.forAll(Gen.identifier, argGen) { (path, args) =>
    val argStrings = path :: (args.map {
      case (reg, ten) => reg + "=" + ten.toDoc.renderWideStream.mkString
    })
    Score.command.parse(argStrings).isRight
  }
}
