package com.stripe.agate.tensor

import com.google.protobuf.ByteString
import org.scalacheck.Properties
import org.typelevel.claimant.Claim

class OnnxShowTest extends Properties("OnnxShowTest") {
  property("String instance encodes bytes succinctly") = Claim(
    OnnxShow.showString
      .show(ByteString.copyFrom(Array[Byte](0, Byte.MaxValue, -1, Byte.MinValue))) == "'\\x00\\x7f\\xff\\x80'"
  )
}
