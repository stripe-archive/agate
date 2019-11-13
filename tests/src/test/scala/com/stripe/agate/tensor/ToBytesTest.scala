package com.stripe.agate
package tensor

import org.scalacheck.Properties
import org.scalacheck.Prop.{forAllNoShrink => forAll}

import java.io._
import org.typelevel.claimant.Claim

object ToBytesTest extends Properties("ToBytesTest") {
  property("long values round-trip") = forAll { (n0: Long, n1: Long) =>
    val tb = ToBytes[Long]
    val baos = new ByteArrayOutputStream(tb.size)
    tb.put(baos, n0)
    tb.put(baos, n1)
    baos.close()
    val bytes = baos.toByteArray
    val n2 = tb.read(bytes, 0)
    val n3 = tb.read(bytes, tb.size)
    val ok = n2 == n0 && n3 == n1
    if (!ok) {
      println(s"n2 (%d = %016x) = n0 (%d = %016x)".format(n2, n2, n0, n0))
      println(s"n3 (%d = %016x) = n1 (%d = %016x)".format(n3, n3, n1, n1))
      println("bytes = " + bytes.map(n => "%02x".format(n)).mkString)
    }
    Claim(n2 == n0) && Claim(n3 == n1)
  }
}
