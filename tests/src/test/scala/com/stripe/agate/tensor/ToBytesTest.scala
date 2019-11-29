package com.stripe.agate
package tensor

import com.google.protobuf.ByteString
import org.scalacheck.{Arbitrary, Gen, Properties}
import org.scalacheck.Prop.{forAllNoShrink => forAll}

import java.io._
import org.typelevel.claimant.Claim

object ToBytesTest extends Properties("ToBytesTest") {
  property("long values round-trip") = forAll { (n0: Long, n1: Long) =>
    val tb = ToBytes[Long]
    val baos = new ByteArrayOutputStream()
    tb.put(baos, n0)
    tb.put(baos, n1)
    baos.close()
    val bytes = baos.toByteArray
    val n2 = tb.read(bytes, 0)
    val n3 = tb.read(bytes, tb.size(n0))
    val ok = n2 == n0 && n3 == n1
    if (!ok) {
      println(s"n2 (%d = %016x) = n0 (%d = %016x)".format(n2, n2, n0, n0))
      println(s"n3 (%d = %016x) = n1 (%d = %016x)".format(n3, n3, n1, n1))
      println("bytes = " + bytes.map(n => "%02x".format(n)).mkString)
    }
    Claim(n2 == n0) && Claim(n3 == n1)
  }

  implicit val arbByteString: Arbitrary[ByteString] =
    Arbitrary(Gen.listOf(Arbitrary.arbitrary[Byte]).map { bs =>
      ByteString.copyFrom(bs.toArray)
    })

  property("ByteString values round-trip") = forAll { (s0: ByteString, s1: ByteString) =>
    val tb = ToBytes[ByteString]
    val baos = new ByteArrayOutputStream()
    tb.put(baos, s0)
    tb.put(baos, s1)
    baos.close()
    val bytes = baos.toByteArray
    val s2 = tb.read(bytes, 0)
    val s3 = tb.read(bytes, tb.size(s0))
    val expectedSize = tb.size(s0) + tb.size(s1)
    Claim(bytes.length == expectedSize) && Claim(s2 == s0) && Claim(s3 == s1)
  }
}
