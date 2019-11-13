package com.stripe.agate
package tensor

import scala.reflect.ClassTag

import org.scalacheck.{Arbitrary, Gen, Prop, Properties}
import org.scalacheck.Arbitrary.arbitrary
import org.typelevel.claimant.Claim

import com.stripe.agate.laws.Check._
import Prop.{forAllNoShrink => forAll}
import Shape.Dim
// import TestImplicits._

object StorageTest extends Properties("StorageTest") {
  def chooseSkewSmall(from: Int, to: Int): Gen[Int] =
    Gen.choose(from, to).flatMap(Gen.choose(from, _))

  def genChunkedAndUnchunked[A: Arbitrary: ClassTag]: Gen[(Storage.Chunked[A], Storage[A], Int)] =
    for {
      size <- Gen.choose(0, 1000)
      data <- Gen.listOfN(size, arbitrary[A]).map(_.toArray)
      offset <- chooseSkewSmall(0, size)
      chunkSize <- Gen.choose(1, math.max(1, size))
    } yield {
      val storage = Storage.ArrayStorage(data, offset)
      val chunks: Array[Storage[A]] = Array.range(0, size, chunkSize).map { i =>
        Storage.ArrayStorage(data.slice(i, i + chunkSize), 0)
      }
      val chunked = Storage.Chunked(chunks, chunkSize, offset)
      (chunked, storage, size - offset)
    }

  case class WriteIntoArgs(offset: Long, dim: Dim, len: Long)

  // Generate "safe" arguments for writeInto
  def genWriteIntoArgs(sourceSize: Int): Gen[WriteIntoArgs] =
    for {
      targetOffset <- chooseSkewSmall(0, 100)
      sourceOffset <- Gen.choose(0, sourceSize)
      maxLength = sourceSize - sourceOffset
      stride <- chooseSkewSmall(0, maxLength)
      len <- if (stride == 0) Gen.choose(0, maxLength)
      else Gen.choose(0, (maxLength + stride - 1) / stride)
    } yield WriteIntoArgs(targetOffset, Dim(sourceOffset, stride), len)

  def writeIntoAndReturn[A: ClassTag](
      source: Storage[A],
      targetOffset: Long,
      dim: Dim,
      len: Long
  ): Vector[A] = {
    val data = new Array[A]((targetOffset + len).toInt)
    val target = WritableStorage.ArrayStorage(data, 0)
    source.writeInto(target, targetOffset, dim, len)
    data.toVector
  }

  property("Chunked writeInto matches unchunked") = {
    val genExample: Gen[(Storage.Chunked[Float], Storage[Float], WriteIntoArgs)] = for {
      (chunked, unchunked, sourceSize) <- genChunkedAndUnchunked[Float]
      args <- genWriteIntoArgs(sourceSize)
    } yield (chunked, unchunked, args)

    forAll(genExample) {
      case (chunked, unchunked, WriteIntoArgs(offset, dim, len)) =>
        val actual = writeIntoAndReturn(chunked, offset, dim, len)
        val expected = writeIntoAndReturn(unchunked, offset, dim, len)
        Claim(actual == expected)
    }
  }

  property("loadMapped loads serialized storage") = {
    val genExample: Gen[(Array[Float], Int)] = for {
      count <- Gen.choose(0, 100)
      data <- Gen.listOfN(count, arbitrary[Float])
      // Chunks must contain at least 1 whole value (eg 4 bytes for float).
      chunkSize <- chooseSkewSmall(4, math.max(count * 4, 4))
    } yield (data.toArray, chunkSize)

    forAll(genExample) {
      case (data, chunkSize) =>
        import java.io.{File, FileOutputStream}
        import cats.effect.{IO, Resource}
        val file = File.createTempFile("agate", ".storage")
        try {
          val actual =
            Resource
              .fromAutoCloseable(IO.delay(new FileOutputStream(file)))
              .use { os =>
                IO.delay(
                  Storage.ArrayStorage(data, 0).writeIntoStream(os, Dim(0L, 1L), data.length)
                )
              }
              .flatMap { _ =>
                Storage
                  .loadMapped(DataType.Float32, file.toPath, data.length, chunkSize)
                  .use { storage =>
                    IO.delay(writeIntoAndReturn(storage, 0L, Dim(0L, 1L), data.length))
                  }
              }
              .unsafeRunSync()
          Claim(actual == data.toVector)
        } finally {
          file.delete()
          ()
        }
    }
  }
}
