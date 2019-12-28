package com.stripe.agate.tensor

import java.io.OutputStream
import java.lang.Float.{floatToRawIntBits, intBitsToFloat}
import java.lang.Double.{doubleToRawLongBits, longBitsToDouble}
import com.google.protobuf.ByteString
import com.stripe.dagon.HMap

/**
 * A type class for serializing values of type `A` into bytes.
 */
trait ToBytes[@specialized A] {

  /**
   * The strategy is used to determine whether values of type `A` are a "fixed
   * length" encoding or "variable length" encoding. In the case of fixed
   * length encodings, the same amount of bytes will always be used to
   * serialize a value. For example, all `Int` values require 4 bytes. To
   * serialize a tensor, we can just encode all the values in row-major order.
   * For types with a variable length encoding, like `ByteString`, each value
   * may require a different number of bytes. In this case, we need to be able
   * to quickly determine the size of a serialized value at runtime. This size
   * can be used when serializing a tensor to compute an index of offsets,
   * which can be serialized along with the data. The index allows for random
   * access into the serialized data.
   *
   * Note: Ideally, we'd just have 2 sub-classes of `ToBytes` to describe the
   * different strategies, but this doesn't play well with specialization and
   * pattern matching.
   */
  def strategy: ToBytes.Strategy[A]

  def read(bytes: Array[Byte], offset: Int): A
  def put(os: OutputStream, n: A): Unit

  /**
   * Returns the number of bytes required to serialize `a`.
   *
   * Note: pattern matching on strategy in a specialized context leads to
   * compile time type errors in the specialized implementations. This method
   * should be used the size is required, since this kind of indirection to get
   * things working correctly.
   */
  def size(a: A): Int = strategy.size(a)

  def isFixedLength: Boolean = strategy match {
    case ToBytes.Strategy.FixedLength(_) => true
    case _                               => false
  }
}

object ToBytes {
  def apply[A](implicit ev: ToBytes[A]): ToBytes[A] = ev

  sealed trait Strategy[A] {
    def size(value: A): Int
  }

  object Strategy {
    case class FixedLength[A](size: Int) extends Strategy[A] {
      def size(value: A): Int = size
    }
    trait VarLength[A] extends Strategy[A]
  }

  implicit val byteToBytes: ToBytes[Byte] =
    new ToBytes[Byte] {
      def strategy: Strategy[Byte] = Strategy.FixedLength(1)
      def read(bytes: Array[Byte], i: Int): Byte = bytes(i)
      def put(os: OutputStream, n: Byte): Unit = os.write(n & 0xff)
    }

  implicit val shortToBytes: ToBytes[Short] =
    new ToBytes[Short] {
      def strategy: Strategy[Short] = Strategy.FixedLength(2)
      def read(bytes: Array[Byte], i: Int): Short = {
        val b0 = (bytes(i) & 0xff)
        val b1 = (bytes(i + 1) & 0xff) << 8
        (b0 | b1).toShort
      }
      def put(os: OutputStream, n: Short): Unit = {
        os.write(n & 0xff)
        os.write((n >>> 8) & 0xff)
      }
    }

  implicit val intToBytes: ToBytes[Int] =
    new ToBytes[Int] {
      def strategy: Strategy[Int] = Strategy.FixedLength(4)
      def read(bytes: Array[Byte], i: Int): Int = {
        val b0 = (bytes(i) & 0xff)
        val b1 = (bytes(i + 1) & 0xff) << 8
        val b2 = (bytes(i + 2) & 0xff) << 16
        val b3 = (bytes(i + 3) & 0xff) << 24
        (b0 | b1 | b2 | b3)
      }
      def put(os: OutputStream, n: Int): Unit = {
        os.write(n & 0xff)
        os.write((n >>> 8) & 0xff)
        os.write((n >>> 16) & 0xff)
        os.write((n >>> 24) & 0xff)
      }
    }

  implicit val longToBytes: ToBytes[Long] =
    new ToBytes[Long] {
      def strategy: Strategy[Long] = Strategy.FixedLength(8)
      def read(bytes: Array[Byte], i: Int): Long = {
        val b0 = (bytes(i + 0) & 0XFFL) << 0L
        val b1 = (bytes(i + 1) & 0XFFL) << 8L
        val b2 = (bytes(i + 2) & 0XFFL) << 16L
        val b3 = (bytes(i + 3) & 0XFFL) << 24L
        val b4 = (bytes(i + 4) & 0XFFL) << 32L
        val b5 = (bytes(i + 5) & 0XFFL) << 40L
        val b6 = (bytes(i + 6) & 0XFFL) << 48L
        val b7 = (bytes(i + 7) & 0XFFL) << 56L
        (b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7)
      }
      def put(os: OutputStream, n: Long): Unit = {
        os.write(((n >>> 0) & 0XFFL).toInt)
        os.write(((n >>> 8) & 0XFFL).toInt)
        os.write(((n >>> 16) & 0XFFL).toInt)
        os.write(((n >>> 24) & 0XFFL).toInt)
        os.write(((n >>> 32) & 0XFFL).toInt)
        os.write(((n >>> 40) & 0XFFL).toInt)
        os.write(((n >>> 48) & 0XFFL).toInt)
        os.write(((n >>> 56) & 0XFFL).toInt)
      }
    }

  implicit val floatToBytes: ToBytes[Float] =
    new ToBytes[Float] {
      val tb = ToBytes.intToBytes
      def strategy: Strategy[Float] = Strategy.FixedLength(4)
      def read(bytes: Array[Byte], i: Int): Float =
        intBitsToFloat(tb.read(bytes, i))
      def put(os: OutputStream, n: Float): Unit =
        tb.put(os, floatToRawIntBits(n))
    }

  implicit val doubleToBytes: ToBytes[Double] =
    new ToBytes[Double] {
      val tb = ToBytes.longToBytes
      def strategy: Strategy[Double] = Strategy.FixedLength(8)
      def read(bytes: Array[Byte], i: Int): Double =
        longBitsToDouble(tb.read(bytes, i))
      def put(os: OutputStream, n: Double): Unit =
        tb.put(os, doubleToRawLongBits(n))
    }

  // This encodes a ByteString by encoding its length as a varint first,
  // followed by the data themselves.
  implicit val byteStringToBytes: ToBytes[ByteString] =
    new ToBytes[ByteString] {
      val strategy: Strategy.VarLength[ByteString] =
        new Strategy.VarLength[ByteString] {
          def size(a: ByteString): Int = {
            val len = a.size
            // In the case of 0, we still use a single byte to store the 0.
            val width = math.max(32 - java.lang.Integer.numberOfLeadingZeros(len), 1)
            val varIntSize = (width + 7 - 1) / 7 // math.ceil(width / 7).toInt
            varIntSize + len
          }
        }

      def read(bytes: Array[Byte], offset: Int): ByteString = {
        var len = 0
        var s = 0
        var n = bytes(offset)
        var i = 1
        while (n < 0) {
          len = len | ((n & 0x7F) << s)
          s += 7
          n = bytes(offset + i)
          i += 1
        }
        len = len | (n << s)
        ByteString.copyFrom(bytes, offset + i, len)
      }

      def put(os: OutputStream, a: ByteString): Unit = {
        var n = a.size
        while ((n & ~0x7F) != 0) {
          os.write((n & 0x7F) | 0x80)
          n = n >>> 7
        }
        os.write(n)
        a.writeTo(os)
      }
    }

  implicit val booleanToBytes: ToBytes[Boolean] =
    new ToBytes[Boolean] {
      def strategy: Strategy[Boolean] = Strategy.FixedLength(1)
      def read(bytes: Array[Byte], i: Int): Boolean = bytes(i) != 0
      def put(os: OutputStream, n: Boolean): Unit =
        os.write(if (n) 1 else 0)
    }

  val forDataTypeMap: HMap[DataType.Aux, ToBytes] =
    HMap
      .empty[DataType.Aux, ToBytes]
      .updated(DataType.Uint8, byteToBytes)
      .updated(DataType.Uint16, shortToBytes)
      .updated(DataType.Int8, byteToBytes)
      .updated(DataType.Int16, shortToBytes)
      .updated(DataType.Int32, intToBytes)
      .updated(DataType.Int64, longToBytes)
      .updated(DataType.BFloat16, shortToBytes)
      .updated(DataType.Float16, shortToBytes)
      .updated(DataType.Float32, floatToBytes)
      .updated(DataType.Float64, doubleToBytes)
      .updated(DataType.String, byteStringToBytes)
      .updated(DataType.Bool, booleanToBytes)

  def forDataType(dt: DataType): ToBytes[dt.Elem] =
    forDataTypeMap(dt)
}
