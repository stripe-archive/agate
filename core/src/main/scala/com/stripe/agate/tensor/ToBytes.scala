package com.stripe.agate.tensor

import java.io.OutputStream
import java.lang.Float.{floatToRawIntBits, intBitsToFloat}
import java.lang.Double.{doubleToRawLongBits, longBitsToDouble}
import com.stripe.dagon.HMap

trait ToBytes[@specialized A] {
  def size: Int
  def read(bytes: Array[Byte], i: Int): A
  def put(os: OutputStream, n: A): Unit
}

object ToBytes {
  def apply[A](implicit ev: ToBytes[A]): ToBytes[A] = ev

  implicit val byteToBytes: ToBytes[Byte] =
    new ToBytes[Byte] {
      def size: Int = 1
      def read(bytes: Array[Byte], i: Int): Byte = bytes(i)
      def put(os: OutputStream, n: Byte): Unit = os.write(n & 0xff)
    }

  implicit val shortToBytes: ToBytes[Short] =
    new ToBytes[Short] {
      def size: Int = 2
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
      def size: Int = 4
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
      def size: Int = 8
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
      def size: Int = 4
      def read(bytes: Array[Byte], i: Int): Float =
        intBitsToFloat(tb.read(bytes, i))
      def put(os: OutputStream, n: Float): Unit =
        tb.put(os, floatToRawIntBits(n))
    }

  implicit val doubleToBytes: ToBytes[Double] =
    new ToBytes[Double] {
      val tb = ToBytes.longToBytes
      def size: Int = 8
      def read(bytes: Array[Byte], i: Int): Double =
        longBitsToDouble(tb.read(bytes, i))
      def put(os: OutputStream, n: Double): Unit =
        tb.put(os, doubleToRawLongBits(n))
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

  def forDataType(dt: DataType): ToBytes[dt.Elem] =
    forDataTypeMap(dt)
}
