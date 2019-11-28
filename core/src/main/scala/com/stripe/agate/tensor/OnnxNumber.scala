package com.stripe.agate.tensor

import cats.evidence.Is
import cats.Functor
import cats.arrow.FunctionK
import com.stripe.agate.tensor.{BFloat16 => B16, Float16 => F16}
import com.stripe.dagon.HMap
import java.lang.Float.{floatToRawIntBits, intBitsToFloat}
import java.lang.Math
import scala.util.{Failure, Success, Try}

/*
 * TODO, we need to think about the element types a bit more.
 *
 * e.g:
 * 0. signed ints, int8, int16, int32, int64
 * 1. floating point numbers: float16, float32, float64
 * 2. unsigned numbers: uint8, uint16, (still form a semiring so we can GEMM them)
 * 3. complex numbers: complex64, complex128
 * 4. string (not a number but can be cast by parsing)
 * 5. bool (not really a number at all)
 *
 * So,
 * real = signed int | floating | unsigned int
 * number = real | complex
 * signed real = signed int | floating | complex
 */

/**
 * represent Onnx's notion of a number
 */
sealed trait OnnxNumber[@specialized(Byte, Short, Int, Long, Float, Double) A] {
  type Data <: DataType.Aux[A]
  val dataType: Data
  def typeName: String = dataType.typeName

  def render(x: A): String

  def zero: A
  def one: A
  def times(left: A, right: A): A
  def div(x: A, y: A): A
  def plus(left: A, right: A): A
  def minus(left: A, right: A): A

  def toFloat(a: A): Float
  def fromFloat(f: Float): A
  def toDouble(a: A): Double
  def fromDouble(d: Double): A
  def toLong(a: A): Long
  def fromLong(n: Long): A

  def max(x: A, y: A): A
  def minimum: A

  def timesFloat(f: Float, a: A): A

  // this fails on 2.11 due to a bug with specialization
  // final def cast(dt: DataType): A => dt.Elem = {
  //   val dest = OnnxNumber.forDataType(dt)
  //   if (dest.isInstanceOf[OnnxIntegral[_]]) { (a: A) =>
  //     dest.fromLong(toLong(a))
  //   } else { (a: A) =>
  //     dest.fromDouble(toDouble(a))
  //   }
  // }

  override def toString: String = getClass.getName + s"(${dataType.typeName})"
}

/**
 * Onnx has several types of Integral numbers
 */
sealed trait OnnxIntegral[@specialized(Byte, Short, Int, Long) A] extends OnnxNumber[A] {
  // return a negative number, or a number < size
  def validIntIndex(a: A, size: Long): Int
}

/**
 * Onnx has several types of Integral numbers
 */
sealed trait OnnxFloating[@specialized(Float, Double) A] extends OnnxNumber[A] {
  def relu: A => A
  def sigmoid: A => A
  def tanh: A => A
  def exp(x: A): A
  def log(x: A): A
  def invSqrt(x: A, eps: Double): A
  def sqrt(x: A): A
  def batchNormalize(
      mean: A,
      variance: A,
      scale: A,
      bias: A,
      epsilon: Float
  ): A => A
}

object OnnxNumber {
  type Aux[D <: DataType, A] = OnnxNumber[A] { type Data = D }

  def toIntegral[A](on: OnnxNumber[A]): Try[OnnxIntegral[A]] =
    on match {
      case in: OnnxIntegral[A] => Success(in)
      case other               => Failure(new Exception(s"$other is not integral"))
    }

  def toFloating[A](on: OnnxNumber[A]): Try[OnnxFloating[A]] =
    on match {
      case fl: OnnxFloating[A] => Success(fl)
      case other               => Failure(new Exception(s"$other is not floating"))
    }

  val Uint8: OnnxIntegral[Byte] =
    new OnnxIntegral[Byte] {
      type Data = DataType.Uint8.type
      def render(x: Byte): String = (x & 0xff).toString
      val dataType = DataType.Uint8
      val zero: Byte = 0.toByte
      val one: Byte = 1.toByte
      @inline private def lift(b: Byte): Int = b.toInt & 0xff
      def times(left: Byte, right: Byte): Byte = (left * right).toByte
      def div(left: Byte, right: Byte): Byte = (lift(left) / lift(right)).toByte
      def plus(left: Byte, right: Byte): Byte = (left + right).toByte
      def minus(left: Byte, right: Byte): Byte = (left - right).toByte
      def max(left: Byte, right: Byte): Byte =
        java.lang.Integer.max(left & 0xff, right & 0xff).toByte
      def minimum: Byte = 0.toByte
      def toFloat(a: Byte): Float = (a & 0xff).toFloat
      def fromFloat(f: Float): Byte = (f.toInt & 0xff).toByte
      def toDouble(a: Byte): Double = (a & 0xff).toDouble
      def fromDouble(d: Double): Byte = (d.toInt & 0xff).toByte
      def toLong(a: Byte): Long = a & 0XFFL
      def fromLong(n: Long): Byte = (n & 0XFFL).toByte

      def timesFloat(f: Float, a: Byte): Byte =
        fromDouble((toDouble(a) * f.toDouble))
      def validIntIndex(a: Byte, size: Long): Int = {
        val n = a & 0xff
        if (n < size) n.toInt else -1
      }
    }

  val Uint16: OnnxIntegral[Short] =
    new OnnxIntegral[Short] {
      type Data = DataType.Uint16.type
      def render(x: Short): String = (x & 0xffff).toString
      val dataType = DataType.Uint16
      val zero: Short = 0.toShort
      val one: Short = 1.toShort
      @inline private def lift(b: Short): Int = b.toInt & 0xffff
      def times(left: Short, right: Short): Short = (left * right).toShort
      def div(left: Short, right: Short): Short = (lift(left) / lift(right)).toShort
      def plus(left: Short, right: Short): Short = (left + right).toShort
      def minus(left: Short, right: Short): Short = (left - right).toShort
      def max(left: Short, right: Short): Short =
        java.lang.Integer.max(left & 0xffff, right & 0xffff).toShort
      def minimum: Short = 0.toShort
      def toFloat(a: Short): Float = (a & 0xffff).toFloat
      def fromFloat(f: Float): Short = (f.toInt & 0xffff).toShort
      def toDouble(a: Short): Double = (a & 0xffff).toDouble
      def fromDouble(d: Double): Short = (d.toInt & 0xffff).toShort
      def toLong(a: Short): Long = a & 0XFFFFL
      def fromLong(n: Long): Short = (n & 0XFFFFL).toShort

      def timesFloat(f: Float, a: Short): Short =
        fromDouble((toDouble(a) * f.toDouble))
      def validIntIndex(a: Short, size: Long): Int = {
        val n = a & 0xffff
        if (n < size) n.toInt else -1
      }
    }

  val Int8: OnnxIntegral[Byte] =
    new OnnxIntegral[Byte] {
      type Data = DataType.Int8.type
      val dataType = DataType.Int8
      def render(x: Byte): String = x.toString
      val zero: Byte = 0.toByte
      val one: Byte = 1.toByte
      def times(left: Byte, right: Byte): Byte = (left * right).toByte
      def div(left: Byte, right: Byte): Byte = (left / right).toByte
      def plus(left: Byte, right: Byte): Byte = (left + right).toByte
      def minus(left: Byte, right: Byte): Byte = (left - right).toByte
      def max(left: Byte, right: Byte): Byte =
        java.lang.Integer.max(left.toInt, right.toInt).toByte
      def minimum: Byte = Byte.MinValue
      def toFloat(a: Byte): Float = a.toFloat
      def fromFloat(f: Float): Byte = f.toByte
      def toDouble(a: Byte): Double = a.toDouble
      def fromDouble(d: Double): Byte = d.toByte
      def toLong(a: Byte): Long = a.toLong
      def fromLong(d: Long): Byte = d.toByte
      def timesFloat(f: Float, a: Byte): Byte =
        (a.toDouble * f.toDouble).toByte
      def validIntIndex(a: Byte, size: Long): Int =
        if (a < size) a.toInt else -1
    }

  val Int16: OnnxIntegral[Short] =
    new OnnxIntegral[Short] {
      type Data = DataType.Int16.type
      val dataType = DataType.Int16
      def render(x: Short): String = x.toString
      val zero: Short = 0.toShort
      val one: Short = 1.toShort
      def times(left: Short, right: Short): Short = (left * right).toShort
      def div(left: Short, right: Short): Short = (left / right).toShort
      def plus(left: Short, right: Short): Short = (left + right).toShort
      def minus(left: Short, right: Short): Short = (left - right).toShort
      def max(left: Short, right: Short): Short =
        java.lang.Integer.max(left.toInt, right.toInt).toShort
      def minimum: Short = Short.MinValue
      def toFloat(a: Short): Float = a.toFloat
      def fromFloat(f: Float): Short = f.toShort
      def toDouble(a: Short): Double = a.toDouble
      def fromDouble(d: Double): Short = d.toShort
      def toLong(a: Short): Long = a.toLong
      def fromLong(d: Long): Short = d.toShort
      def timesFloat(f: Float, a: Short): Short =
        (a.toDouble * f.toDouble).toShort
      def validIntIndex(a: Short, size: Long): Int =
        if (a < size) a.toInt else -1
    }

  val Int32: OnnxIntegral[Int] =
    new OnnxIntegral[Int] {
      type Data = DataType.Int32.type
      val dataType = DataType.Int32
      def render(x: Int): String = x.toString
      def zero: Int = 0
      def one: Int = 1
      def times(left: Int, right: Int): Int = left * right
      def div(left: Int, right: Int): Int = left / right
      def plus(left: Int, right: Int): Int = left + right
      def minus(left: Int, right: Int): Int = left - right
      def max(left: Int, right: Int): Int =
        java.lang.Integer.max(left, right)
      def minimum: Int = Int.MinValue
      def toFloat(a: Int): Float = a.toFloat
      def fromFloat(f: Float): Int = f.toInt
      def toDouble(a: Int): Double = a.toDouble
      def fromDouble(d: Double): Int = d.toInt
      def toLong(a: Int): Long = a.toLong
      def fromLong(d: Long): Int = d.toInt
      def timesFloat(f: Float, a: Int): Int =
        (a.toDouble * f.toDouble).toInt
      def validIntIndex(a: Int, size: Long): Int =
        if (a < size) a else -1
    }

  val Int64: OnnxIntegral[Long] =
    new OnnxIntegral[Long] {
      type Data = DataType.Int64.type
      val dataType = DataType.Int64
      def render(x: Long): String = x.toString
      def zero: Long = 0L
      def one: Long = 1L
      def times(left: Long, right: Long): Long = (left * right)
      def div(left: Long, right: Long): Long = (left / right)
      def plus(left: Long, right: Long): Long = (left + right)
      def minus(left: Long, right: Long): Long = (left - right)
      def max(left: Long, right: Long): Long =
        java.lang.Long.max(left, right)
      def minimum: Long = Long.MinValue
      def toFloat(a: Long): Float = a.toFloat
      def fromFloat(f: Float): Long = f.toLong
      def toDouble(a: Long): Double = a.toDouble
      def fromDouble(d: Double): Long = d.toLong
      def timesFloat(f: Float, a: Long): Long =
        (a.toDouble * f.toDouble).toLong
      def toLong(a: Long): Long = a
      def fromLong(n: Long): Long = n
      def validIntIndex(a: Long, size: Long): Int =
        if (Int.MinValue <= a && a < size) a.toInt else -1
    }

  val BFloat16: OnnxFloating[Short] =
    new OnnxFloating[Short] {
      type Data = DataType.BFloat16.type
      val dataType = DataType.BFloat16

      def render(x: Short): String = lift(x).toString

      val zero: Short = B16.Zero.raw
      val one: Short = B16.One.raw

      @inline private def lift(n: Short): BFloat16 = new BFloat16(n)

      def times(left: Short, right: Short): Short =
        (lift(left) * lift(right)).raw
      def plus(left: Short, right: Short): Short =
        (lift(left) + lift(right)).raw
      def minus(left: Short, right: Short): Short =
        (lift(left) - lift(right)).raw
      def max(left: Short, right: Short): Short =
        B16.max(lift(left), lift(right)).raw

      val minimum: Short = B16.NegativeInfinity.raw

      def toFloat(a: Short): Float = lift(a).toFloat
      def fromFloat(f: Float): Short = B16.fromFloat(f).raw
      def toDouble(a: Short): Double = lift(a).toDouble
      def fromDouble(d: Double): Short = B16.fromDouble(d).raw
      def toLong(a: Short): Long = lift(a).toFloat.toLong
      def fromLong(d: Long): Short = B16.fromFloat(d.toFloat).raw

      def timesFloat(f: Float, a: Short): Short =
        B16.fromFloat(f * lift(a).toFloat).raw

      def relu: Short => Short = { (a: Short) =>
        B16.max(B16.Zero, lift(a)).raw
      }
      def sigmoid: Short => Short = { (a: Short) =>
        B16.fromDouble(1.0 / (1.0 + Math.exp(-lift(a).toDouble))).raw
      }
      def tanh: Short => Short = { (x: Short) =>
        val e2x = Math.exp(lift(x).toDouble * -2.0)
        B16.fromDouble((1.0 - e2x) / (1.0 + e2x)).raw
      }

      def exp(x: Short): Short =
        B16.fromDouble(Math.exp(lift(x).toDouble)).raw
      def log(x: Short): Short =
        B16.fromDouble(Math.log(lift(x).toDouble)).raw
      def div(x: Short, y: Short): Short =
        (lift(x) / lift(y)).raw
      def invSqrt(x: Short, eps: Double): Short =
        B16.fromDouble(1.0 / Math.sqrt(lift(x).toDouble + eps)).raw
      def sqrt(x: Short): Short =
        B16.fromDouble(Math.sqrt(lift(x).toDouble)).raw
      def batchNormalize(
          mean: Short,
          variance: Short,
          scale: Short,
          bias: Short,
          epsilon: Float
      ): Short => Short = {
        val scale1 = lift(scale).toDouble / Math.sqrt(lift(variance).toDouble + epsilon.toDouble)
        val const = lift(bias).toDouble - (lift(mean).toDouble * scale1)
        (x: Short) => B16.fromDouble(lift(x).toDouble * scale1 + const).raw
      }
    }

  val Float16: OnnxFloating[Short] =
    new OnnxFloating[Short] {
      type Data = DataType.Float16.type
      val dataType = DataType.Float16

      def render(x: Short): String = lift(x).toString

      val zero: Short = F16.Zero.raw
      val one: Short = F16.One.raw

      @inline private def lift(n: Short): Float16 = new Float16(n)

      def times(left: Short, right: Short): Short =
        (lift(left) * lift(right)).raw
      def plus(left: Short, right: Short): Short =
        (lift(left) + lift(right)).raw
      def minus(left: Short, right: Short): Short =
        (lift(left) - lift(right)).raw
      def max(left: Short, right: Short): Short =
        F16.max(lift(left), lift(right)).raw

      val minimum: Short = F16.NegativeInfinity.raw

      def toFloat(a: Short): Float = lift(a).toFloat
      def fromFloat(f: Float): Short = F16.fromFloat(f).raw
      def toDouble(a: Short): Double = lift(a).toDouble
      def fromDouble(d: Double): Short = F16.fromDouble(d).raw
      def toLong(a: Short): Long = lift(a).toFloat.toLong
      def fromLong(d: Long): Short = F16.fromFloat(d.toFloat).raw

      def timesFloat(f: Float, a: Short): Short =
        F16.fromFloat(f * lift(a).toFloat).raw

      def relu: Short => Short = { (a: Short) =>
        F16.max(F16.Zero, lift(a)).raw
      }
      def sigmoid: Short => Short = { (a: Short) =>
        F16.fromDouble(1.0 / (1.0 + Math.exp(-lift(a).toDouble))).raw
      }
      def tanh: Short => Short = { (x: Short) =>
        val e2x = Math.exp(lift(x).toDouble * -2.0)
        F16.fromDouble((1.0 - e2x) / (1.0 + e2x)).raw
      }

      def exp(x: Short): Short =
        F16.fromDouble(Math.exp(lift(x).toDouble)).raw
      def log(x: Short): Short =
        F16.fromDouble(Math.log(lift(x).toDouble)).raw
      def div(x: Short, y: Short): Short =
        (lift(x) / lift(y)).raw
      def invSqrt(x: Short, eps: Double): Short =
        F16.fromDouble(1.0 / Math.sqrt(lift(x).toDouble + eps)).raw
      def sqrt(x: Short): Short =
        F16.fromDouble(Math.sqrt(lift(x).toDouble)).raw
      def batchNormalize(
          mean: Short,
          variance: Short,
          scale: Short,
          bias: Short,
          epsilon: Float
      ): Short => Short = {
        val scale1 = lift(scale).toDouble / Math.sqrt(lift(variance).toDouble + epsilon.toDouble)
        val const = lift(bias).toDouble - (lift(mean).toDouble * scale1)
        (x: Short) => F16.fromDouble(lift(x).toDouble * scale1 + const).raw
      }
    }

  val Float32: OnnxFloating[Float] =
    new OnnxFloating[Float] {
      type Data = DataType.Float32.type
      val dataType = DataType.Float32
      def render(x: Float): String = x.toString
      def zero: Float = 0f
      def one: Float = 1f
      def times(left: Float, right: Float): Float =
        left * right
      def plus(left: Float, right: Float): Float =
        left + right
      def minus(left: Float, right: Float): Float =
        left - right
      def max(left: Float, right: Float): Float =
        java.lang.Float.max(left, right)
      def minimum: Float = Float.NegativeInfinity
      def toFloat(a: Float): Float = a
      def fromFloat(f: Float): Float = f
      def toDouble(a: Float): Double = a.toDouble
      def fromDouble(d: Double): Float = d.toFloat
      def toLong(a: Float): Long = a.toLong
      def fromLong(d: Long): Float = d.toFloat

      def timesFloat(f: Float, a: Float): Float =
        f * a

      def relu: Float => Float = { (a: Float) =>
        java.lang.Float.max(0f, a)
      }
      def sigmoid: Float => Float = { (a: Float) =>
        (1.0 / (1.0 + Math.exp(-a.toDouble))).toFloat
      }
      def tanh: Float => Float = { (x: Float) =>
        val e2x = Math.exp(x.toDouble * -2.0)
        ((1.0 - e2x) / (1.0 + e2x)).toFloat
      }

      def exp(x: Float): Float =
        Math.exp(x.toDouble).toFloat
      def log(x: Float): Float =
        Math.log(x.toDouble).toFloat
      def div(x: Float, y: Float): Float = x / y
      def invSqrt(x: Float, eps: Double): Float =
        (1.0 / Math.sqrt(x.toDouble + eps)).toFloat
      def sqrt(x: Float): Float = Math.sqrt(x.toDouble).toFloat
      def batchNormalize(
          mean: Float,
          variance: Float,
          scale: Float,
          bias: Float,
          epsilon: Float
      ): Float => Float = {
        val scale1 = scale.toDouble / Math.sqrt(variance.toDouble + epsilon.toDouble)
        val const = bias.toDouble - (mean.toDouble * scale1)
        (x: Float) => (x * scale1 + const).toFloat
      }
    }

  val Float64: OnnxFloating[Double] =
    new OnnxFloating[Double] {
      type Data = DataType.Float64.type
      val dataType = DataType.Float64
      def render(x: Double): String = x.toString
      def zero: Double = 0.0
      def one: Double = 1.0
      def times(left: Double, right: Double): Double =
        left * right
      def plus(left: Double, right: Double): Double =
        left + right
      def minus(left: Double, right: Double): Double =
        left - right
      def max(left: Double, right: Double): Double =
        java.lang.Double.max(left, right)
      def minimum: Double = Double.NegativeInfinity
      def toFloat(a: Double): Float = a.toFloat
      def fromFloat(f: Float): Double = f.toDouble
      def toDouble(a: Double): Double = a
      def fromDouble(d: Double): Double = d
      def toLong(a: Double): Long = a.toLong
      def fromLong(d: Long): Double = d.toDouble

      def timesFloat(f: Float, a: Double): Double =
        f * a

      def relu: Double => Double = { (a: Double) =>
        java.lang.Double.max(0d, a)
      }
      def sigmoid: Double => Double = { (a: Double) =>
        (1.0 / (1.0 + Math.exp(-a)))
      }
      def tanh: Double => Double = { (x: Double) =>
        val e2x = Math.exp(x * -2.0)
        (1.0 - e2x) / (1.0 + e2x)
      }
      def exp(x: Double): Double =
        Math.exp(x)
      def log(x: Double): Double =
        Math.log(x)
      def div(x: Double, y: Double): Double = x / y
      def invSqrt(x: Double, eps: Double): Double =
        (1.0 / Math.sqrt(x + eps))
      def sqrt(x: Double): Double = Math.sqrt(x)
      def batchNormalize(
          mean: Double,
          variance: Double,
          scale: Double,
          bias: Double,
          epsilon: Float
      ): Double => Double = {
        val scale1 = scale / Math.sqrt(variance + epsilon.toDouble)
        val const = bias - (mean * scale1)
        (x: Double) => (x * scale1 + const)
      }
    }

  def forDataType(dt: DataType): OnnxNumber[dt.Elem] =
    forDataTypeMap.get(dt: DataType.Aux[dt.Elem]) match {
      case Some(num) => num
      case None      => throw new IllegalArgumentException(s"not a numeric datatype=$dt")
    }

  private[this] val forDataTypeMap: HMap[DataType.Aux, OnnxNumber] =
    HMap
      .empty[DataType.Aux, OnnxNumber]
      .updated(DataType.Uint8, OnnxNumber.Uint8)
      .updated(DataType.Uint16, OnnxNumber.Uint16)
      .updated(DataType.Int8, OnnxNumber.Int8)
      .updated(DataType.Int16, OnnxNumber.Int16)
      .updated(DataType.Int32, OnnxNumber.Int32)
      .updated(DataType.Int64, OnnxNumber.Int64)
      .updated(DataType.BFloat16, OnnxNumber.BFloat16)
      .updated(DataType.Float16, OnnxNumber.Float16)
      .updated(DataType.Float32, OnnxNumber.Float32)
      .updated(DataType.Float64, OnnxNumber.Float64)

  def cast(from: DataType, to: DataType): from.Elem => to.Elem = {
    val onF = forDataType(from)
    val onT = forDataType(to)

    if (onT.isInstanceOf[OnnxIntegral[_]]) { (a: from.Elem) =>
      onT.fromLong(onF.toLong(a))
    } else { (a: from.Elem) =>
      onT.fromDouble(onF.toDouble(a))
    }
  }
}
