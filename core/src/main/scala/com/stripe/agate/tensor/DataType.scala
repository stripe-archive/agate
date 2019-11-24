package com.stripe.agate.tensor

import cats.evidence.Is
import scala.reflect.ClassTag

sealed abstract class DataType(val typeName: String) {
  type Elem
  def classTag: ClassTag[Elem]
}

object DataType {
  type Aux[A] = DataType { type Elem = A }

  case object Uint8 extends DataType("uint8") {
    type Elem = Byte
    val classTag = implicitly[ClassTag[Elem]]
  }
  case object Uint16 extends DataType("uint16") {
    type Elem = Short
    val classTag = implicitly[ClassTag[Elem]]
  }
  case object Int8 extends DataType("int8") {
    type Elem = Byte
    val classTag = implicitly[ClassTag[Elem]]
  }
  case object Int16 extends DataType("int16") {
    type Elem = Short
    val classTag = implicitly[ClassTag[Elem]]
  }
  case object Int32 extends DataType("int32") {
    type Elem = Int
    val classTag = implicitly[ClassTag[Elem]]
  }
  case object Int64 extends DataType("int64") {
    type Elem = Long
    val classTag = implicitly[ClassTag[Elem]]
  }
  case object BFloat16 extends DataType("bfloat16") {
    type Elem = Short
    val classTag = implicitly[ClassTag[Short]]
  }
  case object Float16 extends DataType("float16") {
    type Elem = Short
    val classTag = implicitly[ClassTag[Short]]
  }
  case object Float32 extends DataType("float32") {
    type Elem = Float
    val classTag = implicitly[ClassTag[Elem]]
  }
  case object Float64 extends DataType("float64") {
    type Elem = Double
    val classTag = implicitly[ClassTag[Elem]]
  }

  private[this] val someRefl: Option[Is[Any, Any]] =
    Some(Is.refl[Any])

  /**
   * If two DataType values are equal, they are the same type since DataType
   * is sealed and we only have the above case object instances. If they
   * are the same, then their Elem types are the same
   */
  def maybeElemIs(dt1: DataType, dt2: DataType): Option[Is[dt1.Elem, dt2.Elem]] =
    if (dt1 == dt2) someRefl.asInstanceOf[Option[Is[dt1.Elem, dt2.Elem]]] else None
}
