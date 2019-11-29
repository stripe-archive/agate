package com.stripe.agate.tensor

import cats.Show
import com.google.protobuf.ByteString
import com.stripe.dagon.HMap

object OnnxShow {
  val showUint8: Show[Byte] = Show.show(n => (n & 0xff).toString)
  val showUint16: Show[Short] = Show.show(n => (n & 0xffff).toString)
  val showInt8: Show[Byte] = Show.fromToString
  val showInt16: Show[Short] = Show.fromToString
  val showInt32: Show[Int] = Show.fromToString
  val showInt64: Show[Long] = Show.fromToString
  val showBFloat16: Show[Short] = Show.show(n => new BFloat16(n).toString)
  val showFloat16: Show[Short] = Show.show(n => new Float16(n).toString)
  val showFloat32: Show[Float] = Show.fromToString
  val showFloat64: Show[Double] = Show.fromToString
  val showString: Show[ByteString] = Show.show { s =>
    // Renders a Python-style bytes string.
    val bldr = new StringBuilder("'")
    val it = s.iterator()
    while (it.hasNext) {
      val c = it.nextByte().toInt
      if ((c >= 32 && c <= 126) && (c != '\''.toInt) && (c != '\\'.toInt)) {
        bldr.append(c.toChar)
      } else {
        bldr.append(f"\\x$c%02x")
      }
    }
    bldr.append("'")
    bldr.toString
  }
  val showBool: Show[Boolean] = Show.fromToString

  def forDataType(dt: DataType): Show[dt.Elem] =
    forDataTypeMap(dt)

  private[this] val forDataTypeMap: HMap[DataType.Aux, Show] =
    HMap
      .empty[DataType.Aux, Show]
      .updated(DataType.Uint8, showUint8)
      .updated(DataType.Uint16, showUint16)
      .updated(DataType.Int8, showInt8)
      .updated(DataType.Int16, showInt16)
      .updated(DataType.Int32, showInt32)
      .updated(DataType.Int64, showInt64)
      .updated(DataType.BFloat16, showBFloat16)
      .updated(DataType.Float16, showFloat16)
      .updated(DataType.Float32, showFloat32)
      .updated(DataType.Float64, showFloat64)
      .updated(DataType.String, showString)
      .updated(DataType.Bool, showBool)
}
