package com.stripe.agate.tensor

sealed abstract class TensorException(message: String) extends Exception(message)

object TensorException {
  case class InsufficentBytesToLoad(bytesLength: Long, dt: DataType, len: Long)
      extends TensorException(
        s"loading ${dt.typeName} values, expected $len but only saw $bytesLength bytes"
      )
}
