package com.stripe.agate.tensor

sealed abstract class TensorException(message: String, cause: Throwable = null)
    extends Exception(message, cause)

object TensorException {
  case class InsufficentBytesToLoad(
      bytesLength: Long,
      dt: DataType,
      len: Long,
      cause: Throwable = null
  ) extends TensorException(
        s"loading ${dt.typeName} values, expected at least $len but only saw $bytesLength bytes",
        cause
      )
}
