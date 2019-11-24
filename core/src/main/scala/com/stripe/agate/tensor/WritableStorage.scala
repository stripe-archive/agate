package com.stripe.agate.tensor

sealed trait WritableStorage[@specialized A] {
  def apply(idx: Long): A
  def writeAt(idx: Long, f: A): Unit
  def toStorage(implicit alloc: StorageAllocator[A]): Storage[A]
  def toArrayStorage(implicit alloc: StorageAllocator[A]): Storage.ArrayStorage[A]
  def writeFrom(data: Array[A], fromOffset: Int, toOffset: Int, stride: Int, len: Int): Unit
}

object WritableStorage {
  final case class ArrayStorage[@specialized A](ary: Array[A], offset: Int)
      extends WritableStorage[A] {
    final def apply(idx: Long): A =
      ary(idx.toInt + offset)

    final def writeAt(idx: Long, f: A): Unit =
      ary(idx.toInt + offset) = f

    final def writeFrom(
        data: Array[A],
        fromOffset: Int,
        toOffset: Int,
        stride: Int,
        len: Int
    ): Unit = {
      val physicalOffset = offset + toOffset
      if (stride == 0) {
        val const = data(fromOffset)
        var i = 0
        while (i < len) {
          ary(physicalOffset + i) = const
          i += 1
        }
      } else if (stride == 1) {
        System.arraycopy(data, fromOffset, ary, physicalOffset, len)
      } else {
        var i = 0
        var j = fromOffset
        while (i < len) {
          ary(physicalOffset + i) = data(j)
          i += 1
          j += stride
        }
      }
    }

    final def toArrayStorage(implicit alloc: StorageAllocator[A]): Storage.ArrayStorage[A] =
      alloc.toArrayStorage(ary, offset)

    final def toStorage(implicit alloc: StorageAllocator[A]): Storage[A] =
      alloc.toArrayStorage(ary, offset)
  }
}
