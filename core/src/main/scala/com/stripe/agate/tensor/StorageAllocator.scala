package com.stripe.agate.tensor

import com.stripe.dagon.HMap
import scala.reflect.ClassTag

sealed abstract class StorageAllocator[@specialized A: ClassTag] {
  final def allocate(size: Long): WritableStorage[A] = {
    require((0L <= size) && (size <= Int.MaxValue), s"invalid size: $size")
    WritableStorage.ArrayStorage(new Array[A](size.toInt), 0)
  }

  final def toArrayStorage(ary: Array[A], offset: Int): Storage.ArrayStorage[A] =
    Storage.ArrayStorage(ary, offset)
}

object StorageAllocator {
  def apply[A](implicit ev: StorageAllocator[A]): StorageAllocator[A] = ev

  implicit val byteAllocator: StorageAllocator[Byte] =
    new StorageAllocator[Byte] {}

  implicit val shortAllocator: StorageAllocator[Short] =
    new StorageAllocator[Short] {}

  implicit val intAllocator: StorageAllocator[Int] =
    new StorageAllocator[Int] {}

  implicit val longAllocator: StorageAllocator[Long] =
    new StorageAllocator[Long] {}

  implicit val floatAllocator: StorageAllocator[Float] =
    new StorageAllocator[Float] {}

  implicit val doubleAllocator: StorageAllocator[Double] =
    new StorageAllocator[Double] {}

  def forDataType(dt: DataType): StorageAllocator[dt.Elem] =
    forDataTypeMap(dt)

  private[this] val forDataTypeMap: HMap[DataType.Aux, StorageAllocator] =
    HMap
      .empty[DataType.Aux, StorageAllocator]
      .updated(DataType.Uint8, byteAllocator)
      .updated(DataType.Uint16, shortAllocator)
      .updated(DataType.Int8, byteAllocator)
      .updated(DataType.Int16, shortAllocator)
      .updated(DataType.Int32, intAllocator)
      .updated(DataType.Int64, longAllocator)
      .updated(DataType.BFloat16, shortAllocator)
      .updated(DataType.Float16, shortAllocator)
      .updated(DataType.Float32, floatAllocator)
      .updated(DataType.Float64, doubleAllocator)
}
