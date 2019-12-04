package com.stripe.agate

import com.google.protobuf.ByteString
import com.stripe.agate.eval.Model
import com.stripe.agate.tensor.{DataType, Shape, Tensor, ToBytes}
import java.io.ByteArrayOutputStream
import onnx.onnx.TensorProto
import org.scalacheck.{Arbitrary, Cogen, Gen}

object Onnx {
  def toDims(dims: Shape.Dims): Seq[Long] =
    dims.components.map(_.toLong)

  def toDataType(dt: DataType): Some[Int] =
    Some(Model.dataTypeToInt(dt))

  def toRawData(t: Tensor.Unknown): Some[ByteString] = {
    implicit val tb = ToBytes.forDataType(t.dataType)
    val baos = new ByteArrayOutputStream(t.dims.totalSize.toInt)
    t.writeIntoStream(baos)
    baos.close()
    val bytes = baos.toByteArray
    Some(ByteString.copyFrom(bytes))
  }

  def copySeq(t: Tensor.Unknown, tp: TensorProto): TensorProto = {
    def toSeq[A](dt: DataType)(f: dt.Elem => A): Seq[A] =
      DataType.maybeElemIs(t.dataType, dt) match {
        case Some(isa) =>
          val it: Iterator[dt.Elem] = isa.substitute[Iterator](t.scalars)
          it.map(f).toSeq
        case None => sys.error(s"unreachable: ${t.dataType} was not $dt")
      }

    t.dataType match {
      case dt: DataType.Uint8.type =>
        tp.copy(int32Data = toSeq[Int](dt)(_ & 0xff))
      case dt: DataType.Uint16.type =>
        tp.copy(int32Data = toSeq[Int](dt)(_ & 0xffff))
      case dt: DataType.Int8.type =>
        tp.copy(int32Data = toSeq[Int](dt)(_.toInt))
      case dt: DataType.Int16.type =>
        tp.copy(int32Data = toSeq[Int](dt)(_.toInt))
      case dt: DataType.Int32.type =>
        tp.copy(int32Data = toSeq[Int](dt)(x => x))
      case dt: DataType.Int64.type =>
        tp.copy(int64Data = toSeq[Long](dt)(x => x))
      case dt: DataType.BFloat16.type =>
        tp.copy(int32Data = toSeq[Int](dt)(_ & 0xffff))
      case dt: DataType.Float16.type =>
        tp.copy(int32Data = toSeq[Int](dt)(_ & 0xffff))
      case dt: DataType.Float32.type =>
        tp.copy(floatData = toSeq[Float](dt)(x => x))
      case dt: DataType.Float64.type =>
        tp.copy(doubleData = toSeq[Double](dt)(x => x))
      case dt: DataType.String.type =>
        tp.copy(stringData = toSeq[ByteString](dt)(x => x))
      case dt: DataType.Bool.type =>
        tp.copy(int32Data = toSeq[Int](dt)(x => if (x) 1 else 0))
    }
  }

  def genTensorProtoFromTensor(name: String, t: Tensor.Unknown): Gen[TensorProto] = {
    val base =
      TensorProto(name = Some(name), dims = toDims(t.dims), dataType = toDataType(t.dataType))

    Gen.oneOf(
      Gen.lzy(Gen.const(base.copy(rawData = toRawData(t)))),
      Gen.lzy(Gen.const(copySeq(t, base)))
    )
  }
}
