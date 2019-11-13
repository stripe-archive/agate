package com.stripe.agate.ops

import com.stripe.agate.tensor.{
  DataType,
  OnnxFloating,
  OnnxNumber,
  Shape,
  Storage,
  StorageAllocator,
  Tensor
}
import scala.util.Try

import Shape.Dims

object BatchNormalization {

  // data: N x C x D1 x D2 x ...
  //
  // mean: C
  def apply(
      dataType: DataType
  )(
      data0: Tensor[dataType.type],
      scale0: Tensor[dataType.type],
      bias0: Tensor[dataType.type],
      mean0: Tensor[dataType.type],
      variance0: Tensor[dataType.type],
      epsilon: Float
  ): Try[Tensor[dataType.type]] = {

    implicit val alloc = StorageAllocator.forDataType(dataType)
    val num = OnnxNumber.forDataType(dataType)
    OnnxNumber.toFloating(num).map { fl =>
      implicit val floatA = fl

      require(data0.rank > 0)
      val data = if (data0.rank == 1) data0.unsqueeze(1).get else data0
      val _ :: c :: _ = data0.dims.components

      val cax = Shape.axes(c)

      val (scale, bias, mean, variance) =
        (for {
          t0 <- scale0.broadcastTo(cax)
          t1 <- bias0.broadcastTo(cax)
          t2 <- mean0.broadcastTo(cax)
          t3 <- variance0.broadcastTo(cax)
        } yield (t0, t1, t2, t3)).get

      val size = data.dims.totalSize
      val output = alloc.allocate(size)
      val outputAxes = data.axes

      data.slices(1).foreach {
        case (i, slice) =>
          val cc = Shape.coords(i)

          val f = floatA.batchNormalize(
            mean(cc),
            variance(cc),
            scale(cc),
            bias(cc),
            epsilon
          )

          slice.dims.coords.foreach { subCoords =>
            val x = slice(subCoords)
            val res = f(x)
            val outputCoords = subCoords.insert(1, i, Shape.Coord)
            val outputIndex = outputAxes.rowMajorIndex(outputCoords)
            output.writeAt(outputIndex, res)
          }
      }

      val st: Storage[dataType.Elem] = output.toStorage
      val axes: Dims = outputAxes.asRowMajorDims
      Tensor(dataType, axes)(st)
    }
  }
}
