package com.stripe.agate.ops

import cats.implicits._
import com.stripe.agate.tensor.{Shape, Storage}
import com.stripe.agate.tensor.{
  DataType,
  OnnxIntegral,
  OnnxNumber,
  StorageAllocator,
  Tensor,
  WritableStorage
}
import scala.util.Try

import Shape._

object Gather {

  def apply[D1 <: DataType, D2 <: DataType](
      data: Tensor[D1],
      indices: Tensor[D2],
      axis: Long
  ): Try[Tensor[data.dataType.type]] = {

    val num = OnnxNumber.forDataType(indices.dataType)

    OnnxNumber.toIntegral(num).map { in =>
      implicit val oi: OnnxIntegral[indices.dataType.Elem] = in
      implicit val alloc = StorageAllocator.forDataType(data.dataType)

      // builds the axes of the resulting gathered tensor
      def makeAxes(iax: Axes, prefix: Axes, suffix: Axes): Axes =
        iax match {
          case NonEmpty(len, _, Empty) =>
            prefix ++ NonEmpty(len, Axis, suffix)
          case NonEmpty(len, _, rest) =>
            val rest2 = makeAxes(rest, prefix, suffix)
            NonEmpty(len, Axis, rest2)
          case Empty =>
            prefix ++ suffix
        }

      // given the index coords and slice coords, we need to build up
      // complete coords that will work in the output tensor. to do
      // this, we take the last index component out of indexCoords, and
      // "unslice" it into the slice coords according to the `axis`
      // parameter. the result is a coordinate in the output tensor's
      // coordinate space. If indexCoords is empty, then the final
      // coordinates are the same as the slice coords.
      def combineCoords(indexCoords: Coords, sliceCoords: Coords): Coords =
        indexCoords match {
          case Shape.Empty => sliceCoords
          case _ =>
            val (init, last, _) = indexCoords.last
            init ++ sliceCoords.insert(axis, last, Coord)
        }

      // do the work of writing out the data. at a high-level
      def writeData(output: WritableStorage[data.dataType.Elem], outputAxes: Axes): Unit = {

        val axisLen = data.dims.at(axis) match {
          case Some((len, _, _)) => len
          case None              => sys.error(s"axis $axis was not found in axes ${data.axes}")
        }

        val it = indices.dims.coords
        while (it.hasNext) {
          val indexCoords = it.next
          val x = indices(indexCoords)
          val n = oi.validIntIndex(x, axisLen)
          require(n >= 0, s"$x is not a valid index for size: $axisLen of axis $axis")

          val subdata = data.slice(axis, n.toLong)
          val it2 = subdata.dims.coords
          while (it2.hasNext) {
            val subCoords = it2.next
            val value = subdata(subCoords)
            val outputCoords = combineCoords(indexCoords, subCoords)
            val i = outputAxes.rowMajorIndex(outputCoords)
            output.writeAt(i, value)
          }
        }
      }

      data.dims
        .split(axis)
        .map {
          case (prefix, _, _, suffix) =>
            val outputAxes = makeAxes(indices.axes, prefix.as(Axis), suffix.as(Axis))
            require(outputAxes.rank == data.rank + indices.rank - 1)

            // allocate our output array and write into it
            val output = alloc.allocate(outputAxes.totalSize)
            writeData(output, outputAxes)
            Tensor(data.dataType, Shape.rowMajorDims(outputAxes))(output.toStorage)
        }
        .get
    }
  }
}
