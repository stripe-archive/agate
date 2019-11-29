package com.stripe.agate.ops

import com.stripe.agate.tensor.{DataType, OnnxNumber, Shape, Storage, StorageAllocator, Tensor}
import scala.util.Try

import Shape.{Coord, NonEmpty}

object LayerNorm {
  def apply(
      dt: DataType
  )(
      input: Tensor[dt.type],
      normalizedShape: List[Long],
      weight: Option[Tensor[dt.type]],
      bias: Option[Tensor[dt.type]],
      eps: Double
  ): Try[Tensor[dt.type]] = {
    val axes = input.axes
    val cs = input.dims.components
    val inputSize = cs.size
    val normalizedSize = normalizedShape.size
    val total = input.dims.totalSize
    val n = cs.take(inputSize - normalizedSize).foldLeft(1L)(_ * _)

    def finish(out: Tensor[dt.type], on: OnnxNumber[dt.Elem]): Try[Tensor[dt.type]] =
      (weight, bias) match {
        case (Some(w), Some(b)) =>
          val ow = Tensor
            .map2(dt)(out, w) { (x, y) =>
              on.times(x, y)
            }
            .get
          Tensor.map2(dt)(ow, b.broadcastTo(axes).get) { (x, y) =>
            on.plus(x, y)
          }
        case (Some(w), None) =>
          Tensor.map2(dt)(out, w)(on.times(_, _))
        case (None, Some(b)) =>
          Tensor.map2(dt)(out, b)(on.plus(_, _))
        case (None, None) =>
          Try(out)
      }

    for {
      on <- OnnxNumber.forDataType(dt)
      div <- Try { require(total % n == 0); total / n }
      reshaped <- input.reshape(Shape.axes(1L, n, div))
      reshapedOut <- LayerNorm.normalize(dt)(reshaped, eps.toDouble)
      out <- reshapedOut.reshape(axes)
      res <- finish(out, on)
    } yield res
  }

  def normalize(dt: DataType)(input: Tensor[dt.type], epsilon: Double): Try[Tensor[dt.type]] =
    for {
      num <- OnnxNumber.forDataType(dt)
      fl <- OnnxNumber.toFloating(num)
    } yield {
      implicit val alloc = StorageAllocator.forDataType(dt)
      implicit val floatA = fl

      val (saveMean, saveInvStd) = LayerNorm.stats(dt)(input, epsilon).get

      val _ :: c :: _ = input.dims.components

      val dims = input.axes.asRowMajorDims
      val output = alloc.allocate(dims.totalSize)

      (0L until c).foreach { i =>
        val in = input.slice(1, i)
        val mean = saveMean(Shape.coords(i))
        val invStd = saveInvStd(Shape.coords(i))

        in.dims.coords.foreach { coord =>
          val NonEmpty(c0, Coord, crest) = coord
          val oc = NonEmpty(c0, Coord, NonEmpty(i, Coord, crest))
          val index = Shape.coordsToIndex(dims, oc)

          val x = in(coord)
          val y = fl.minus(x, mean)
          val z = fl.times(y, invStd)
          output.writeAt(index, z)
        }
      }

      Tensor(dt, input.dims)(output.toStorage)
    }

  def stats(
      dt: DataType
  )(input: Tensor[dt.type], epsilon: Double): Try[(Tensor[dt.type], Tensor[dt.type])] =
    for {
      num <- OnnxNumber.forDataType(dt)
      fl <- OnnxNumber.toFloating(num)
    } yield {
      implicit val alloc = StorageAllocator.forDataType(dt)
      implicit val floatA = fl

      require(input.rank > 1)

      val _ :: c :: _ = input.dims.components
      val n = fl.fromLong(input.dims.totalSize / c)

      val saveMean = alloc.allocate(c)
      val saveVarTransform = alloc.allocate(c)

      (0L until c).foreach { i =>
        val in = input.slice(1, i)
        val sum = in.scalars.foldLeft(fl.zero)(fl.plus(_, _))
        val mean = fl.div(sum, n)
        saveMean.writeAt(i, mean)

        val varSum = in.scalars.foldLeft(fl.zero) { (sum, x) =>
          val xm = fl.minus(x, mean)
          val xxm = fl.times(xm, xm)
          val xsum = fl.plus(sum, xxm)
          xsum
        }
        val vln = fl.div(varSum, n)
        val invstd = fl.invSqrt(vln, epsilon)
        saveVarTransform.writeAt(i, invstd)
      }
      val ds = Shape.rowMajorDims(c)
      val t1 = Tensor(dt, ds)(saveMean.toStorage)
      val t2 = Tensor(dt, ds)(saveVarTransform.toStorage)
      (t1, t2)
    }
}
