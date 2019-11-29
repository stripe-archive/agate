package com.stripe.agate.ops

import com.stripe.agate.tensor.{
  DataType,
  OnnxNumber,
  Shape,
  Storage,
  StorageAllocator,
  Tensor,
  WritableStorage
}
import scala.{specialized => sp}
import scala.reflect.ClassTag
import scala.util.Try

import com.stripe.agate.Util.RequireIntOps

object Gemm {
  def transposeArray[@sp A](in: Array[A], height: Int, width: Int)(
      implicit ct: ClassTag[A]
  ): Array[A] = {
    require(in.length == (height * width))
    if (height == 1 || width == 1) {
      // if height==1, then the matrix is 1xW, i.e. a vector of length W.
      // the transpose of this is a Wx1 matrix, i.e. the same vector.
      //
      // the same argument works for width=1, or widht=1 and height=1 (scalar).
      in
    } else {
      val out = new Array[A](in.length)
      var i = 0
      while (i < height) {
        var j = 0
        val inspan = i * width
        while (j < width) {
          val inx = inspan + j
          val outx = (j * height) + i
          out(outx) = in(inx)
          j += 1
        }
        i += 1
      }
      out
    }
  }

  private def getRowMajorArray[D <: DataType](d: D)(t0: Tensor[d.type]): Option[Array[d.Elem]] =
    t0.storage match {
      case Storage.ArrayStorage(data, 0) =>
        t0.dims match {
          case Shape.NonEmpty(_, Shape.Dim(0, w0), Shape.NonEmpty(w1, Shape.SingleDim, Shape.Empty))
              if w0 == w1 =>
            Some(data)
          case _ => None
        }
      case _ => None
    }

  private def getColMajorArray[D <: DataType](d: D)(t0: Tensor[d.type]): Option[Array[d.Elem]] =
    t0.storage match {
      case Storage.ArrayStorage(data, 0) =>
        t0.dims match {
          case Shape.NonEmpty(_, Shape.SingleDim, Shape.NonEmpty(w1, Shape.Dim(0, w0), Shape.Empty))
              if w0 == w1 =>
            Some(data)
          case _ => None
        }
      case _ => None
    }

  def intoRowMajorArray[@sp A](t0: Tensor[_])(implicit ct: ClassTag[A]): Array[A] = {
    val t = t0.asInstanceOf[Tensor[DataType.Aux[A]]]
    getRowMajorArray(t.dataType)(t.asDataTyped) match {
      case Some(arr) => arr
      case None =>
        val n = t.dims.totalSize.requireInt
        val arr: Array[A] = new Array[A](n)
        val out = WritableStorage.ArrayStorage(arr, 0)
        t.writeInto(out, 0)
        arr
    }
  }

  def intoColMajorArray[@sp A](t0: Tensor[_])(implicit ct: ClassTag[A]): Array[A] = {
    val t = t0.asInstanceOf[Tensor[DataType.Aux[A]]]
    getColMajorArray(t.dataType)(t.asDataTyped) match {
      case Some(arr) => arr
      case None =>
        val arr = intoRowMajorArray[A](t)
        val (h, w) = t0.dims.rowsByCols.get
        transposeArray[A](arr, h.toInt, w.toInt)(ct)
    }
  }

  // do GEMM on an optimized set of arrays
  //
  // requirements:
  //   a is row-major (height x common)
  //   b is col-major (common x width)
  //   c is row-major (height x width)
  //
  // output is row-major (height x width)
  def arrayGemm[@specialized A](
      a: Array[A],
      b: Array[A],
      c: Array[A],
      alpha: A,
      beta: A,
      height: Int,
      common: Int,
      width: Int,
      on: OnnxNumber[A]
  )(implicit ct: ClassTag[A]): Array[A] = {
    // we allow arrays longer than needed if we have truncated rows
    // of an existing row-major tensor, this can happen
    require(a.length >= (height.toLong * common.toLong).requireInt)
    require(b.length >= (common.toLong * width.toLong).requireInt)
    require(c.length >= (height.toLong * width.toLong).requireInt)
    val output = new Array[A](height * width) // (height x width) matrix
    var y = 0
    while (y < height) {
      val yoffset = y * common /* common = width for a */
      val zoffset = y * width /* width = width for output */
      var x = 0
      while (x < width) {
        val xoffset = x * common /* common = height for b */
        var i = 0
        var out = on.zero
        while (i < common) {
          out = on.plus(out, on.times(a(yoffset + i), b(xoffset + i)))
          i += 1
        }
        output(zoffset + x) = on.times(alpha, out)
        x += 1
      }
      y += 1
    }

    var i = 0
    while (i < c.length) {
      val n = output(i)
      output(i) = on.plus(n, on.times(beta, c(i)))
      i += 1
    }
    output
  }

  def apply(
      dataType: DataType
  )(
      a0: Tensor[dataType.type],
      b0: Tensor[dataType.type],
      c0: Tensor[dataType.type],
      alpha: Float,
      beta: Float,
      transA: Boolean,
      transB: Boolean
  ): Try[Tensor[dataType.type]] = Try {
    implicit val onnxA = OnnxNumber.forDataTypeOrThrow(dataType)
    implicit val alloc = StorageAllocator.forDataType(dataType)
    if (a0.dims.rank != 2 || b0.dims.rank != 2) {
      sys.error(s"dimension were wrong: a=${a0.axes.axesString} b=${b0.axes.axesString}")
    } else {
      val a = if (transA) a0.transpose(0, 1).get else a0
      val b = if (transB) b0.transpose(0, 1).get else b0

      // we need to satisfy:
      //   a: H x C
      //   b: C x W
      //   c: H x W

      val (h0, common0) = a.dims.rowsByCols.get
      val (common1, w1) = b.dims.rowsByCols.get
      val cAxes = Shape.axes(h0, w1)
      val c = c0.broadcastTo(cAxes).get
      val (h2, w2) = c.dims.rowsByCols.get

      if (h0 != h2 || common0 != common1 || w1 != w2) {
        sys.error(
          s"sizes didn't match: h0=$h0 h2=$h2 common0=$common0 common1=$common1 w1=$w1 w2=$w2"
        )
      } else {
        val height = h0.toInt
        val common = common0.toInt
        val width = w1.toInt

        // we get about 2x speed up doing an explicit pattern-match +
        // casting to a known primitive type. this is because our code
        // is indirect enough that scala's specialization doesn't
        // fully kick in.
        //
        // right now we've only specialized for Float32 but if we find
        // ourselves needing faster GEMM for other types (e.g. Int32,
        // Float16, etc.) then add the additional cases.
        dataType match {
          case DataType.Float32 =>
            val dt = DataType.Float32
            val xa = intoRowMajorArray[Float](a.asInstanceOf[Tensor.F])
            val xb = intoColMajorArray[Float](b.asInstanceOf[Tensor.F])
            val xc = intoRowMajorArray[Float](c.asInstanceOf[Tensor.F])
            val xout =
              arrayGemm[Float](xa, xb, xc, alpha, beta, height, common, width, OnnxNumber.Float32)
            Tensor(DataType.Float32)(xout, 0, c.axes.asRowMajorDims)
              .asInstanceOf[Tensor[dataType.type]]
          case _ =>
            val ct = dataType.classTag
            val xa = intoRowMajorArray[dataType.Elem](a)(ct)
            val xb = intoColMajorArray[dataType.Elem](b)(ct)
            val xc = intoRowMajorArray[dataType.Elem](c)(ct)
            val on = OnnxNumber.forDataTypeOrThrow(dataType)
            val xout = arrayGemm[dataType.Elem](
              xa,
              xb,
              xc,
              on.fromFloat(alpha),
              on.fromFloat(beta),
              height,
              common,
              width,
              on
            )(ct)
            Tensor(dataType)(xout, 0, c.axes.asRowMajorDims)
        }
      }
    }
  }
}
