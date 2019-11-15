package com.stripe.agate.tensor

import cats.{ApplicativeError, Functor}
import cats.arrow.FunctionK
import cats.data.NonEmptyList
import cats.effect.{IO, Resource}
import cats.evidence.Is
import cats.implicits._
import com.stripe.agate.Util
import java.io.{BufferedOutputStream, FileOutputStream, OutputStream}
import java.lang.Float.{floatToRawIntBits, intBitsToFloat}
import java.lang.Math
import java.nio.file.{Files, Path}
import java.util.Arrays
import org.typelevel.paiges.Doc
import scala.util.{Failure, Success, Try}

import Shape._

abstract class Tensor[D <: DataType] {

  val dataType: D
  def dims: Dims
  def storage: Storage[dataType.Elem]

  /**
   * This replaces the type parameter with dataType.type
   * this is useful in some cases where we have lost track
   * of the type of D
   */
  def asDataTyped: Tensor[dataType.type] =
    Tensor(dataType, dims)(storage)

  /**
   * convert this tensor to a Shape.
   * This errors if we don't have a 1-D tensor that can
   * be cast to Int64
   *
   * The name is long to avoid confusion with axes, which
   * are the axes of this tensor
   */
  def convertDataToAxes: Try[Shape.Axes] =
    axes match {
      case NonEmpty(len, _, Empty) =>
        // we have a 1-D array
        val longTen = cast(DataType.Int64)
        val shapeList = longTen.scalars.toList
        Success(Shape.axes(shapeList: _*))
      case invalidShape =>
        Failure(new Exception(s"invalid tensor to convert to Tensor($dataType, $axes) to axes"))
    }

  override def equals(that: Any): Boolean =
    that match {
      case that: Tensor[_] =>
        (this eq that) || {
          (this.dataType == that.dataType) && {
            val s0 = dims.as(Axis)
            val s1 = that.dims.as(Axis)
            s0 == s1 && s0.forall(cs => this(cs) == that(cs))
          }
        }
      case _ => false
    }

  override def hashCode: Int = {
    val s = dims.as(Axis)
    val n0 = s.hashCode * 31 + dataType.hashCode
    s.coords.foldLeft(n0)((n, c) => n ^ this(c).##)
  }

  override def toString: String =
    if (dims.totalSize < 1000) toDoc.render(100)
    else s"Tensor($dataType, $axes)(...)"

  def axes: Axes =
    dims.as(Axis)

  def axesString: String =
    dims.axesString

  def rank: Int =
    dims.rank

  /**
   * This is equivalent to numpy.reshape(data, axes)
   * the total size is unchanged, this only remaps
   * the addressing of the tensor
   */
  def reshape(axes: Axes): Try[Tensor[dataType.type]] =
    if (dims.isRowMajor) dims.reshapeRowMajor(axes).map(withDims)
    else toRowMajor.reshape(axes)

  /**
   * if the current tensor is not rowmajor, rewrite it
   * to row-major
   */
  def toRowMajor: Tensor[dataType.type] =
    if (dims.isRowMajor) this.asDataTyped
    else map(identity)

  private[tensor] def withDims(d: Dims): Tensor[dataType.type] =
    Tensor(dataType, d)(storage)

  def broadcastTo(axes: Axes): Try[Tensor[dataType.type]] =
    dims.broadcastTo(axes).map(withDims)

  def squeeze(axis: Long): Option[Tensor[dataType.type]] =
    dims.squeeze(axis).map(withDims)

  def unsqueeze(axis: Long): Option[Tensor[dataType.type]] =
    dims.unsqueeze(axis).map(withDims)

  def closeTo(that: Tensor[dataType.type], error: Float = 1e-5f): Boolean = {
    val s0 = axes
    val s1 = that.axes
    val num = OnnxNumber.forDataType(dataType)
    s0 == s1 && s0.forall { cs =>
      val x0 = this(cs)
      val x1 = that(cs)
      Util.closeTo(num.toFloat(x0), num.toFloat(x1), error)
    }
  }

  def l1NormTo(that: Tensor[dataType.type]): Double = {
    val num = OnnxNumber.forDataType(dataType)
    val s0 = axes
    require(s0 == that.axes, s"expected $s0 == ${that.axes}")
    var res = 0.0
    s0.foreach { cs =>
      val x0 = this(cs)
      val x1 = that(cs)
      res += Math.abs(num.toDouble(x0) - num.toDouble(x1))
    }
    res
  }

  def apply(coords: Coords): dataType.Elem =
    coords match {
      case NonEmpty(n, _, ds) => sliceFirst(n).apply(ds)
      case Empty              => scalar
    }

  def toDoc: Doc = {
    val sep = Doc.comma + Doc.line
    dims match {
      case Empty =>
        val num = OnnxNumber.forDataType(dataType)
        Doc.text(num.render(scalar))
      case NonEmpty(len, _, _) =>
        val docs = (0L until len).map(i => sliceFirst(i).toDoc)
        (Doc.char('[') + Doc.intercalate(sep, docs) + Doc.char(']')).grouped
    }
  }

  def nonZero: Tensor[DataType.Int64.type] = {
    // Note that this will return a tensor of shape (rank, #nonzero)
    // where #nonzero is the number of non-zero elements in the tensor.

    val num = OnnxNumber.forDataType(dataType)

    val filtered = axes.coords.foldLeft(List[Shape[Coord]]()) { (list, coord) =>
      {
        val x = this(coord)
        if (!num.zero.equals(x)) {
          list ++ List[Shape[Coord]](coord)
        } else {
          list
        }
      }
    }

    val listOfVectors = filtered.map { coords =>
      {
        val array = coords.toList.map(_._1).toArray
        Tensor.vector(DataType.Int64)(array)
      }
    }

    if (listOfVectors.isEmpty) {
      // If there are no non-zero values, return tensor of shape (rank, 0)
      Tensor.const(DataType.Int64)(0, Shape.axes(dims.rank, 0))
    } else {
      Tensor.stack(DataType.Int64, 1)(listOfVectors).get
    }
  }

  def map(f: dataType.Elem => dataType.Elem): Tensor[dataType.type] = {
    implicit val alloc = StorageAllocator.forDataType(dataType)
    val n = dims.totalSize
    val output = alloc.allocate(n)
    writeInto(output, 0)
    var i = 0L
    while (i < n) {
      output.writeAt(i, f(output(i)))
      i += 1L
    }
    Tensor(dataType, dims.asRowMajorDims)(output.toStorage)
  }

  def map(dest: DataType)(f: dataType.Elem => dest.Elem): Tensor[dest.type] = {
    implicit val alloc = StorageAllocator.forDataType(dest)
    val n = dims.totalSize
    val output = alloc.allocate(n)
    var i = 0L
    val it = scalars
    while (it.hasNext) {
      val x = it.next
      output.writeAt(i, f(x))
      i += 1L
    }
    Tensor(dest, dims.asRowMajorDims)(output.toStorage)
  }

  def cast(dest: DataType): Tensor[dest.type] =
    DataType.maybeElemIs(dataType, dest) match {
      case Some(isa) =>
        //isa.substitute[Tensor](this)
        this.asInstanceOf[Tensor[dest.type]] //fixme
      case None =>
        val f: dataType.Elem => dest.Elem =
          OnnxNumber.cast(dataType, dest)
        this.map(dest)(f)
    }

  /**
   * map each element to another type, then use a commutative and associative function to reduce
   * along a set of given dimensions.
   *
   * empty axes mean reduce along all axes
   */
  def foldMap(dest: DataType, reduceAxes: List[Long], keepDims: Boolean)(
      f: dataType.Elem => dest.Elem,
      g: (dest.Elem, dest.Elem) => dest.Elem
  ): Try[Tensor[dest.type]] = {
    val outputShapeTry: Try[Axes] = axes.reduce(reduceAxes, keepDims)

    outputShapeTry.map { outputShape =>
      // for all the reduce axes, we enumerate everything
      // in those dimensions
      val contributors: Coords => Iterator[Coords] =
        if (reduceAxes.isEmpty) {
          // reduce all the points down:
          { (coords: Coords) =>
            axes.coords
          }
        } else {
          val axesWithSize = reduceAxes.map { axis =>
            // this axis is valid of outputShapeTry will be a Failure
            (axis, axes.at(axis).get._1)
          }
          val sortedAxes = axesWithSize.sorted

          def loop(axesSize: List[(Long, Long)], axis: Long): Coords => Iterator[Coords] =
            axesSize match {
              case (a0, sz) :: arest =>
                if (a0 == axis) {
                  // we need to iterate over this entire dimension
                  val fanout = (0L until sz)
                  val restfn = loop(arest, axis + 1L)

                  {
                    case ne @ Shape.NonEmpty(_, _, rest) =>
                      // if we keep the dimensions, we need to skip this index
                      // else, we deleted it, so we pass the whole thing in
                      val next = if (keepDims) rest else ne
                      restfn(next).flatMap { rest =>
                        fanout.map { idx =>
                          Shape.NonEmpty(idx, Shape.Coord, rest)
                        }
                      }
                    case Empty =>
                      if (keepDims) {
                        sys.error(
                          s"unreachable: Empty should be invalid for reduceAxes: $reduceAxes, axis = $axis, axesSize = $axesSize"
                        )
                      } else {
                        // else, we deleted this dimension, and call down with empty
                        restfn(Empty).flatMap { rest =>
                          fanout.map { idx =>
                            Shape.NonEmpty(idx, Shape.Coord, rest)
                          }
                        }
                      }
                  }
                } else {
                  val restfn = loop(axesSize, axis + 1L)

                  {
                    case Shape.NonEmpty(c, _, rest) =>
                      restfn(rest).map { rest =>
                        Shape.NonEmpty(c, Shape.Coord, rest)
                      }
                    case Empty =>
                      sys.error(
                        s"unreachable: Empty should be invalid for reduceAxes: $reduceAxes, axis = $axis, axesSize = $axesSize"
                      )
                  }
                }
              case Nil => { c =>
                Iterator.single(c)
              }
            }

          loop(sortedAxes, 0L)
        }

      implicit val alloc = StorageAllocator.forDataType(dest)
      val n = outputShape.totalSize
      val output = alloc.allocate(n)
      val it = outputShape.coords
      while (it.hasNext) {
        val c = it.next
        val reduced = contributors(c)
          .map { coord =>
            f(apply(coord))
          }
          .reduce(g)
        val i = outputShape.rowMajorIndex(c)
        output.writeAt(i, reduced)
      }
      Tensor(dest, outputShape.asRowMajorDims)(output.toStorage)
    }
  }

  def writeIntoStream(os: OutputStream)(implicit tb0: ToBytes[dataType.Elem]): Unit =
    dims match {
      case Empty =>
        storage.writeIntoStream(os, SingleDim, 1)
      case NonEmpty(len, dim, Empty) =>
        storage.writeIntoStream(os, dim, len.toLong)
      case NonEmpty(len, _, _) =>
        var i = 0L
        while (i < len) {
          sliceFirst(i).writeIntoStream(os)
          i += 1L
        }
    }

  // write data into output in row-major form
  def writeInto(output: WritableStorage[dataType.Elem], start: Long): Long =
    dims match {
      case Empty =>
        storage.writeInto(output, start, SingleDim, 1)
        1L
      case NonEmpty(len, dim, Empty) =>
        val lenLong = len.toLong
        storage.writeInto(output, start, dim, lenLong)
        lenLong
      case NonEmpty(len, _, _) =>
        var i = 0L
        var loc = start.toLong
        while (i < len) {
          loc += sliceFirst(i).writeInto(output, loc)
          i += 1L
        }
        loc - (start.toLong)
    }

  /**
   * When called on a scalar (tensor with zero components), returns a
   * scalar value.
   *
   * In other cases this method throws an error.
   */
  def scalar: dataType.Elem =
    dims match {
      case Empty             => storage(0)
      case NonEmpty(1, _, _) => sliceFirst(0).scalar
      case _                 => sys.error(s"not a scalar: $this")
    }

  /**
   * Iterate over all items in row major order
   */
  def scalars: Iterator[dataType.Elem] =
    dims match {
      case NonEmpty(1, _, _) =>
        sliceFirst(0).scalars
      case NonEmpty(len, Dim(off, stride), Empty) =>
        (0L until len).iterator.map(i => storage(i * stride + off))
      case NonEmpty(len, _, rest) =>
        (0L until len).iterator.flatMap { (i: Long) =>
          val s: Tensor[dataType.type] = sliceFirst(i)
          val isa = implicitly[Is[s.dataType.Elem, dataType.Elem]]
          isa.substitute[Iterator](s.scalars)
        }
      case Empty =>
        Iterator.single(scalar)
    }

  def indices(axis: Long): Iterable[Long] =
    slices(axis).map(_._1)

  def slices(axis: Long): Seq[(Long, Tensor[dataType.type])] =
    dims.at(axis) match {
      case Some((length, _, _)) =>
        (0L until length).map(i => (i, slice(axis, i)))
      case None =>
        Nil
    }

  def max: dataType.Elem = {
    val num = OnnxNumber.forDataType(dataType)
    val it = scalars
    if (it.hasNext) {
      var best = it.next
      while (it.hasNext) {
        best = num.max(best, it.next)
      }
      best
    } else {
      sys.error("empty tensor has no max!")
    }
  }

  /**
   * Get the maximum values along a given axis.
   */
  def maxAxes(axes: Long*): Try[Tensor[dataType.type]] =
    if (axes.isEmpty) Success(Tensor.const(dataType)(max, Shape.Empty))
    else {
      val num = OnnxNumber.forDataType(dataType)
      foldMap(dataType, axes.toList, keepDims = false)(x => x, num.max(_, _))
    }

  /**
   * This is like numpy array slicing
   */
  def select(range: List[Shape.AxisRange]): Tensor[dataType.type] =
    withDims(dims.select(range))

  def sum: dataType.Elem = {
    val num = OnnxNumber.forDataType(dataType)
    scalars.foldLeft(num.zero)(num.plus)
  }

  /**
   * same as numpy.sum
   * discards each axis we sum, reducing the rank by that
   * number of inputs
   */
  def sumAxes(axes: Long*): Try[Tensor[dataType.type]] =
    if (axes.isEmpty) Success(Tensor.const(dataType)(sum, Shape.Empty))
    else {
      val num = OnnxNumber.forDataType(dataType)
      foldMap(dataType, axes.toList, keepDims = false)(x => x, num.plus(_, _))
    }

  /**
   * same as numpy.sum but keep the rank the same: leave 1 sized tensor
   * in the place of the previous summed axis
   * if no axes are given, sum over all axes
   */
  def sumAxesKeep(sumAxes: Long*): Try[Tensor[dataType.type]] =
    if (sumAxes.isEmpty) {
      // sum on all axes
      Success(Tensor.const(dataType)(sum, axes.toSingleton))
    } else {
      val num = OnnxNumber.forDataType(dataType)
      foldMap(dataType, sumAxes.toList, keepDims = true)(x => x, num.plus(_, _))
    }

  // slice the first axis
  private def sliceFirst(index: Long): Tensor[dataType.type] =
    dims match {
      case NonEmpty(_, Dim(off, stride), ds) =>
        val offset2 = index * stride + off
        Tensor(dataType, ds)(storage.slice(offset2))
      case Empty =>
        sys.error(s"scalar data cannot be sliced: $this")
    }

  /**
   * Slice along a given axis taking one of the subtensors
   * along that axis
   *
   * Note this is not exactly what ONNX calls slice
   */
  def slice(axis: Long, index: Long): Tensor[dataType.type] =
    if (axis == 0L) sliceFirst(index)
    else
      dims.at(axis) match {
        case None =>
          sys.error(s"axis $axis was not valid for $this")
        case Some((_, Dim(off, stride), ds)) =>
          val offset2 = index * stride + off
          Tensor(dataType, ds)(storage.slice(offset2))
      }

  def chunk(axis: Long, size: Long): Try[Seq[Tensor[dataType.type]]] =
    dims.chunk(axis = axis, size = size).map { dimSeq =>
      dimSeq.map(withDims(_))
    }

  def transpose(axis1: Long, axis2: Long): Option[Tensor[dataType.type]] =
    dims.transpose(axis1, axis2).map { ds =>
      withDims(ds)
    }

  /**
   * Transpose given the permutation list
   */
  def transpose(axes: List[Long]): Try[Tensor[dataType.type]] =
    dims.transpose(axes).map { ds =>
      withDims(ds)
    }

  /**
   * Reverse the dimensions. This is the usual transpose for a matrices
   */
  def transposeDefault: Tensor[dataType.type] =
    withDims(dims.transposeDefault)

  def assertDataType(dt: DataType): Try[Tensor[dt.type]] =
    DataType.maybeElemIs(dataType, dt) match {
      case Some(isa) =>
        val st = isa.substitute[Storage](storage)
        Success(Tensor(dt, dims)(st))
      case None =>
        Failure(new Exception(s"tensor's type ${dataType} is not $dt"))
    }
}

object Tensor {
  type F = Tensor[DataType.Float32.type]
  val F = Tensor

  type Unknown = Tensor[_ <: DataType]
  type U = Tensor[_ <: DataType]

  def apply(dt: DataType, dims0: Dims)(storage0: Storage[dt.Elem]): Tensor[dt.type] =
    new Tensor[dt.type] {
      final val dataType: dt.type = dt
      final val dims: Dims = dims0
      final val storage = storage0
    }

  val Zero: Tensor.F =
    scalar(DataType.Float32)(0f)

  val One: Tensor.F =
    scalar(DataType.Float32)(1f)

  /**
   * This is a broadcasting map2 following numpy broadcasting
   * rules
   */
  def map2(
      dt: DataType
  )(x0: Tensor[dt.type], y0: Tensor[dt.type])(
      f: (dt.Elem, dt.Elem) => dt.Elem
  ): Try[Tensor[dt.type]] =
    Shape.broadcast(x0.dims, y0.dims).map {
      case (xdims, ydims) =>
        val x1 = Tensor(dt, xdims)(x0.storage)
        val y1 = Tensor(dt, ydims)(y0.storage)
        val axes = xdims.as(Shape.Axis)
        implicit val alloc = StorageAllocator.forDataType(dt)
        val writable: WritableStorage[dt.Elem] = alloc.allocate(axes.totalSize)
        var i = 0L
        xdims.coords.foreach { coords =>
          val ex = x1(coords)
          val ey = y1(coords)
          writable.writeAt(i, f(ex, ey))
          i += 1L
        }
        Tensor(dt, axes.asRowMajorDims)(writable.toStorage)
    }

  // tensors.forall(_.axes == tensors(0).axes)
  // 0 <= axis <= axes.size
  //
  // this is like unslice:
  // Tensor.stack(dt, axis)(tensors).get.slices(axis) == tensors
  //
  // Tensor.stack(t.dataType, axis)(t.slices(axis)) = t
  //
  def stack(dt: DataType, axis: Long)(tensors: Seq[Tensor[dt.type]]): Try[Tensor[dt.type]] =
    Try {
      require(tensors.nonEmpty, "no tensors provided!")
      val axes = tensors.head.axes
      val allAxes = tensors.map(_.axes)
      require(allAxes.forall(_ == axes), s"misaligned axes: $allAxes")
      require(0 <= axis && axis <= axes.rank, s"invalid axis: 0 <= $axis <= ${axes.rank}")

      val n = tensors.size.toLong
      val outputAxes: Axes = axes.insert(axis, n, Axis)
      implicit val alloc = StorageAllocator.forDataType(dt)
      val output = alloc.allocate(axes.totalSize * tensors.size)

      tensors.zipWithIndex.foreach {
        case (slice, i0) =>
          val i = i0.toLong
          val it = slice.dims.coords
          while (it.hasNext) {
            val coords: Coords = it.next
            val value = slice(coords)
            val outputCoords: Coords = coords.insert(axis, i, Coord)
            val j = outputAxes.rowMajorIndex(outputCoords)
            output.writeAt(j, value)
          }
      }
      Tensor(dt, outputAxes.asRowMajorDims)(output.toStorage)
    }

  /**
   * equivalent to numpy.concatenate
   * https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
   */
  def concatenate(dt: DataType, axis: Long)(tensors: Seq[Tensor[dt.type]]): Try[Tensor[dt.type]] =
    if (tensors.isEmpty)
      Failure(
        new IllegalArgumentException(
          s"tensors may not be empty in concatenate($dt, $axis)($tensors)"
        )
      )
    else {
      val axes = tensors.map(_.axes)
      Shape
        .concatenate(axis = axis, axes.head, axes.tail: _*)
        .map { concatShape =>
          // on the given axis, we have a set of lens:
          val axisLens: Seq[Long] = axes.map(_.at(axis).get._1)
          val axisMap: Array[Long] = axisLens.scanLeft(0L)(_ + _).toArray
          // we will be looking into this often
          val tarray = tensors.toArray
          def toInput(c: Coords): dt.Elem = {
            /*
             * This is a bit tricky due to the API of binarySearch, but
             * the algorithm works like this:
             * 1. split the input coordinates to identify the index along the axis we are concat
             *    along
             * 2. search for which of the tensors we would fetch this index from and adjust the
             *    position to account for the previous tensors we are skipping
             * 3. build a new coord instance with the correct position
             */
            val (pre, idx, _, post) = c.split(axis).get
            /*
             * This is a bit subtle: note axisMap came from scanLeft(0)(_ + _)
             * so it is 1 longer than tensors: we have prepended 0.
             * Next, note all indices >= 0, so we will never seek to insert
             * before the first element in the array.
             *
             * when absent, binarySearch, returns -(x) - 1 = y
             * x = -(y + 1) = -(y) - 1
             *
             * Note, since we have padded with 0, the inserting point
             * will be exactly correct (when we find the item in the search)
             * or be pointing to one greater, so we need to subtract.
             */
            val bidx = Arrays.binarySearch(axisMap, idx)
            val pos = if (bidx < 0) -(bidx + 1) - 1 else bidx
            val idx1 = idx - axisMap(pos)
            val innerCoord = pre ++ Shape.NonEmpty(idx1, Shape.Coord, post)
            tarray(pos)(innerCoord)
          }

          implicit val alloc = StorageAllocator.forDataType(dt)
          val output = alloc.allocate(concatShape.totalSize)
          val it = concatShape.coords
          while (it.hasNext) {
            val coord = it.next
            val j = concatShape.rowMajorIndex(coord)
            output.writeAt(j, toInput(coord))
          }
          Tensor(dt, concatShape.asRowMajorDims)(output.toStorage)
        }
    }

  /**
   * This is the opposite of chunk, which glues together tensors along
   * a given axis
   */
  def unchunk(dt: DataType, axis: Long)(tensors: Seq[Tensor[dt.type]]): Try[Tensor[dt.type]] =
    Tensor.stack(dt, axis)(tensors).flatMap { (t0: Tensor[dt.type]) =>
      t0.dims.unchunk(axis).map { (d: Dims) =>
        val t1: Tensor[dt.type] = t0.withDims(d)
        t1
      }
    }

  def scalar(dt: DataType)(x: dt.Elem): Tensor[dt.type] =
    const(dt)(x, Empty)

  def zero(axes: Axes): Tensor.F =
    const(DataType.Float32)(0f, axes)

  def one(axes: Axes): Tensor.F =
    const(DataType.Float32)(1f, axes)

  def const(dt: DataType)(x: dt.Elem, axes: Axes): Tensor[dt.type] = {
    implicit val alloc = StorageAllocator.forDataType(dt)
    val data = alloc.allocate(1)
    data.writeAt(0, x)
    Tensor(dt, axes.as(BroadcastDim))(data.toStorage)
  }

  def of[D <: DataType.Aux[E], E, H](dt: D)(h: H)(implicit ev: Build[H, E]): Tensor[dt.type] =
    ev.build[D](dt, h)

  def apply[H](h: H)(implicit ev: Build[H, Float]): Tensor[DataType.Float32.type] =
    of[DataType.Float32.type, Float, H](DataType.Float32)(h)

  def apply(dt: DataType)(data: Array[dt.Elem], offset: Int, dims: Dims): Tensor[dt.type] = {
    val alloc = StorageAllocator.forDataType(dt)
    Tensor(dt, dims)(alloc.toArrayStorage(data, offset))
  }

  /**
   * This is the inverse of convertDataToAxes, represent
   * the axes as a 1-D tensor with type DataType.Int64
   */
  def convertAxesToTensor(s: Shape.Axes): Tensor[DataType.Int64.type] =
    vector(DataType.Int64)(s.components.toArray)

  def vector(dt: DataType)(data: Array[dt.Elem]): Tensor[dt.type] =
    apply(dt)(data, 0, NonEmpty(data.length.toLong, SingleDim, Empty))

  def matrix(
      dt: DataType
  )(data: Array[dt.Elem], offset: Int, height: Long, width: Long): Tensor[dt.type] =
    apply(dt)(data, offset, Shape((height, Dim(0L, width)), (width, SingleDim)))

  def identityMatrix(dt: DataType, size: Long): Tensor[dt.type] = {
    implicit val alloc = StorageAllocator.forDataType(dt)
    val num = OnnxNumber.forDataType(dt)
    val len = size * (size.toLong)
    val data = alloc.allocate(len)
    var i = 0L
    while (i < len) {
      data.writeAt(i, num.one)
      i += size + 1L
    }
    val dims = NonEmpty(size, Dim(0L, size), NonEmpty(size, SingleDim, Empty))
    Tensor(dt, dims)(data.toStorage)
  }

  def save[D <: DataType](path: Path, tensor: Tensor[D]): IO[Unit] =
    Resource
      .make(IO(new BufferedOutputStream(new FileOutputStream(path.toFile))))(os => IO(os.close()))
      .use { bos =>
        IO {
          implicit val tb = ToBytes.forDataType(tensor.dataType)
          tensor.writeIntoStream(bos)
        }
      }

  def loadBytes(bytes: Array[Byte], dt: DataType, axes: Axes): Try[Tensor[dt.type]] = {
    implicit val alloc = StorageAllocator.forDataType(dt)
    val size = axes.totalSize
    val tb = ToBytes.forDataType(dt)
    val step = tb.size
    val len = size * step
    if (bytes.length < len)
      Failure(TensorException.InsufficentBytesToLoad(bytes.length.toLong, dt, len))
    else
      Success {
        val data = alloc.allocate(size)
        var i = 0
        var j = 0L
        while (i < len) {
          val x = tb.read(bytes, i)
          data.writeAt(j, x)
          i += step
          j += 1L
        }
        Tensor(dt, axes.asRowMajorDims)(data.toStorage)
      }
  }

  // ~32MB
  private final val maxInMemoryTensorSizeInBytes: Int = 32 * 1024 * 1024
  // ~16MB
  private final val maxChunkSizeInBytes: Int = 16 * 1024 * 1024

  def load(path: Path, dt: DataType, axes: Axes): Resource[IO, Tensor[dt.type]] = {

    def fs: Resource[IO, Tensor[dt.type]] =
      loadMappedRowMajorTensor(dt, path, axes)

    def mem: IO[Tensor[dt.type]] =
      IO(Files.readAllBytes(path))
        .flatMap { bytes =>
          ApplicativeError[IO, Throwable].fromTry(loadBytes(bytes, dt, axes))
        }

    Resource
      .suspend(IO.delay {
        val n = path.toFile.length
        if (n < 0 || maxInMemoryTensorSizeInBytes < n) fs else Resource.liftF(mem)
      })
  }

  def loadMappedTensor(dt: DataType, path: Path, dims: Dims): Resource[IO, Tensor[dt.type]] =
    Storage.loadMapped(dt, path, dims.totalSize, maxChunkSizeInBytes).map { fbuf =>
      Tensor(dt, dims)(fbuf)
    }

  def loadMappedRowMajorTensor(
      dt: DataType,
      path: Path,
      axes: Axes
  ): Resource[IO, Tensor[dt.type]] =
    loadMappedTensor(dt, path, axes.asRowMajorDims)
}
