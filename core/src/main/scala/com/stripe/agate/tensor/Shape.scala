package com.stripe.agate.tensor

import cats.{Applicative, Eval, Traverse}
import cats.implicits._
import scala.util.{Failure, Success, Try}

sealed abstract class Shape[+A] {
  import Shape.{Empty, NonEmpty}

  override def toString: String =
    toList.mkString("Shape(", ", ", ")")

  /**
   * This is also known as rank
   */
  def rank: Int =
    this match {
      case Empty                => 0
      case NonEmpty(_, _, rest) => rest.rank + 1
    }

  def components: List[Long] =
    this match {
      case NonEmpty(len, _, rest) => len :: rest.components
      case Empty                  => Nil
    }

  def toList: List[(Long, A)] =
    this match {
      case NonEmpty(len, a, rest) => (len, a) :: rest.toList
      case Empty                  => Nil
    }

  def rowsByCols: Option[(Long, Long)] =
    this match {
      case NonEmpty(rows, _, NonEmpty(cols, _, Empty)) => Some((rows, cols))
      case _                                           => None
    }

  def at(axis: Long): Option[(Long, A, Shape[A])] =
    split(axis).map {
      case (pre, len, a, post) =>
        (len, a, pre ++ post)
    }

  def lengthOf(axis: Long): Option[Long] =
    this match {
      case NonEmpty(len, a, s) =>
        if (axis == 0) Some(len)
        else s.lengthOf(axis - 1)
      case Empty => None
    }

  def ++[A1 >: A](that: Shape[A1]): Shape[A1] =
    this match {
      case NonEmpty(len, a, s) => NonEmpty(len, a, s ++ that)
      case Empty               => that
    }

  def split(axis: Long): Option[(Shape[A], Long, A, Shape[A])] =
    this match {
      case Empty =>
        None
      case NonEmpty(len, a, rest) if axis == 0 =>
        Some((Empty, len, a, rest))
      case NonEmpty(len0, a0, rest) =>
        rest.split(axis - 1).map {
          case (pre, len, a, post) => (NonEmpty(len0, a0, pre), len, a, post)
        }
    }

  /**
   * Return None if either of the axes are not valid
   */
  def transpose(axis1: Long, axis2: Long): Option[Shape[A]] =
    if (this == Empty) Some(this)
    else if (axis1 == axis2) Some(this)
    else if (axis1 > axis2) transpose(axis2, axis1)
    else {
      // since axis1 < axis2, if we split by axis2 first, we know that
      // pre2 contains axis1, and also that the numbering of the axes
      // in pre2 is unchanged.
      split(axis2).flatMap {
        case (pre2, len2, a2, post2) =>
          pre2.split(axis1).map {
            case (pre1, len1, a1, post1) =>
              pre1 ++ NonEmpty(len2, a2, post1) ++ NonEmpty(len1, a1, post2)
          }
      }
    }

  /**
   * Given a permutation, perform that permutation
   * on the Shape
   *
   * this implies the input must have length == rank
   */
  def transpose(axes: List[Long]): Try[Shape[A]] = {
    val valid: Try[Unit] =
      this match {
        case Empty =>
          if (axes.isEmpty) Success(())
          else Failure(new IllegalArgumentException(s"invalid permutation: $axes for Empty"))
        case nonEmpty =>
          if (axes.sorted == (0 until rank).map(_.toLong).toList) Success(())
          else Failure(new IllegalArgumentException(s"invalid permutation: $axes for $nonEmpty"))
      }

    valid.map { _ =>
      val shapeArray: Array[(Long, A)] = toList.toArray
      // now just remap all of these things:
      val items = axes.reverseIterator.map { idxLong =>
        val idx = idxLong.toInt
        shapeArray(idx)
      }
      items.foldLeft(Shape.empty[A]) {
        case (s, (l, a)) =>
          NonEmpty(l, a, s)
      }
    }
  }

  /**
   * reverse the dimensions
   */
  def transposeDefault: Shape[A] =
    this match {
      case Empty => Empty
      case nonEmpty =>
        val axes = (0 until rank).reverseIterator.map(_.toLong).toList
        // we know this get is safe
        transpose(axes).get
    }

  /**
   * Insert (`len`, `a`) into the shape at position `i`.
   *
   * We require 0 <= i <= s.rank.
   */
  def insert[A1 >: A](i: Long, len: Long, a1: A1): Shape[A1] =
    this match {
      case _ if i == 0L =>
        NonEmpty(len, a1, this)
      case Empty =>
        sys.error("Shape.insert index out-of-bounds")
      case NonEmpty(len0, a0, rest) =>
        NonEmpty(len0, a0, rest.insert(i - 1, len, a1))
    }

  /**
   * Remove the last element of the shape, returning the rest of the
   * shape and the data.
   *
   * Unlike removing the first element of the shape, this method is
   * relatively expensive.
   *
   * Requires shape to be non-empty.
   */
  def last: (Shape[A], Long, A) =
    this match {
      case Empty =>
        sys.error("Shape.last called on Empty")
      case NonEmpty(len, a, Empty) =>
        (Empty, len, a)
      case NonEmpty(len0, a0, rest) =>
        val (s, len, a) = rest.last
        (NonEmpty(len0, a0, s), len, a)
    }
}

object Shape {
  // 1. coordinates = Shape[Coord] refers to a point (or an element)
  // 2. dimensions = Shape[Layout] refers to an axis along with a stride for data layout
  // 3. axes = Shape[Axis] refers to an axis (with no data layout)

  sealed trait Coord
  final val Coord = new Coord { override def toString: String = "Coord" }

  final case class Dim(offset: Long, stride: Long)

  val SingleDim = Dim(0L, 1L)
  val BroadcastDim = Dim(0L, 0L)

  sealed trait Axis
  final val Axis = new Axis { override def toString: String = "Axis" }

  type Coords = Shape[Coord]
  type Dims = Shape[Dim]
  type Axes = Shape[Axis]

  /**
   * This is used by select (Onnx.slice) to build new Dims
   * for a Tensor
   */
  final case class AxisRange(start: Long, end: Long, step: Long) {
    require(
      step != 0L,
      s"step may not be zero, AxisRange(start = $start, end = $end, step = $step)"
    )
  }
  object AxisRange {

    /**
     * Make a range with a default step size of 0
     */
    def apply(start: Long, end: Long): AxisRange = AxisRange(start, end, 1L)
  }

  /**
   * what is the shape you would get in numpy
   * see:
   * https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
   */
  def concatenate(axis: Long, s0: Axes, srest: Axes*): Try[Axes] =
    srest.toList.foldM(s0)(_.concatenate(axis, _))

  def coords(n: Long*): Coords =
    Shape(n.map((_, Coord)): _*)

  def axes(n: Long*): Axes =
    Shape(n.map((_, Axis)): _*)

  def dims(pairs: (Long, Long)*): Dims =
    Shape(pairs.map {
      case (1L, _)       => (1L, SingleDim)
      case (len, stride) => (len, Dim(0L, stride))
    }: _*)

  def rowMajorDims(ns: Long*): Dims =
    rowMajorDims(axes(ns: _*))

  def rowMajorDims(axes: Axes): Dims = {
    def recur(ax: Axes): (Long, Dims) =
      ax match {
        case Empty =>
          (1L, Empty)
        case NonEmpty(1, _, rest) =>
          val (stride, dims) = recur(rest)
          (stride, NonEmpty(1L, SingleDim, dims))
        case NonEmpty(len, _, rest) =>
          val (stride, dims) = recur(rest)
          (stride * len, NonEmpty(len, Dim(0L, stride), dims))
      }
    recur(axes)._2
  }

  def coordsToIndex(dims: Dims, coords: Coords): Long = {
    def recur(ds: Dims, cs: Coords): Long =
      (ds, cs) match {
        case (Empty, Empty) =>
          0
        case (NonEmpty(_, Dim(offset, stride), ds1), NonEmpty(i, _, cs1)) =>
          (i * stride + offset) + Shape.coordsToIndex(ds1, cs1)
        case (Empty, _) =>
          sys.error(s"too many coordinates ($coords) for dims ($dims)")
        case (_, Empty) =>
          sys.error(s"not enough coordinates ($coords) for dims ($dims)")
      }
    recur(dims, coords)
  }

  def isRowMajor(dims: Dims): Boolean = {
    def recur(ds0: Dims): Option[Long] =
      ds0 match {
        case Empty =>
          Some(1L)
        case NonEmpty(1L, SingleDim, ds) =>
          recur(ds)
        case NonEmpty(len, Dim(off, stride), ds) =>
          if (off == 0L) {
            recur(ds).flatMap { n =>
              if (n == stride) Some(n * len) else None
            }
          } else None
      }
    recur(dims).isDefined
  }

  implicit class AxesOps(val axes: Axes) extends AnyVal {

    /**
     * If we have two shapes that are equal in size in all but one axis, we can
     * concatenate them along the non-equal axis
     */
    def concatenate(axis: Long, rhs: Axes): Try[Axes] = {
      def invalidShape =
        Failure(new Exception(s"invalid concatenate, $axes.concatenate($axis, $rhs)"))

      def loop(axis: Long, lhs: Axes, rhs: Axes): Try[Axes] =
        if (axis == 0L) {
          (lhs, rhs) match {
            case (NonEmpty(len0, _, rest0), NonEmpty(len1, _, rest1)) if rest0 == rest1 =>
              Success(NonEmpty(len0 + len1, Axis, rest0))
            case _ => invalidShape
          }
        } else {
          (lhs, rhs) match {
            case (NonEmpty(len0, _, rest0), NonEmpty(len1, _, rest1)) if len0 == len1 =>
              loop(axis - 1L, rest0, rest1).map(NonEmpty(len0, Axis, _))
            case _ => invalidShape
          }
        }

      if (axis < 0L) invalidShape
      else loop(axis, axes, rhs)
    }

    def rowMajorIndex(coords: Coords): Long = {
      def recur(axes: Axes, coords: Coords): (Long, Long) =
        (axes, coords) match {
          case (Empty, Empty) =>
            (0L, 1L)
          case (NonEmpty(len, _, ax), NonEmpty(index, _, cs)) =>
            val (n, stride) = recur(ax, cs)
            (n + index * stride, stride * len)
          case (ax, cs) =>
            // $COVERAGE-OFF$
            sys.error(s"broken invariant: axes=$ax coords=$cs")
          // $COVERAGE-ON$
        }
      recur(axes, coords)._1
    }

    /**
     * see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
     * convert the shape of the current Axes into a new Axes
     */
    def inferReshapeTo(rhs: Axes): Try[Axes] = {
      // there are two weird values:
      // 0: means take the size of the current thing
      // -1: at most 1 dimension can be -1, meaning infer the dimension

      // remove all the zeros, this must be done before infering -1
      def fixZeros(lhs: Axes, rhs0: Axes): Try[Axes] =
        (lhs, rhs0) match {
          case (_, Empty) =>
            Success(Empty)
          case (NonEmpty(ls, _, lrest), NonEmpty(0, _, rrest)) =>
            fixZeros(lrest, rrest).map(NonEmpty(ls, Axis, _))
          case (Empty, NonEmpty(0, _, rrest)) =>
            fixZeros(Empty, rrest).map(NonEmpty(1L, Axis, _))
          case (NonEmpty(_, _, lrest), NonEmpty(rs, _, rrest)) =>
            fixZeros(lrest, rrest).map(NonEmpty(rs, Axis, _))
          case (Empty, NonEmpty(rs, _, rrest)) =>
            fixZeros(Empty, rrest).map(NonEmpty(rs, Axis, _))
        }

      // assuming
      def infer(rhs0: Axes): Try[Axes] = {
        val comps = rhs0.components
        if (comps.exists(_ == -1L)) {
          if (comps.count(_ == -1L) > 1)
            Failure(
              new IllegalArgumentException(
                s"in $axes.inferReshapeTo($rhs) cannot infer multiple dimensions"
              )
            )
          else {
            val inferIndex = comps.indexOf(-1L)
            // this get can't fail
            val (left, _, _, right) = rhs0.split(inferIndex).get
            val knownSize = left.totalSize * right.totalSize
            val leftSize = axes.totalSize
            val thisSize = leftSize / knownSize
            val rem = leftSize % knownSize
            if (rem != 0L)
              Failure(
                new IllegalArgumentException(
                  s"in $axes.inferReshapeTo($rhs) knownSize ($knownSize) does not divide $leftSize"
                )
              )
            else Success(left ++ (NonEmpty(thisSize, Axis, right)))
          }
        } else if (axes.totalSize == rhs0.totalSize) Success(rhs0)
        else
          Failure(
            new IllegalArgumentException(
              s"in $axes.inferReshapeTo($rhs) size mismatch: (${axes.totalSize}) vs $rhs0 (${rhs0.totalSize})"
            )
          )
      }

      fixZeros(axes, rhs).flatMap(infer)
    }

    def unsqueeze(axis: Long): Option[Axes] =
      axes match {
        case _ if axis == 0 =>
          Some(NonEmpty(1, Axis, axes))
        case NonEmpty(len, a, rest) =>
          rest.unsqueeze(axis - 1).map(NonEmpty(len, a, _))
        case Empty =>
          None
      }

    /**
     * convert to something of the same rank, with
     * totalSize == 1L
     */
    def toSingleton: Axes =
      axes match {
        case Empty => Empty
        case NonEmpty(_, _, rest) =>
          NonEmpty(1, Axis, rest.toSingleton)
      }

    def reduce(along: List[Long], keepDims: Boolean): Try[Axes] = {
      def loop(a: Axes, reds: List[Long], pos: Long): Try[Axes] =
        (a, reds) match {
          case (a, Nil) => Success(a)
          case (Empty, _) =>
            Failure(new Exception(s"unreduced axes: $axes.reduce($along, $keepDims)"))
          case (NonEmpty(c, _, rest), a0 :: arest) =>
            if (a0 == pos) {
              val restRes = loop(rest, arest, pos + 1L)
              if (keepDims) restRes.map(NonEmpty(1, Axis, _))
              else restRes
            } else loop(rest, reds, pos + 1L).map(NonEmpty(c, Axis, _))
        }
      loop(axes, along.sorted, 0L)
    }
  }

  implicit class DimsOps(val lhs: Dims) extends AnyVal {
    def coordsToIndex(coords: Coords): Long =
      Shape.coordsToIndex(lhs, coords)

    def isRowMajor: Boolean =
      Shape.isRowMajor(lhs)

    /**
     * see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
     * convert the shape of the current rowMajor Dims into a new Dims
     */
    def reshapeRowMajor(axes: Axes): Try[Dims] =
      if (Shape.isRowMajor(lhs)) {
        lhs.as(Axis).inferReshapeTo(axes).map(_.asRowMajorDims)
      } else Failure(new IllegalArgumentException(s"$lhs is not rowMajor in reshapeRowMajor"))

    /**
     * this is like numpy's slicing operator:
     * e.g. foo[0:10:1, 2:8:2]
     * We can always select, but may return effectively empty ranges
     */
    def select(range: List[AxisRange]): Dims = {
      def loop(lhs: Dims, rhs: List[AxisRange]): Dims =
        (lhs, rhs) match {
          case (Empty, _)  => Empty
          case (left, Nil) =>
            // we keep everything if the range is shorter than the dims
            left
          case (NonEmpty(len, Dim(off, stride), rest), AxisRange(start, end, step) :: rrest) =>
            val tail = loop(rest, rrest)
            // len1 is the smallest value that makes this not true when step > 0
            // start + len1 * step < end
            // when step < 0
            // start + len1 * step > end
            val diff = end - start
            // take the ceil
            val len1 =
              if (diff > 0L) {
                if (step > 1L) (diff + 1L) / step
                else if (step == 1L) diff
                else 0L
              } else if (diff < 0L) {
                if (step < -1L) (diff - 1L) / step
                else if (step == -1L) -diff
                else 0L
              } else 0L

            val dim1 = Dim(off + stride * start, stride * step)
            NonEmpty(len1, dim1, tail)
        }
      loop(lhs, range)
    }

    def squeeze(axis: Long): Option[Dims] =
      lhs match {
        case NonEmpty(1L, SingleDim, rest) if axis == 0L =>
          Some(rest)
        case NonEmpty(len, dim, rest) =>
          if (axis == 0) None else rest.squeeze(axis - 1).map(NonEmpty(len, dim, _))
        case Empty =>
          if (axis == 0) Some(Empty) else None
      }

    def unsqueeze(axis: Long): Option[Dims] =
      lhs match {
        case _ if axis == 0 =>
          Some(NonEmpty(1, SingleDim, lhs))
        case NonEmpty(len, dim, rest) =>
          rest.unsqueeze(axis - 1).map(NonEmpty(len, dim, _))
        case Empty =>
          None
      }

    def unchunk(axis: Long): Try[Dims] = {
      def invalid(msg: String): Try[Dims] =
        Failure(new Exception(s"axis $axis was not valid for $lhs: $msg"))
      def recur(ds: Dims, i: Long): Try[Dims] =
        ds match {
          case NonEmpty(len0, dim0, rest0) =>
            if (i == 0L) {
              rest0 match {
                case NonEmpty(len1, Dim(off1, stride1), rest1) =>
                  /*
                   * the formula we have to satisfy is this:
                   * (off0 + i * stride0) + (off1 + j * stride1)
                   * for all i in [0, len0) and j in [0, len1)
                   *
                   * gives the same indices as:
                   * (off2 + k * stride2) for k in [0, len0 * len1)
                   *
                   * this is true if off0 + off1 = off2
                   * and if i * stride0 + j * stride1 = k * stride2
                   * forms a 1:1 mapping from (i, j) <=> k
                   * this is true when stride1 = stride2
                   * and (stride0 = len1 * stride1)
                   */
                  val Dim(off0, stride0) = dim0
                  if (stride0 == len1 * stride1)
                    Success(NonEmpty(len1 * len0, Dim(off0 + off1, stride1), rest1))
                  else
                    invalid(s"expected $stride0 == $len1 * $stride1")
                case Empty =>
                  invalid("can't unchunk into scalar dimension")
              }
            } else {
              recur(rest0, i - 1L).map(r => NonEmpty(len0, dim0, r))
            }
          case Empty =>
            invalid("axis does not exist")
        }
      if (axis < 0L) invalid(s"illegal negative axis: $axis")
      else recur(lhs, axis)
    }

    def chunk(axis: Long, size: Long): Try[Seq[Dims]] = {
      // put this here where we capture the outer lhs and axis
      def invalid: Try[Seq[Dims]] =
        Failure(new Exception(s"dim = $lhs axis=$axis length is not divisable by $size"))

      def recur(axis: Long, lhs: Dims): Try[Seq[Dims]] =
        lhs match {
          case Empty => Failure(new Exception(s"we cannot chunk a scalar"))
          case NonEmpty(len, d, rest) =>
            if (axis == 0) {
              // we need to be able to divide len, and if so, add a bunch of offsets
              val Dim(off, stride) = d
              if (len % size == 0) {
                val chunks = len / size
                Success((0L until chunks).map { chunk =>
                  NonEmpty(size, Dim(off + chunk * size * stride, stride), rest)
                })
              } else invalid
            } else
              recur(axis - 1L, rest).map { chunks =>
                chunks.map(NonEmpty(len, d, _))
              }
        }

      if (axis < 0L || size < 0L)
        Failure(new IllegalArgumentException(s"invalid axis=$axis, size=$size"))
      else recur(axis, lhs)
    }

    /**
     * This is an implementation of unidirectional broadcasting.
     * Try to broadcast the lhs into the given Axes.
     */
    def broadcastTo(axes: Axes): Try[Dims] = {
      def recur(revdims: List[(Long, Dim)], revaxes: List[(Long, Axis)]): Try[List[(Long, Dim)]] =
        (revdims, revaxes) match {
          case (Nil, Nil) =>
            Success(Nil)
          case (Nil, (len, _) :: arest) =>
            if (len == 1) {
              recur(Nil, arest).map(dims => (1L, SingleDim) :: dims)
            } else {
              recur(Nil, arest).map(dims => (len, BroadcastDim) :: dims)
            }
          case ((len, _) :: drest, Nil) =>
            if (len == 1) {
              recur(drest, Nil)
            } else {
              Failure(new Exception(s"couldn't broadcast $len to 1 in $lhs.broadcastTo($axes"))
            }
          case ((dlen, dim) :: drest, (alen, _) :: arest) =>
            if (dlen == alen) {
              recur(drest, arest).map(dims => (dlen, dim) :: dims)
            } else if (dlen == 1) {
              recur(drest, arest).map(dims => (alen, BroadcastDim) :: dims)
            } else {
              Failure(
                new Exception(s"couldn't broadcast $dlen to $alen in $lhs.broadcastTo($axes)")
              )
            }
        }

      recur(lhs.toList.reverse, axes.toList.reverse)
        .map(Shape.reverseDimListToDims(_))
    }
  }

  sealed trait HasCoords[A] {
    def coords(s: Shape[A]): Iterator[Shape[Coord]]
  }

  object HasCoords {
    private class UnsafeCoords[A] extends HasCoords[A] {
      def coords(s: Shape[A]): Iterator[Coords] = {
        def iter(s0: Shape[A]): Iterator[Coords] =
          s0 match {
            case Empty =>
              Iterator(Empty)
            case NonEmpty(len, _, rest) =>
              for {
                x <- (0L until len).iterator
                cs <- iter(rest)
              } yield NonEmpty(x, Coord, cs)
          }
        iter(s)
      }
    }

    implicit val axisHasCoords: HasCoords[Axis] =
      new UnsafeCoords[Axis]

    implicit val dimHasCoords: HasCoords[Dim] =
      new UnsafeCoords[Dim]
  }

  /**
   * This is general broadcasting which can do broadcasting on
   * the left or right according to:
   *
   * https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
   */
  def broadcast(lhs: Dims, rhs: Dims): Try[(Dims, Dims)] = {
    type E = (Long, Dim)

    def recur(ldims: List[E], rdims: List[E]): Try[(List[E], List[E])] =
      (ldims, rdims) match {
        case (Nil, Nil) =>
          Success((Nil, Nil))
        case (Nil, (rp @ (rsz, rdim)) :: rrest) =>
          if (rsz == 1) {
            recur(Nil, rrest).map { case (lds, rds) => (rp :: lds, rp :: rds) }
          } else {
            recur(Nil, rrest).map { case (lds, rds) => ((rsz, BroadcastDim) :: lds, rp :: rds) }
          }
        case ((lp @ (lsz, ldim)) :: lrest, Nil) =>
          if (lsz == 1) {
            recur(lrest, Nil).map { case (lds, rds) => (lp :: lds, lp :: rds) }
          } else {
            recur(lrest, Nil).map { case (lds, rds) => (lp :: lds, (lsz, BroadcastDim) :: rds) }
          }
        case ((lp @ (lsz, ldim)) :: lrest, (rp @ (rsz, rdim)) :: rrest) =>
          if (lsz == rsz) {
            recur(lrest, rrest).map {
              case (lds, rds) =>
                (lp :: lds, rp :: rds)
            }
          } else if (lsz == 1) {
            recur(lrest, rrest).map {
              case (lds, rds) =>
                ((rsz, BroadcastDim) :: lds, rp :: rds)
            }
          } else if (rsz == 1) {
            recur(lrest, rrest).map {
              case (lds, rds) =>
                (lp :: lds, (lsz, BroadcastDim) :: rds)
            }
          } else {
            Failure(new Exception(s"couldn't broadcast $lsz and $rsz in broadcast($lhs, $rhs)"))
          }
      }

    recur(lhs.toList.reverse, rhs.toList.reverse).map {
      case (xs, ys) =>
        (reverseDimListToDims(xs), reverseDimListToDims(ys))
    }
  }

  private def reverseDimListToDims(xs: List[(Long, Dim)]): Dims =
    xs.foldLeft(Shape.empty[Dim]) {
      case (ds, (len, dim)) =>
        NonEmpty(len, dim, ds)
    }

  case object Empty extends Shape[Nothing]
  case class NonEmpty[A](size: Long, tag: A, rest: Shape[A]) extends Shape[A]

  def empty[A]: Shape[A] = Empty

  def apply[A](ns: (Long, A)*): Shape[A] =
    ns.foldRight(empty[A]) { case ((n, a), rest) => NonEmpty(n, a, rest) }

  implicit class HasCoordOps[A](val lhs: Shape[A]) extends AnyVal {
    def axesString: String =
      lhs.components.iterator.map(_.toString).mkString("x")

    def asRowMajorDims: Dims =
      Shape.rowMajorDims(lhs.as(Axis))

    def coords(implicit h: HasCoords[A]): Iterator[Shape[Coord]] =
      h.coords(lhs)

    def forall(p: Coords => Boolean)(implicit h: HasCoords[A]): Boolean =
      h.coords(lhs).forall(p)

    def foreach(f: Coords => Unit)(implicit h: HasCoords[A]): Unit =
      h.coords(lhs).foreach(f)

    // def foldMap[B](f: Coords => B)(implicit m: Monoid[B], h: HasCoords[A]): B =
    //   m.combineAll(h.coords(lhs).map(f))

    def totalSize(implicit h: HasCoords[A]): Long =
      lhs match {
        case Empty              => 1L
        case NonEmpty(n, _, ds) => n * ds.totalSize
      }
  }

  implicit val functorForShape: Traverse[Shape] =
    new Traverse[Shape] {
      override def map[A, B](s: Shape[A])(f: A => B): Shape[B] =
        s match {
          case Shape.NonEmpty(len, a, rest) => Shape.NonEmpty(len, f(a), map(rest)(f))
          case Shape.Empty                  => Shape.Empty
        }
      def traverse[F[_], A, B](
          s: Shape[A]
      )(f: A => F[B])(implicit ev: Applicative[F]): F[Shape[B]] =
        s match {
          case Shape.Empty =>
            ev.pure(Shape.Empty)
          case Shape.NonEmpty(len, a, rest) =>
            ev.map2(f(a), traverse(rest)(f)) { case (b, s) => NonEmpty(len, b, s) }
        }
      def foldLeft[A, B](s: Shape[A], b0: B)(f: (B, A) => B): B =
        s match {
          case Shape.Empty                => b0
          case Shape.NonEmpty(_, a, rest) => foldLeft(rest, f(b0, a))(f)
        }
      def foldRight[A, B](s: Shape[A], b0: Eval[B])(f: (A, Eval[B]) => Eval[B]): Eval[B] =
        s match {
          case Shape.Empty                => b0
          case Shape.NonEmpty(_, a, rest) => f(a, Eval.defer(foldRight(rest, b0)(f)))
        }
    }
}
