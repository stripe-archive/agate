package com.stripe.agate.tensor

import cats.implicits._
import shapeless._
import Shape._

/**
 * Shapeless-powered type-class used to build literal tensors.
 */
trait Build[H, E] {
  type Row

  def apply(h: H): (Dims, Coords => E)

  def build[D <: DataType { type Elem = E }](dt: D, h: H): Tensor[dt.type] = {
    implicit val alloc = StorageAllocator.forDataType(dt)
    val (dims, f) = apply(h)
    val n = dims.totalSize
    val data = alloc.allocate(n)
    var i = 0L
    dims.as(Axis).foreach { coord =>
      data.writeAt(i, f(coord))
      i += 1L
    }
    Tensor(dt, dims)(data.toStorage)
  }
}

object Build extends BuildLowPri {

  type Aux[H, F, E] = Build[H, F] { type Row = E }

  type Res[F] = (Dims, Coords => F)

  implicit def fromScalar[A]: Build.Aux[A, A, A] =
    new Build[A, A] {
      type Row = A
      def apply(x: A): Res[A] =
        (Empty, {
          case Empty => x
          case cs    => sys.error(s"invalid coords: $cs")
        })
    }

  implicit def fromTensor[E, A](implicit ev: Build[E, A]): Build.Aux[E :: HNil, A, E] =
    new Build[E :: HNil, A] {
      type Row = E
      def apply(h: E :: HNil): Res[A] = {
        val (ds, f) = ev(h.head)
        (NonEmpty(1, Shape.SingleDim, ds), {
          case NonEmpty(0, Coord, cs) => f(cs)
          case cs                     => sys.error(s"invalid coords: $cs")
        })
      }
    }

  implicit def fromTensors[H <: HList, A, E](
      implicit eve: Lazy[Build[E, A]],
      evh: Lazy[Build.Aux[H, A, E]]
  ): Build.Aux[E :: H, A, E] =
    new Build[E :: H, A] {
      type Row = E
      def apply(h: E :: H): Res[A] = {
        val (es, ef) = eve.value(h.head)
        val (NonEmpty(hlen, hdim, hds), hf) = evh.value(h.tail)
        val hdim2 = if (hlen == 1L) Dim(0L, es.totalSize) else hdim
        (NonEmpty(hlen + 1, hdim2, hds), {
          case Empty              => sys.error("invalid empty coords")
          case NonEmpty(0, _, cs) => ef(cs)
          case NonEmpty(n, _, cs) => hf(NonEmpty(n - 1, Coord, cs))
        })
      }
    }
}

trait BuildLowPri {
  implicit def buildFromGen[R, A, H <: HList](
      implicit gen: Generic.Aux[R, H],
      ev: Build[H, A]
  ): Build.Aux[R, A, ev.Row] =
    new Build[R, A] {
      type Row = ev.Row
      def apply(r: R): Build.Res[A] = ev(gen.to(r))
    }
}
