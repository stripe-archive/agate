package com.stripe.agate.ops

import com.stripe.agate.tensor.{DataType, OnnxFloating, OnnxNumber, Shape, StorageAllocator, Tensor}
import scala.util.Try

import DataType.Int32
import Shape.{Coord, Coords, Dims, Empty, NonEmpty}

object MaxPool {

  def apply(
      dt: DataType
  )(
      input: Tensor[dt.type],
      autoPad: AutoPad,
      ceilMode: Boolean,
      dilations: Option[List[Long]],
      kernelShape: List[Long],
      pads: List[Long],
      storageOrder: StorageOrder,
      strides: List[Long]
  ): Try[Output[dt.type]] = {

    val on: OnnxNumber[dt.Elem] = OnnxNumber.forDataType(dt)

    require(autoPad == AutoPad.NotSet)
    require(storageOrder == StorageOrder.RowMajor)
    require(dilations == None)

    val dimN :: dimC :: dimDs = input.dims.components
    val n = dimDs.size

    require(kernelShape.size == n, s"invalid kernel: $kernelShape ($n)")
    require(pads.size == (n * 2) && pads.forall(_ == 0), s"invalid pads: $pads ($n)")
    require(strides.size == n, s"invalid strides: $strides ($n)")

    def ceildiv(n: Long, d: Long): Long = (n + d - 1L) / d
    def floordiv(n: Long, d: Long): Long = n / d

    val triples = (dimDs zip (kernelShape zip strides))
    val outDs = triples.map {
      case (d, (k, s)) =>
        if (ceilMode) ceildiv(d - k, s) + 1L
        else floordiv(d - k, s) + 1L
    }

    val outputAxes = Shape.axes((dimN :: dimC :: outDs.toList): _*)
    val outputSize = outputAxes.totalSize
    implicit val alloc = StorageAllocator.forDataType(dt)
    val writable = alloc.allocate(outputSize)
    (0L until outputSize).foreach(writable.writeAt(_, on.minimum))
    val outputDims = outputAxes.asRowMajorDims

    /*
stride2
kernel=5

kernel / stride = 2.5 (2 or 3)

out + input * stride

(input - kernel + 1) / stride = 0 (round up)
(input - kernel + stride) / stride = earliest
               input / stride = latest (round down)

   input --->
   0 1 2 3 4 5 6 7 8 9 10
o  a b c d e f g h i j k
u  0 0 0 0 0
t      1 1 1 1 1
           2 2 2 2 2
               3 3 3 3 3
                   [4 4 4]
                       [5]

     */

    // (0, 2)

    // stride=2
    // kernel=3
    // ceil=false
    //
    // 0 1 2 3 4 5 6 7 8
    // a b c d e f g h i

    def coordsToOutputCoords(
        ds: List[Long],
        kernel: List[Long],
        stride: List[Long]
    ): Coords => Iterator[Coords] =
      (ds, kernel, stride) match {
        case (d :: ds, k :: ks, s :: ss) =>
          val f = coordsToOutputCoords(ds, ks, ss)
          val firstIncomplete = Math.max((d - k + s) / s, 0L)

          if ((!ceilMode) && (firstIncomplete == 0)) {
            { (coords: Coords) =>
              Iterator.empty
            }
          } else {
            { (coords: Coords) =>
              coords match {
                case NonEmpty(c, _, crest) =>
                  val start = Math.max((c - k + s) / s, 0L)
                  val end0 = c / s
                  // if we're not in ceil mode and we "go off the end"
                  // of a kernel, we need to truncate our range
                  val end = if (ceilMode || firstIncomplete > end0) end0 else (firstIncomplete - 1L)
                  val range = (start to end)
                  //println(s"(d=$d, c=$c, k=$k, s=$s) -> start=$start end0=$end0 end=$end firstIncomplete = $firstIncomplete")
                  for {
                    cs <- f(crest) // recurse once first, rather than on each inner value
                    n <- range
                  } yield NonEmpty(n, Coord, cs)
                case x =>
                  sys.error(s"mismatch args should not happen: $ds $x $kernel $stride")
              }
            }
          }
        case (Nil, Nil, Nil) => { (coords: Coords) =>
          coords match {
            case Empty => Iterator.single(Empty)
            case x     => sys.error(s"mismatch args should not happen: $ds $x $kernel $stride")
          }
        }
        case (w, y, z) =>
          sys.error(s"illegal builder: $w $y $z")
      }

    def prependNC(coords: Coords): Coords =
      NonEmpty(dimN, Coord, NonEmpty(dimC, Coord, coords))

    val makeIterator: Coords => Iterator[Coords] =
      coordsToOutputCoords(dimDs, kernelShape, strides)

    input.slices(axis = 0).foreach {
      case (ni, slice0) =>
        slice0.slices(axis = 0).foreach {
          case (ci, slice1) =>
            slice1.dims.coords.foreach { coords =>
              val value: dt.Elem = slice1(coords)
              makeIterator(coords).foreach { outPart =>
                val outCoords = NonEmpty(ni, Coord, NonEmpty(ci, Coord, outPart))
                val outIndex: Long = Shape.coordsToIndex(outputDims, outCoords)
                if (outIndex < outputSize) {
                  val currMax = writable(outIndex)
                  writable.writeAt(outIndex, on.max(currMax, value))
                } else {
                  println(s"${coords.components} -> ${outPart.components}")
                  println(s"index=$outIndex size=$outputSize")
                  sys.error("invalid index")
                }
              }
            }
        }
    }

    Try(Output(Tensor(dt, outputDims)(writable.toStorage)))
  }

  sealed abstract class AutoPad
  object AutoPad {
    case object NotSet extends AutoPad
    case object SameUpper extends AutoPad
    case object SameLower extends AutoPad
    case object Valid extends AutoPad

    def fromString(s: String): Option[AutoPad] =
      s match {
        case "NOTSET"     => Some(NotSet)
        case "SAME_UPPER" => Some(SameUpper)
        case "SAME_LOWER" => Some(SameLower)
        case "VALID"      => Some(Valid)
        case _            => None
      }

    def default: AutoPad = NotSet
  }

  sealed abstract class StorageOrder
  object StorageOrder {
    case object RowMajor extends StorageOrder
    case object ColMajor extends StorageOrder

    def default: StorageOrder = RowMajor
  }

  case class Output[D <: DataType](output: Tensor[D])
}
