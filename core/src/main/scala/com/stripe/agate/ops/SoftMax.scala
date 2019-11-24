package com.stripe.agate.ops

import com.stripe.agate.tensor.{DataType, OnnxFloating, OnnxNumber, Tensor}
import scala.util.Try

object Softmax {
  /**
   * Run the actual softmax algorithm across an entire tensor.
   */
  private def softMax(
      dt: DataType
  )(fl: OnnxFloating[dt.Elem], t0: Tensor[dt.type]): Tensor[dt.type] = {
    val maxVal: dt.Elem = t0.max
    val t1: Tensor[dt.type] = t0.map(x => fl.exp(fl.minus(x, maxVal)))
    val expSum: dt.Elem = t1.sum
    t1.map { (x: dt.Elem) =>
      fl.div(x, expSum)
    }
  }

  /**
   * If axis=0 then run the softmax algorithm across the entire tensor.
   *
   * If axis>0, then we want to recurse on each of our "outer" slices
   * (along axis 0), and then recombine those back into their
   * respective places.
   *
   * TODO: For higher rank tensors, this process is currently somewhat
   * inefficient due to building intermediate tensors. We could
   * probably thread a single WritableStorage value through all of
   * this, but for a first pass the current approach works OK.
   */
  private def run(
      dt: DataType
  )(axis: Int, fl: OnnxFloating[dt.Elem], t0: Tensor[dt.type]): Tensor[dt.type] =
    if (axis == 0) {
      softMax(dt)(fl, t0)
    } else {
      val seq: Seq[Tensor[dt.type]] =
        t0.slices(0).map { case (_, slice) => run(dt)(axis - 1, fl, slice) }
      Tensor.stack(dt, 0)(seq).get
    }

  /**
   * Run Softmax.
   *
   * This method ensures that our axis is valid and that we have a
   * floating point type, and then dispatches to run to do the rest of
   * the job.
   */
  def apply[D <: DataType](input: Tensor[D], axis: Int): Try[Tensor[input.dataType.type]] =
    if (axis < 0) {
      Try(sys.error(s"invalid axis: $axis"))
    } else {
      val num: OnnxNumber[input.dataType.Elem] = OnnxNumber.forDataType(input.dataType)
      OnnxNumber.toFloating(num).flatMap { (fl: OnnxFloating[input.dataType.Elem]) =>
        Try(run(input.dataType)(axis, fl, input.asDataTyped))
      }
    }
}
