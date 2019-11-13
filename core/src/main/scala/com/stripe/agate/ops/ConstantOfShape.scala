package com.stripe.agate.ops

import com.stripe.agate.tensor.{DataType, Shape, Tensor}
import scala.util.{Failure, Success, Try}

import Shape.Axes

object ConstantOfShape {
  def apply(
      dt: DataType
  )(
      value: Tensor[dt.type],
      shape: Tensor[DataType.Int64.type]
  ): Try[Tensor[dt.type]] = {

    val maybeAxes: Try[Axes] =
      if (shape.rank == 1) {
        val ns: List[Long] = shape.scalars.toList
        Success(Shape.axes(ns: _*))
      } else {
        Failure(new Exception(s"invalid axes for shape tensor: ${shape.axesString}"))
      }

    for {
      axes <- maybeAxes
      t <- value.broadcastTo(axes)
    } yield t
  }
}
