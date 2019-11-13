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
import scala.util.{Failure, Success, Try}

import Shape._

object EmbeddingBag {
  sealed abstract class Mode
  object Mode {
    case object Sum extends Mode
    case object Mean extends Mode
    case object Max extends Mode

    def fromEnum(i: Long): Option[Mode] = i match {
      case 0L => Some(Sum)
      case 1L => Some(Mean)
      case 2L => Some(Max)
      case _  => None
    }
  }

  // Validate and normalize the indices, offsets and per-index weights provided
  // for an EmbeddingBag operation. Regardless if `input` is a rank-1 or -2
  // tensor, this will return a rank-1 tensor of indices along with a list of
  // the "slices" that must be reduced in this tensor. It will also ensure that
  // `perIndexWeights` is reshaped to match the shape of `input`.
  private def normalizeInput(
      input: Tensor[DataType.Int64.type],
      offsets: Option[Tensor[DataType.Int64.type]],
      perIndexWeights: Option[Tensor[DataType.Float32.type]]
  ): Try[
    (Tensor[DataType.Int64.type], Vector[(Long, Long)], Option[Tensor[DataType.Float32.type]])
  ] =
    (input.dims, offsets) match {
      case (Shape.NonEmpty(len, _, Shape.Empty), Some(offsets)) =>
        if (offsets.rank != 1) {
          Failure(
            new IllegalArgumentException(
              s"required offsets tensor of rank 1, but rank=${offsets.rank}"
            )
          )
        } else {
          val offsetsVec: Vector[Long] = offsets.scalars.toVector
          if (offsetsVec.isEmpty) {
            Failure(new IllegalArgumentException("required non-empty offsets tensor"))
          } else if (offsetsVec.head != 0) {
            val offsetStr = offsetsVec.mkString(",")
            Failure(new IllegalArgumentException(s"offsets must start with 0, offsets=$offsetStr"))
          } else if (!offsetsVec.forall(i => i >= 0 && i <= len)) {
            // Note: use of i <= len (vs i < len) is intentional, as PyTorch allows this.
            val offsetStr = offsetsVec.mkString(",")
            Failure(new IllegalArgumentException(s"offsets out of bounds: offsets=$offsetStr"))
          } else {
            val slices = offsetsVec.zip(offsetsVec.tail :+ len)
            Success((input, slices, perIndexWeights))
          }
        }
      case (_, Some(_)) =>
        Failure(
          new IllegalArgumentException(s"expected input tensor of rank 1, but rank=${input.rank}")
        )
      case (Shape.NonEmpty(n, _, Shape.NonEmpty(m, _, Shape.Empty)), None) =>
        val slices = Vector.tabulate(n.toInt) { i =>
          (i * m, (i + 1) * m)
        }
        val flattened = Shape.axes(n * m)
        perIndexWeights match {
          case None => input.reshape(flattened).map((_, slices, None))
          case Some(weights) =>
            for {
              input0 <- input.reshape(flattened)
              weights0 <- weights.reshape(flattened)
            } yield (input0, slices, Some(weights0))
        }
      case (_, None) =>
        Failure(
          new IllegalArgumentException(s"expected input tensor of rank 2, but rank=${input.rank}")
        )
    }

  // If `perIndexWeights` is `None`, then this just returns `gathered` as-is.
  //
  // If `perIndexWeights` is defined, then scale each row in the `gathered`
  // matrix using the weights in the `perIndexWeights` vector. There must be as
  // many rows in `gathered` as there are weights (elements) in
  // `perIndexWeights`. The `mode` is required for validation - if
  // `perIndexWeights` is defined and `mode != Sum`, then this will fail with a
  // nice error message.
  private def applyWeights(
      gathered: Tensor[DataType.Float32.type],
      perIndexWeights: Option[Tensor[DataType.Float32.type]],
      mode: Mode
  ): Try[Tensor[DataType.Float32.type]] =
    perIndexWeights match {
      case Some(weights) if mode == Mode.Sum =>
        for {
          extendedWeights <- weights.reshape(weights.axes ++ Shape.axes(1))
          scaled <- Tensor.map2(DataType.Float32)(gathered, extendedWeights)(_ * _)
        } yield scaled
      case Some(_) =>
        Failure(new IllegalArgumentException(s"perIndexWeights not valid with mode=${mode}"))
      case None =>
        Success(gathered)
    }

  // Reduce each of the slices in the 2D matrix `gathered` using the specified
  // mode. This expects a 2D matrix to aggregate, where each row corresponds to
  // an embedding output by a lookup. The `slices` define the ranges of rows
  // that must be reduced. The output tensor will be a matrix with as many rows
  // as there are slices. Each row vector is computed by reducing - sum, mean,
  // element-wise max - all of the rows in `gathered` in the range for the
  // corresponding slice in `slices`.
  private def reduce(
      gathered: Tensor[DataType.Float32.type],
      slices: Vector[(Long, Long)],
      mode: Mode
  ): Try[Tensor[DataType.Float32.type]] = {
    val cols = gathered.dims.lengthOf(1).get
    val colsRange = Shape.AxisRange(0, cols)
    slices
      .traverse {
        case (from, until) if from < until =>
          val chunk = gathered.select(Shape.AxisRange(from, until) :: colsRange :: Nil)
          mode match {
            case Mode.Sum => chunk.sumAxes(0L)
            case Mode.Max => chunk.maxAxes(0L)
            case Mode.Mean =>
              val num = OnnxNumber.Float32
              val n = num.fromLong(until - from)
              chunk.foldMap(DataType.Float32, 0L :: Nil, false)(num.div(_, n), num.plus _)
          }

        case _ =>
          // This matches PyTorch's behaviour, which is to use 0 vectors when
          // the slice is empty.
          Success(Tensor.zero(Shape.axes(cols)))
      }
      .flatMap { chunks =>
        Tensor.stack(DataType.Float32, 0L)(chunks)
      }
  }

  /**
   * An implementation of PyTorch's EmbeddingBag operator. EmbeddingBags serve
   * 2 primary purposes, depending on your use case:
   *
   *  - EmbeddingBag is an operator that optimizes a group of N categorical
   *    embeddings that are then summed/averaged by instead training a single,
   *    large embedding for all categoricals and performing the sum in-place.
   *    This greatly reduces training time.
   *  - EmbeddingBag is an operator that allows dynamically-sized number of
   *    lookups into an embedding space, and summing all the results together.
   *    For example, looking up arbitrarily sized tokenized sentences to produce
   *    a mean embedding, such as would be done by something like FastText.
   *
   * EmbeddingBag allows 2 types of input - either a 2D matrix of indices is
   * provided, or a 1D vector of indices is provided along with a 1D vector of
   * "offsets." In either case, the goal is to take a chunk/vector of indices,
   * do a bunch of lookups into an embedding, and then reduce these embeddings
   * (sum, mean, max, etc) to a single vector per-row.
   *
   * If offsets is defined, then it must be a 1D vector of offsets and input
   * must be a 1D vector of long indices. The first offset must be 0 and no
   * offset can be larger than the length of input. The "chunks" that are then
   * reduced are defined by the any 2 adjacent offsets, defining the start
   * (inclusive) and end (exclusive) of the range of indices in input. The last
   * offset is implied to extend to the end of input. For example, if offsets
   * is [0, 2, 3, 6], and input is [0, 1, 5, 3, 9, 2, 1, 2], then the chunks
   * that must be reduced are [ [0, 1], [5], [3, 9, 2], [1, 2] ]. Each chunk is
   * dynamically sized, which is why this type of input can't be provided as a
   * 2D tensor instead.
   *
   * If `offsets` are not defined, then the input must be a 2D matrix of long
   * indices. Each row in the matrix is a chunk that must be reduced.
   *
   * If `perIndexWeights` is defined, then it must have the same shape as
   * `input` and `mode` must be `Sum`. The `perIndexWeights` are used to scale
   * the embeddings returned by each index we lookup in `data`, which is why
   * each weight must correspond to exactly 1 index in `input`.
   *
   * @param data a 2-D matrix of floats, each row is an embedding
   * @param mode the type of aggregation to perform on the embeddings
   * @param input a rank-2 or rank-1 (if offsets is defined) tensor of long
   *     indices
   * @param offsets an optional rank-1 tensor of long offsets into input
   * @param perIndexWeights an optional set of weights to apply to each lookup
   *     from input
   * @return a 2-D matrix of floats, the aggregated embeddings for the batch
   */
  def apply(
      data: Tensor[DataType.Float32.type],
      mode: Mode,
      input: Tensor[DataType.Int64.type],
      offsets: Option[Tensor[DataType.Int64.type]],
      perIndexWeights: Option[Tensor[DataType.Float32.type]]
  ): Try[Tensor[DataType.Float32.type]] =
    for {
      // This normalizes our input so that we always have a rank-1 tensor - a
      // vector of indices - and the corresponding slices we must reduce from
      // this tensor. This way we just always perform our sums as if a 1D input
      // was provided with offsets. We also need to re-adjust our per-index
      // weights to match the shape of the normalized input.
      (lookups, slices, weights) <- normalizeInput(input, offsets, perIndexWeights)
      gathered <- Gather(data, lookups, 0L)
      // If there are any per-index weights, we need to scale the embeddings
      // using them. This will also validate that perIndexWeights is valid with
      // the given mode.
      scaled <- applyWeights(gathered, weights, mode)
      // This will always produce the correct shape output, since the slices each
      // get reduced to a single row in a rank-2 output tensor.
      result <- reduce(scaled, slices, mode)
    } yield result
}
