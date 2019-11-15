package com.stripe.agate.eval

import cats.Foldable
import cats.data.NonEmptyList
import cats.effect.IO
import com.stripe.agate.tensor.{DataType, OnnxFloating, OnnxNumber, Shape, StorageAllocator, Tensor}
import com.stripe.agate.ops.{
  BatchNormalization,
  EmbeddingBag,
  Gather,
  Gemm,
  LayerNorm,
  MaxPool,
  Softmax
}
import java.nio.file.{Path, Paths}
import scala.util.{Failure, Success, Try}

import cats.implicits._

sealed trait Operation {
  def apply(m: Registers): Try[Registers]
}

object Operation {

  def runAll[F[_]: Foldable](fo: F[Operation])(reg: Registers): Try[Registers] =
    fo.foldM(reg) { (reg, op) =>
      op(reg)
    }

  sealed abstract class FloatMapOp(input: Register, output: Register) extends Operation {
    def op[A](num: OnnxFloating[A]): A => A

    def apply(regs0: Registers): Try[Registers] =
      for {
        t0 <- regs0.get(input)
        num = OnnxNumber.forDataType(t0.dataType)
        ev <- OnnxNumber.toFloating(num)
        t1 = t0.map(op(ev))
        regs1 <- regs0.create(output, t1)
      } yield regs1
  }

  object FloatMapOp {
    case class Exp(in: Register, out: Register) extends FloatMapOp(in, out) {
      def op[A](num: OnnxFloating[A]) = num.exp
    }
    case class LeakyRelu(alpha: Double, in: Register, out: Register) extends FloatMapOp(in, out) {
      def op[A](num: OnnxFloating[A]) = {
        val negAlpha = num.minus(num.zero, num.fromDouble(alpha))
        // leakyrelu(a, x) = if (x < 0) -a*x else x = -a * relu(-x) + relu(x)

        { a: A =>
          val negA = num.minus(num.zero, a)
          num.plus(num.times(negAlpha, num.relu(negA)), num.relu(a))
        }
      }
    }
    case class Log(in: Register, out: Register) extends FloatMapOp(in, out) {
      def op[A](num: OnnxFloating[A]) = num.log
    }
    case class Relu(in: Register, out: Register) extends FloatMapOp(in, out) {
      def op[A](num: OnnxFloating[A]) = num.relu
    }
    case class Sigmoid(in: Register, out: Register) extends FloatMapOp(in, out) {
      def op[A](num: OnnxFloating[A]) = num.sigmoid
    }
  }

  /**
   * this is shared code for all the basic binary operators on tensors
   */
  sealed abstract class BinOp(input0: Register, input1: Register, output: Register)
      extends Operation {
    def op[A](num: OnnxNumber[A]): (A, A) => A

    final def apply(regs0: Registers): Try[Registers] =
      for {
        left0 <- regs0.get(input0)
        left = left0.asDataTyped
        right <- regs0.getAndUnify(input1, left.dataType)
        result <- Tensor.map2(left.dataType)(left, right)(op(OnnxNumber.forDataType(left.dataType)))
        regs1 <- regs0.create(output, result)
      } yield regs1
  }

  object BinOp {
    case class Add(left: Register, right: Register, out: Register) extends BinOp(left, right, out) {
      def op[A](num: OnnxNumber[A]): (A, A) => A = num.plus(_, _)
    }
    case class Sub(left: Register, right: Register, out: Register) extends BinOp(left, right, out) {
      def op[A](num: OnnxNumber[A]): (A, A) => A = num.minus(_, _)
    }
    case class Mul(left: Register, right: Register, out: Register) extends BinOp(left, right, out) {
      def op[A](num: OnnxNumber[A]): (A, A) => A = num.times(_, _)
    }
    case class Div(left: Register, right: Register, out: Register) extends BinOp(left, right, out) {
      def op[A](num: OnnxNumber[A]): (A, A) => A = num.div(_, _)
    }
  }

  case class CastOp(input: Register, castTo: DataType, output: Register) extends Operation {
    def apply(regs0: Registers): Try[Registers] =
      for {
        t0 <- regs0.get(input)
        t1 = t0.cast(castTo)
        regs1 <- regs0.create(output, t1)
      } yield regs1
  }

  case class ConcatOp(inputs: NonEmptyList[Register], axis: Long, output: Register)
      extends Operation {
    def apply(regs0: Registers): Try[Registers] =
      for {
        t0 <- regs0.get(inputs.head)
        thead = t0.asDataTyped
        ttail <- inputs.tail.traverse(regs0.getAndUnify(_, thead.dataType))
        out <- Tensor.concatenate(thead.dataType, axis)(thead :: ttail)
        regs1 <- regs0.create(output, out)
      } yield regs1
  }

  case class ConstantOfShape(input: Register, value: Option[Tensor.U], output: Register)
      extends Operation {

    private val getConst: Try[Shape.Axes => Tensor.U] = {
      value match {
        case None =>
          Success({ shape =>
            Tensor.const(DataType.Float32)(0.0f, shape)
          })
        case Some(ten) =>
          Try {
            // this should be a scalar, but just take the first item and be
            // loose in what we accept
            val elem = ten.scalars.next

            { shape =>
              Tensor.const(ten.dataType)(elem, shape)
            }
          }
      }
    }

    def apply(regs0: Registers): Try[Registers] =
      for {
        constFn <- getConst
        axesTen <- regs0.get(input)
        axes <- axesTen.convertDataToAxes
        regs1 <- regs0.create(output, constFn(axes))
      } yield regs1

  }

  case class ConstantOp(t: Tensor.Unknown, output: Register) extends Operation {
    def apply(regs: Registers): Try[Registers] =
      regs.create(output, t)
  }

  case class DropoutOp(input: Register, output: Register, maskOutput: Register, ratio: Float)
      extends Operation {
    def apply(regs0: Registers): Try[Registers] =
      for {
        t0 <- regs0.get(input)
        regs1 <- regs0.create(output, t0)
      } yield regs1
  }

  case class GemmOp(
      a: Register,
      b: Register,
      c: Register,
      alpha: Float,
      beta: Float,
      transA: Boolean,
      transB: Boolean,
      output: Register
  ) extends Operation {

    def apply(regs0: Registers): Try[Registers] =
      for {
        t0 <- regs0.get(a)
        ta = t0.asDataTyped
        tb <- regs0.getAndUnify(b, ta.dataType)
        tc <- regs0.getAndUnify(c, ta.dataType)
        tz <- Try(Gemm(ta.dataType)(ta, tb, tc, alpha, beta, transA, transB).get)
        regs1 <- regs0.create(output, tz)
      } yield regs1
  }

  case class GatherOp(data: Register, indices: Register, output: Register, axis: Long)
      extends Operation {
    def apply(regs0: Registers): Try[Registers] =
      for {
        t0 <- regs0.get(data)
        t1 <- regs0.get(indices)
        t2 <- Gather(t0, t1, axis)
        regs1 <- regs0.create(output, t2)
      } yield regs1
  }

  case class IdentityOp(input: Register, output: Register) extends Operation {
    def apply(regs0: Registers): Try[Registers] =
      for {
        t <- regs0.get(input)
        regs1 <- regs0.create(output, t)
      } yield regs1
  }

  case class MaxPoolOp(
      autoPad: MaxPool.AutoPad,
      ceilMode: Boolean,
      dilations: Option[List[Long]],
      kernelShape: List[Long],
      pads: Option[List[Long]],
      storageOrder: MaxPool.StorageOrder,
      strides: List[Long],
      input: Register,
      output: Register,
      indices: Option[Register]
  ) extends Operation {

    val indicesTry: Try[Unit] =
      indices match {
        case Some(_) => Failure(new Exception("indices output not yet supported in MaxPool"))
        case None    => Success(())
      }

    def apply(regs0: Registers): Try[Registers] =
      for {
        _ <- indicesTry
        t <- regs0.get(input)
        dt = t.asDataTyped
        realPads = pads.getOrElse(List.fill(2 * t.rank)(0L))
        MaxPool.Output(res) <- MaxPool(dt.dataType)(
          dt,
          autoPad,
          ceilMode,
          dilations,
          kernelShape,
          realPads,
          storageOrder,
          strides
        )
        regs1 <- regs0.create(output, res)
      } yield regs1
  }

  case class SliceOp(
      data: Register,
      starts: Register,
      ends: Register,
      axes: Option[Register],
      steps: Option[Register],
      output: Register
  ) extends Operation {
    def apply(regs0: Registers): Try[Registers] = {

      def getIndex(r: Register): Try[Array[Long]] =
        regs0.get(r).flatMap { ten =>
          if (ten.rank == 1) Success(ten.cast(DataType.Int64).scalars.toArray)
          else
            Failure(
              new Exception(s"expected rank 1, found: rank=${ten.rank} in register: $r, $ten")
            )
        }

      for {
        dt <- regs0.get(data)
        st <- getIndex(starts)
        et <- getIndex(ends)
        at <- axes.traverse(getIndex)
        stept <- steps.traverse(getIndex)
        selectRange <- SliceOp.buildSelection(dt.axes, st, et, at, stept)
        res = dt.select(selectRange)
        regs1 <- regs0.create(output, res)
      } yield regs1
    }
  }

  object SliceOp {
    def buildSelection(
        dataAxes: Shape.Axes,
        starts: Array[Long],
        ends: Array[Long],
        axes: Option[Array[Long]],
        steps: Option[Array[Long]]
    ): Try[List[Shape.AxisRange]] = Try {
      val rank = dataAxes.rank

      val axesArray = dataAxes.toList.iterator.map(_._1).toArray

      val realAxes: Map[Int, Long] =
        (axes match {
          case Some(as) => as.toSeq
          case None     => 0L until rank.toLong
        }).zipWithIndex.map(_.swap).toMap

      val realSteps: Array[Long] =
        steps match {
          case Some(as) => as.toArray
          case None     => Array.fill(rank)(1L)
        }

      def fullRange(axis: Long): Shape.AxisRange = {
        val max = axesArray(axis.toInt)
        Shape.AxisRange(0L, max)
      }

      val fullAxes: Iterator[Long] =
        (0L until rank.toLong).filterNot(realAxes.values.toSet).iterator
      val explicit = realAxes.iterator.map {
        case (idx, axis) =>
          val maxV = axesArray(axis.toInt)
          val start0 = if (idx < starts.length) starts(idx) else 0L
          val start = if (start0 < 0) {
            // means count back from the end
            axesArray(axis.toInt) + start0
          } else start0

          val end0 = if (idx < ends.length) ends(idx) else maxV
          val end = if (end0 < 0) {
            axesArray(axis.toInt) + end0
          } else end0
          val step = if (idx < realSteps.length) realSteps(idx) else 1
          def clamp(i: Long): Long = if (i < 0L) 0L else if (i > maxV) maxV else i
          (axis, Shape.AxisRange(clamp(start), clamp(end), step))
      }

      (explicit ++ (fullAxes.map { axis =>
        (axis, fullRange(axis))
      })).toList.sortBy(_._1).map(_._2)

    }

  }

  case class TransposeOp(permOpt: Option[List[Long]], input: Register, output: Register)
      extends Operation {
    def apply(regs0: Registers): Try[Registers] =
      for {
        t <- regs0.get(input)
        perm = permOpt.getOrElse((0L until t.rank).toList.reverse)
        tt <- t.transpose(perm)
        regs1 <- regs0.create(output, tt)
      } yield regs1
  }

  case class UnsqueezeOp(input: Register, axes: List[Long], output: Register) extends Operation {
    val axes1 = axes.sorted.reverse // largest-first
    def apply(regs0: Registers): Try[Registers] =
      for {
        tt <- regs0.get(input)
        t0 = tt.asDataTyped
        t1 <- Try(
          axes1
            .foldM(t0) { (t: Tensor[t0.dataType.type], i) =>
              (t.unsqueeze(i): Option[Tensor[t0.dataType.type]])
            }
            .get
        )
        regs1 <- regs0.create(output, t1)
      } yield regs1
  }

  case class ReduceSumOp(
      axes: Option[List[Long]],
      keepDims: Boolean,
      input: Register,
      output: Register
  ) extends Operation {
    def reduce(t: Tensor.U): Try[Tensor.U] = {
      val a = axes.getOrElse(Nil)
      if (keepDims) t.sumAxesKeep(a: _*)
      else t.sumAxes(a: _*)
    }

    def apply(regs0: Registers): Try[Registers] =
      for {
        t0 <- regs0.get(input)
        t1 <- reduce(t0)
        regs1 <- regs0.create(output, t1)
      } yield regs1
  }

  case class ReshapeOp(input: Register, shape: Register, output: Register) extends Operation {
    def apply(regs0: Registers): Try[Registers] =
      for {
        t0 <- regs0.get(input)
        shapeTen <- regs0.get(shape)
        shape <- shapeTen.convertDataToAxes
        t1 <- t0.reshape(shape)
        regs1 <- regs0.create(output, t1)
      } yield regs1
  }

  case class ShapeOp(input: Register, output: Register) extends Operation {
    def apply(regs0: Registers): Try[Registers] =
      for {
        t0 <- regs0.get(input)
        shape = Tensor.convertAxesToTensor(t0.axes)
        regs1 <- regs0.create(output, shape)
      } yield regs1
  }

  case class SqueezeOp(input: Register, axes: List[Long], output: Register) extends Operation {
    val axes1 = axes.sorted.reverse // largest-first
    def apply(regs0: Registers): Try[Registers] =
      for {
        tt <- regs0.get(input)
        t0 = tt.asDataTyped
        t1 <- Try(
          axes1
            .foldM(t0) { (t, i) =>
              (t.squeeze(i): Option[Tensor[t0.dataType.type]])
            }
            .get
        )
        regs1 <- regs0.create(output, t1)
      } yield regs1
  }

  /*
   Node(input = (24, 9, 10, 11, 12), output = (25), opType = BatchNormalization,
   attrs = (epsilon: FLOAT = 1.0E-5, momentum: FLOAT = 1.0))
   */

  // data: (N x C x D1 x D2 ... Dn)
  // scale: C
  // bias: C
  // mean: C
  // variance: C
  // output: (N x C x D1 x D2 ... Dn)
  //
  // for each slice of C:
  //
  //    ((slice - mean) / sqrt(variance + epsilon)) * scale + bias
  case class BatchNormalizationOp(
      data: Register,
      scale: Register,
      bias: Register,
      mean: Register,
      variance: Register,
      epsilon: Float,
      momentum: Float,
      output: Register
  ) extends Operation {

    def apply(regs: Registers): Try[Registers] =
      for {
        t0 <- regs.get(data)
        d = t0.asDataTyped
        s <- regs.getAndUnify(scale, d.dataType)
        b <- regs.getAndUnify(bias, d.dataType)
        m <- regs.getAndUnify(mean, d.dataType)
        v <- regs.getAndUnify(variance, d.dataType)
        t <- BatchNormalization(d.dataType)(d, s, b, m, v, epsilon)
        regs1 <- regs.create(output, t)
      } yield regs1
  }

  case class SoftmaxOp(input: Register, axis: Int, output: Register) extends Operation {
    def apply(regs: Registers): Try[Registers] =
      for {
        t0 <- regs.get(input)
        t1 = t0.asDataTyped
        t2 <- Softmax(t0, axis)
        regs1 <- regs.create(output, t2)
      } yield regs1
  }

  case class ATenLayerNorm(
      input: Register,
      weight: Register,
      bias: Register,
      output: Register,
      normalizedShape: List[Long],
      eps: Double
  ) extends Operation {
    def apply(regs: Registers): Try[Registers] =
      for {
        t0 <- regs.get(input)
        d = t0.asDataTyped
        w <- regs.getAndUnify(weight, d.dataType)
        b <- regs.getAndUnify(bias, d.dataType)
        res <- LayerNorm(d.dataType)(d, normalizedShape, Some(w), Some(b), eps)
        regs1 <- regs.create(output, res)
      } yield regs1
  }

  case class ATenEmbeddingBag(
      weight: Register,
      input: Register,
      offsets: Register,
      output: Register,
      mode: EmbeddingBag.Mode
  ) extends Operation {
    def apply(regs: Registers): Try[Registers] =
      for {
        w <- regs.getAndUnify(weight, DataType.Float32)
        i <- regs.getAndUnify(input, DataType.Int64)
        o <- if (i.rank == 2) Success(None)
        else regs.getAndUnify(offsets, DataType.Int64).map(Some(_))
        res <- EmbeddingBag(w, mode, i, o, None)
        regs1 <- regs.create(output, res)
      } yield regs1
  }

  case class NonZeroOp(input: Register, output: Register) extends Operation {
    def apply(regs0: Registers): Try[Registers] =
      for {
        t0 <- regs0.get(input)
        result = t0.nonZero
        regs1 <- regs0.create(output, result)
      } yield regs1
  }
}
