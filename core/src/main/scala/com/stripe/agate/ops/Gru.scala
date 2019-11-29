package com.stripe.agate.ops

import com.stripe.agate.tensor.{DataType, OnnxFloating, OnnxNumber, Shape, Tensor}
import scala.util.Try

import DataType.Int32

/**
 * GRU
 *
 * Computes an one-layer GRU. This operator is usually supported via
 * some custom implementation such as CuDNN.
 *
 * See https://github.com/onnx/onnx/blob/master/docs/Operators.md#GRU
 *
 * Notations:
 * - X              - input tensor
 * - z              - update gate
 * - r              - reset gate
 * - h              - hidden gate
 * - t              - time step (t-1 means previous time step)
 * - W[zrh]         - W parameter weight matrix for update, reset, and hidden gates
 * - R[zrh]         - R recurrence weight matrix for update, reset, and hidden gates
 * - Wb[zrh]        - W bias vectors for update, reset, and hidden gates
 * - Rb[zrh]        - R bias vectors for update, reset, and hidden gates
 * - WB[zrh]        - W parameter weight matrix for backward update, reset, and hidden gates
 * - RB[zrh]        - R recurrence weight matrix for backward update, reset, and hidden gates
 * - WBb[zrh]       - W bias vectors for backward update, reset, and hidden gates
 * - RBb[zrh]       - R bias vectors for backward update, reset, and hidden gates
 * - H              - Hidden state
 * - num_directions - 2 if direction == bidirectional else 1
 *
 * Activation functions (first 3 required, rest optional):
 * - Relu(x)                - max(0, x)
 * - Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
 * - Sigmoid(x)             - 1/(1 + e^{-x})
 *   (optional below this line)
 * - Affine(x)              - alpha*x + beta
 * - LeakyRelu(x)           - x if x >= 0 else alpha * x
 * - ThresholdedRelu(x)     - x if x >= alpha else 0
 * - ScaledTanh(x)          - alpha*Tanh(beta*x)
 * - HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
 * - Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
 * - Softsign(x)            - x/(1 + |x|)
 * - Softplus(x)            - log(1 + e^x)
 *
 * Equations (Default: f=Sigmoid, g=Tanh):
 * - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
 * - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
 * - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
 * - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
 * - Ht = (1 - zt) (.) ht + zt (.) Ht-1
 */
object Gru {
  // D must be floating
  // bias defaults to 0 if missing
  def apply(
      dt: DataType
  )(
      input: Tensor[dt.type], // seq_length     x batch_size      x input_size
      weight: Tensor[dt.type], // num_directions x (3*hidden_size) x input_size
      recurrence: Tensor[dt.type], // num_directions x (3*hidden_size) x hidden_size
      bias0: Option[Tensor[dt.type]], // num_directions x (6*hidden_size)
      sequenceLens: Option[Tensor[Int32.type]], //                  batch_size
      initialH: Option[Tensor[dt.type]], // num_directions x batch_size      x hidden_size
      activationAlpha: List[Float],
      activationBeta: List[Float],
      activations: List[Gru.ActivationFn],
      clip: Float,
      direction: Gru.Direction,
      hiddenSize: Int,
      linearBeforeReset: Boolean
  ): Try[Output[dt.type]] =
    Try {
      val numDirections = direction.number
      val List(seqLength, batchSize, inputSize) = input.dims.components

      // println(s"input=${input.axes}")
      // println(s"weight=${weight.axes}")

      val of: OnnxFloating[dt.Elem] = OnnxNumber.toFloating(OnnxNumber.forDataTypeOrThrow(dt)).get

      // [ [ a b c d e f ] ]
      // splitAt(axis = 1, size = 2)
      // [[a b]] , [[c d]], [[e f ]]

      // split weight into these 3, each num_directions x hidden_size x input_size
      val weightHead = weight.slice(axis = 0, 0)
      val Seq(wz, wr, wh) = weightHead.chunk(axis = 0, size = hiddenSize).get

      // split recurrence into these 3, each num_directions x hidden_size x hidden_size
      val recurHead = recurrence.slice(axis = 0, 0)
      val Seq(rz, rr, rh) = recurHead.chunk(axis = 0, size = hiddenSize).get

      // split bias into these 6, each num_directions x hidden_size
      val bias =
        bias0.getOrElse(Tensor.const(dt)(of.zero, Shape.axes(numDirections, 6L * hiddenSize)))
      val biasHead = bias.slice(axis = 0, 0)
      val Seq(wbz, wbr, wbh, rbz, rbr, rbh) = biasHead.chunk(axis = 0, size = hiddenSize).get
      // println(s"biasHead = ${biasHead.dims}")
      // println(s"wbz = ${wbz.dims}")
      // println(s"wbr = ${wbr.dims}")
      // println(s"wbh = ${wbh.dims}")
      // println(s"rbz = ${rbz.dims}")
      // println(s"rbr = ${rbr.dims}")
      // println(s"rbh = ${rbh.dims}")

      type Fn = dt.Elem => dt.Elem

      def mult(x: Tensor[dt.type], y: Tensor[dt.type], z: Tensor[dt.type]): Tensor[dt.type] =
        //println(s"mult: ${x.axesString} * ${y.axesString} + ${z.axesString}")
        Gemm(dt)(x, y, z, 1f, 1f, false, true).get

      def pointwise(x: Tensor[dt.type], y: Tensor[dt.type]): Tensor[dt.type] =
        Tensor.map2(dt)(x, y)(of.times(_, _)).get

      def add(x: Tensor[dt.type], y: Tensor[dt.type]): Tensor[dt.type] =
        //println(s"add: ${x.axesString} + ${y.axesString}")
        Tensor.map2(dt)(x, y)(of.plus(_, _)).get

      def subtract(x: Tensor[dt.type], y: Tensor[dt.type]): Tensor[dt.type] =
        Tensor.map2(dt)(x, y)(of.minus(_, _)).get

      val one: Tensor[dt.type] = Tensor.const(dt)(of.one, Shape.Empty)

      val hiddenAxes =
        Shape.axes(batchSize, hiddenSize)

      val hidden0: Tensor[dt.type] =
        Tensor.const(dt)(of.zero, hiddenAxes)

      // Xt: batch_size x input_size
      // Wz^T: input_size x hidden_size
      // (Xt * Wz^T): batch_size x hidden_size
      // Wbz: hidden_size (broadcasted to: batch_size x hidden_size)
      def step(Xt: Tensor[dt.type], Htm1: Tensor[dt.type], f: Fn, g: Fn): Tensor[dt.type] = {
        // println("step")
        // println(s"Xt=${Xt.axes}")
        // println(s"Htm1=${Htm1.axes}")
        val zt = {
          val tmp0 = mult(Xt, wz, wbz)
          //println(s"xxy: ${rbz.dims}")
          val tmp1 = mult(Htm1, rz, rbz)
          //println("yyz")
          add(tmp0, tmp1).map(f)
        }
        val rt = add(mult(Xt, wr, wbr), mult(Htm1, rr, rbr)).map(f)

        val ht = {
          val tmp0 = mult(Xt, wh, wbh)
          val tmp1 = if (linearBeforeReset) {
            pointwise(rt, mult(Htm1, rh, rbh))
          } else {
            mult(pointwise(rt, Htm1), rh, rbh)
          }
          add(tmp0, tmp1).map(g)
        }

        val Ht = {
          val tmp0 = pointwise(subtract(one, zt), ht)
          val tmp1 = pointwise(zt, Htm1)
          add(tmp0, tmp1)
        }
        Ht
      }

      def runForward(
          f: dt.Elem => dt.Elem,
          g: dt.Elem => dt.Elem
      ): Vector[Tensor[dt.type]] = {
        val slices = input.slices(axis = 0).map(_._2)
        // println(s"slices = ${slices.map(_.axes)}")
        // println(s"initialH = ${initialH.map(_.axes)}")
        // println(s"hidden0 = ${hidden0.axes}")
        val h0: Tensor[dt.type] = initialH.getOrElse(hidden0)
        slices.scanLeft(h0)((Htm1, Xt) => step(Xt, Htm1, f, g)).drop(1).toVector
      }

      def runReverse(
          f: dt.Elem => dt.Elem,
          g: dt.Elem => dt.Elem
      ): Vector[Tensor[dt.type]] = {
        val slices = input.slices(axis = 0).map(_._2)
        // println(s"slices = ${slices.map(_.axes)}")
        // println(s"initialH = ${initialH.map(_.axes)}")
        // println(s"hidden0 = ${hidden0.axes}")
        val h0: Tensor[dt.type] = initialH.getOrElse(hidden0)
        slices.toVector.scanRight(h0)(step(_, _, f, g)).init
      }

      direction match {
        case Gru.Direction.Forward =>
          val List(f, g) = activations.map(_.getFn(of))
          val hs = runForward(f, g)
          val y: Tensor[dt.type] = Tensor.stack(dt, axis = 0L)(hs).get
          val yH = hs.last
          Gru.Output(y, yH)
        case Gru.Direction.Reverse =>
          val List(f, g) = activations.map(_.getFn(of))
          val hs = runReverse(f, g)
          val y: Tensor[dt.type] = Tensor.stack(dt, axis = 0L)(hs).get
          val yH = hs.head
          Gru.Output(y, yH)
        case Gru.Direction.Bidirectional =>
          val List(f, g, rf, rg) = activations.map(_.getFn(of))
          val hs0 = runForward(f, g)
          val hs1 = runReverse(rf, rg)
          val hs = hs0.zip(hs1).map {
            case (t1, t2) =>
              Tensor.stack(dt, axis = 0L)(t1 :: t2 :: Nil).get
          }
          val y: Tensor[dt.type] = Tensor.unchunk(dt, axis = 0L)(hs).get
          val yH = Tensor.unchunk(dt, axis = 0L)(hs0.last :: hs1.head :: Nil).get
          Gru.Output(y, yH)
      }
    }

  sealed abstract class ActivationFn {
    def getFn[A](on: OnnxFloating[A]): A => A =
      this match {
        case ActivationFn.Relu    => on.relu
        case ActivationFn.Sigmoid => on.sigmoid
        case ActivationFn.Tanh    => on.tanh
      }
  }

  object ActivationFn {
    case object Relu extends ActivationFn
    case object Tanh extends ActivationFn
    case object Sigmoid extends ActivationFn
  }

  sealed abstract class Direction {
    def number: Int =
      if (this == Direction.Bidirectional) 2 else 1
  }

  object Direction {
    case object Forward extends Direction
    case object Reverse extends Direction
    case object Bidirectional extends Direction

    def apply(s: String): Direction =
      s match {
        case "forward"       => Forward
        case "reverse"       => Reverse
        case "bidirectional" => Bidirectional
      }
  }

  case class Output[D <: DataType](
      y: Tensor[D], // seq_length x num_directions x batch_size x hidden_size
      yH: Tensor[D] //              num_directions x batch_size x hidden_size
  )
}
