package com.stripe.agate.eval

import cats.data.NonEmptyList
import cats.arrow.FunctionK
import com.google.protobuf.ByteString
import com.stripe.agate.tensor.{DataType, Shape, Storage, StorageAllocator, Tensor}
import com.stripe.agate.ops.{BatchNormalization, EmbeddingBag, Gather, Gemm, MaxPool}
import cats.ApplicativeError
import cats.effect.{IO, Resource}
import cats.implicits._
import java.nio.file.{Path, Paths}
import java.util.Arrays
import scala.util.{Failure, Success, Try}

import onnx.onnx.{
  AttributeProto,
  GraphProto,
  ModelProto,
  NodeProto,
  TensorProto,
  TensorShapeProto,
  TypeProto,
  ValueInfoProto
}
import AttributeProto.AttributeType
import AttributeType.INT

import Shape.Axes

object Model {
  private implicit class SuccessEnrichment[A](val a: A) extends AnyVal {
    def success: Try[A] = Success(a)
  }

  def dataTypeToInt(dt: DataType): Int =
    dt match {
      // Undefined => 0
      case DataType.Float32 => 1
      case DataType.Uint8   => 2
      case DataType.Int8    => 3
      case DataType.Uint16  => 4
      case DataType.Int16   => 5
      case DataType.Int32   => 6
      case DataType.Int64   => 7
      case DataType.String  => 8
      // Boolean => 9
      case DataType.Float16 => 10
      case DataType.Float64 => 11
      // Uint32 => 12
      // Uint64 => 13
      // Complex64 => 14
      // Complex128 => 15
      case DataType.BFloat16 => 16
    }

  def dataTypeFromInt(o: Option[Int]): Try[DataType] =
    o match {
      case None    => Failure(new Exception(s"no data type given"))
      case Some(n) => dataTypeFromInt(n)
    }

  def dataTypeFromInt(n: Int): Try[DataType] =
    n match {
      case 1     => Success(DataType.Float32)
      case 2     => Success(DataType.Uint8)
      case 3     => Success(DataType.Int8)
      case 4     => Success(DataType.Uint16)
      case 5     => Success(DataType.Int16)
      case 6     => Success(DataType.Int32)
      case 7     => Success(DataType.Int64)
      case 8     => Success(DataType.String)
      case 10    => Success(DataType.Float16)
      case 11    => Success(DataType.Float64)
      case 16    => Success(DataType.BFloat16)
      case other => Failure(new Exception(s"unknown data type: $other"))
    }

  case class OpData(
      context: Any, // only used to call .toString in error messages
      inputs: Seq[Register],
      outputs: Seq[Register],
      attrs: Map[String, AttributeProto]
  )

  class OpBuilder(val name: String, val build: OpData => Try[Operation])

  object OpBuilder {
    def apply(name: String)(f: OpData => Try[Operation]): OpBuilder =
      new OpBuilder(name, f)
  }

  private def toTry[A](context: Any, opt: Option[A]): Try[A] =
    opt match {
      case None =>
        Failure(new Exception(s"expected a nonEmpty Option. invalid: $context"))
      case Some(a) => Success(a)
    }

  // indexed using the "operation" attribute since all of these are
  // for the ATen operator (i.e. the operator name will be "ATen").
  val atenBuilders: List[OpBuilder] =
    List(
      OpBuilder("layer_norm") { (x: OpData) =>
        Try {
          val input = x.inputs(0)
          val weight = x.inputs(1)
          val bias = x.inputs(2)
          val output = x.outputs(0)
          val shape: List[Long] = x.attrs.get("normalized_shape").map(_.ints.toList).get
          val eps: Double = x.attrs.get("eps").flatMap(_.f).map(_.toDouble).get
          Operation.ATenLayerNorm(input, weight, bias, output, shape, eps)
        }
      },
      OpBuilder("embedding_bag") { (x: OpData) =>
        Try {
          val weight = x.inputs(0)
          val input = x.inputs(1)
          val offsets = x.inputs(2)
          val output = x.outputs(0)
          val mode = x.attrs.get("mode").flatMap(_.i).flatMap(EmbeddingBag.Mode.fromEnum(_)).get
          Operation.ATenEmbeddingBag(weight, input, offsets, output, mode)
        }
      }
    )

  private def binOpBuilder(
      name: String,
      op: (Register, Register, Register) => Operation.BinOp
  ): OpBuilder =
    OpBuilder(name) { (x: OpData) =>
      Try {
        op(x.inputs(0), x.inputs(1), x.outputs.head)
      }
    }

  val builders: List[OpBuilder] =
    List(
      binOpBuilder("Add", Operation.BinOp.Add(_, _, _)),
      OpBuilder("Cast") { (x: OpData) =>
        val o =
          for {
            in <- x.inputs.headOption
            out <- x.outputs.headOption
          } yield (in, out)

        for {
          (i, o) <- toTry(x.context, o)
          dt <- dataTypeFromInt(x.attrs.get("to").flatMap(_.i).map(_.toInt))
        } yield Operation.CastOp(i, dt, o)
      },
      OpBuilder("Constant") { (x: OpData) =>
        val o = for {
          ap <- x.attrs.get("value")
          output <- x.outputs.headOption
          tp <- ap.t
        } yield (tp, output)
        toTry(x.context, o).flatMap {
          case (tp, output) =>
            loadInternalTensor(tensorType(tp).assertInternal)
              .map { case (_, t) => Operation.ConstantOp(t, output) }
        }
      },
      OpBuilder("ConstantOfShape") { (x: OpData) =>
        val o = for {
          input <- x.inputs.headOption
          output <- x.outputs.headOption
        } yield (input, output)
        // value is optional, so we don't want to flatMap it
        val value = x.attrs.get("value").flatMap(_.t)

        toTry(x.context, o).flatMap {
          case (in, out) =>
            value match {
              case None => Success(Operation.ConstantOfShape(in, None, out))
              case Some(tp) =>
                loadInternalTensor(tensorType(tp).assertInternal)
                  .map {
                    case (_, t) =>
                      Operation.ConstantOfShape(in, Some(t), out)
                  }
            }
        }
      },
      OpBuilder("Concat") { (x: OpData) =>
        val o = for {
          at <- x.attrs.get("axis")
          ine <- NonEmptyList.fromList(x.inputs.toList)
          axis <- at.i
          output <- x.outputs.headOption
        } yield (axis, ine, output)
        toTry(x.context, o).map {
          case (axis, ine, output) =>
            Operation.ConcatOp(ine, axis, output)
        }
      },
      binOpBuilder("Div", Operation.BinOp.Div(_, _, _)),
      OpBuilder("Gather") { (x: OpData) =>
        Operation
          .GatherOp(
            data = x.inputs(0),
            indices = x.inputs(1),
            output = x.outputs(0),
            axis = x.attrs.get("axis").flatMap(_.i).getOrElse(0L)
          )
          .success
      },
      OpBuilder("Dropout") { (x: OpData) =>
        Operation
          .DropoutOp(
            input = x.inputs(0),
            output = x.outputs(0),
            maskOutput = x.outputs(1),
            ratio = x.attrs.get("ratio").flatMap(_.f).getOrElse(0.5f)
          )
          .success
      },
      OpBuilder("Exp") { (x: OpData) =>
        Try(Operation.FloatMapOp.Exp(x.inputs(0), x.outputs(0)))
      },
      OpBuilder("LeakyRelu") { (x: OpData) =>
        Try(
          Operation.FloatMapOp.LeakyRelu(
            x.attrs
              .get("alpha")
              .flatMap(_.floats.headOption)
              .getOrElse(0.01f)
              .toDouble,
            x.inputs(0),
            x.outputs(0)
          )
        )
      },
      OpBuilder("Log") { (x: OpData) =>
        Try(Operation.FloatMapOp.Log(x.inputs(0), x.outputs(0)))
      },
      OpBuilder("MaxPool") { (x: OpData) =>
        import x.attrs
        Try {
          Operation.MaxPoolOp(
            attrs
              .get("auto_pad")
              .flatMap(_.s)
              .flatMap { s =>
                MaxPool.AutoPad.fromString(s.toStringUtf8)
              }
              .getOrElse(MaxPool.AutoPad.default),
            attrs.get("ceil_mode").flatMap(_.i).fold(false)(_ != 0L),
            attrs.get("dilations").map(_.ints.toList),
            attrs("kernel_shape").ints.toList,
            attrs.get("pads").map(_.ints.toList),
            attrs.get("storage_order").flatMap(_.i).fold(MaxPool.StorageOrder.default) {
              case 0L => MaxPool.StorageOrder.RowMajor
              case 1L => MaxPool.StorageOrder.ColMajor
              case s  => sys.error(s"unknown storage order: $s in $x")
            },
            attrs("strides").ints.toList,
            x.inputs.head,
            x.outputs.head,
            x.outputs.lift(1)
          )
        }
      },
      binOpBuilder("Mul", Operation.BinOp.Mul(_, _, _)),
      OpBuilder("Identity") { (x: OpData) =>
        Try(Operation.IdentityOp(x.inputs.head, x.outputs.head))
      },
      OpBuilder("ReduceSum") { (x: OpData) =>
        Operation
          .ReduceSumOp(
            axes = x.attrs.get("axes").map(_.ints.toList),
            keepDims = x.attrs.get("keepdims").flatMap(_.i).map(_ != 0).getOrElse(true),
            input = x.inputs(0),
            output = x.outputs(0)
          )
          .success
      },
      OpBuilder("Slice") { (x: OpData) =>
        Try(
          Operation.SliceOp(
            data = x.inputs(0),
            starts = x.inputs(1),
            ends = x.inputs(2),
            axes = x.inputs.lift(3),
            steps = x.inputs.lift(4),
            output = x.outputs.head
          )
        )
      },
      binOpBuilder("Sub", Operation.BinOp.Sub(_, _, _)),
      OpBuilder("Squeeze") { (x: OpData) =>
        Operation
          .SqueezeOp(
            input = x.inputs(0),
            axes = x.attrs.get("axes").map(_.ints.toList).getOrElse(Nil),
            output = x.outputs(0)
          )
          .success
      },
      OpBuilder("Transpose") { (x: OpData) =>
        Try(
          Operation
            .TransposeOp(x.attrs.get("perm").map(_.ints.toList), x.inputs.head, x.outputs.head)
        )
      },
      OpBuilder("Unsqueeze") { (x: OpData) =>
        Operation
          .UnsqueezeOp(
            input = x.inputs(0),
            axes = x.attrs.get("axes").map(_.ints.toList).getOrElse(Nil),
            output = x.outputs(0)
          )
          .success
      },
      OpBuilder("BatchNormalization") { (x: OpData) =>
        Operation
          .BatchNormalizationOp(
            data = x.inputs(0),
            scale = x.inputs(1),
            bias = x.inputs(2),
            mean = x.inputs(3),
            variance = x.inputs(4),
            epsilon = x.attrs.get("epsilon").flatMap(_.f).getOrElse(1e-5f),
            momentum = x.attrs.get("momentum").flatMap(_.f).getOrElse(0.9f),
            output = x.outputs(0)
          )
          .success
      },
      OpBuilder("Gemm") { (x: OpData) =>
        Operation
          .GemmOp(
            a = x.inputs(0),
            b = x.inputs(1),
            c = x.inputs(2),
            alpha = x.attrs.get("alpha").flatMap(_.f).getOrElse(1f),
            beta = x.attrs.get("beta").flatMap(_.f).getOrElse(1f),
            transA = x.attrs.get("transA").flatMap(_.i).getOrElse(0L) != 0L,
            transB = x.attrs.get("transB").flatMap(_.i).getOrElse(0L) != 0L,
            output = x.outputs(0)
          )
          .success
      },
      OpBuilder("Relu") { (x: OpData) =>
        Try(Operation.FloatMapOp.Relu(x.inputs(0), x.outputs(0)))
      },
      OpBuilder("Reshape") { (x: OpData) =>
        Try(Operation.ReshapeOp(x.inputs(0), x.inputs(1), x.outputs(0)))
      },
      OpBuilder("Sigmoid") { (x: OpData) =>
        Try(Operation.FloatMapOp.Sigmoid(x.inputs(0), x.outputs(0)))
      },
      OpBuilder("Shape") { (x: OpData) =>
        Try(Operation.ShapeOp(x.inputs(0), x.outputs(0)))
      },
      OpBuilder("Softmax") { (x: OpData) =>
        Operation
          .SoftmaxOp(
            input = x.inputs(0),
            axis = x.attrs.get("axis").flatMap(_.i.map(_.toInt)).getOrElse(1),
            output = x.outputs(0)
          )
          .success
      },
      OpBuilder("ATen") { (x: OpData) =>
        x.attrs.get("operator").flatMap(_.s) match {
          case Some(name) => {
            val utf8Name = name.toStringUtf8()
            atenOps.get(utf8Name) match {
              case Some(bldr) => bldr.build(x)
              case None       => Failure(new Exception(s"unsupported ATen operator: $utf8Name"))
            }
          }
          case None =>
            Failure(new Exception("ATen requires an operator (none found)"))
        }
      },
      OpBuilder("NonZero") { (x: OpData) =>
        Try(Operation.NonZeroOp(x.inputs(0), x.outputs(0)))
      }
    )

  val supportedOps: Map[String, OpBuilder] =
    builders.map(b => (b.name, b)).toMap

  val atenOps: Map[String, OpBuilder] =
    atenBuilders.map(b => (b.name, b)).toMap

  def parseOp(node: NodeProto): Try[Operation] = {
    val inputs: Seq[Register] = node.input.map(Register(_))
    val outputs: Seq[Register] = node.output.map(Register(_))
    val attrs: Map[String, AttributeProto] = node.attribute.map(a => (a.name.get, a)).toMap
    val x = OpData(node, inputs, outputs, attrs)
    node.opType match {
      case None =>
        Failure(new Exception(s"missing op type: $node"))
      case Some(o) =>
        supportedOps.get(o) match {
          case Some(ob) => ob.build(x)
          case None     => Failure(new Exception(s"unknown op type: $o ($node)"))
        }
    }
  }

  def parseOps(graph: GraphProto): Try[List[Operation]] =
    graph.node.toList.traverse(parseOp(_))

  def parseShape(tsp: TensorShapeProto): Axes = {
    import TensorShapeProto.Dimension
    import Dimension.Value._
    val ns = tsp.dim.map(_.value).collect { case DimValue(n) => n }
    Shape.axes(ns: _*)
  }

  def parseValue(v: ValueInfoProto): Try[(Register, (DataType, Axes))] = {
    import TypeProto.Value.TensorType
    import TypeProto.Tensor
    v.`type`.map(_.value) match {
      case Some(TensorType(Tensor(Some(tpe), shape))) =>
        dataTypeFromInt(tpe).map { dt =>
          val s = parseShape(shape.get)
          (Register(v.name.get), (dt, s))
        }
      case o =>
        Failure(new Exception(s"unknown type: $o"))
    }
  }

  def parseModelInputs(
      graph: GraphProto
  ): Try[(Vector[Register], Map[Register, (DataType, Axes)])] =
    graph.input.toList.traverse(parseValue(_)).map { pairs =>
      val m = pairs.toMap
      require(m.size == pairs.size, s"duplicate registers detected in $pairs")
      (m.iterator.map(_._1).toVector, m)
    }

  def parseModelOutput(graph: GraphProto): Try[(Register, (DataType, Axes))] =
    graph.output.headOption match {
      case Some(out) =>
        //FIXME, there could be many outputs
        parseValue(out)
      case None =>
        Failure(new Exception(s"found no output in graph.name=${graph.name}"))
    }

  private val ExternalBytes = "__EXTERNAL".getBytes

  def parseDims(ns: Seq[Long]): Axes =
    Shape.axes(ns: _*)

  def parseInitializers(
      graph: GraphProto
  )(tensorPath: String => IO[Path]): Resource[IO, Registers] =
    graph.initializer.toList.foldM(Registers.empty) { (rs: Registers, t: TensorProto) =>
      for {
        (register, wnt) <- loadTensor(t)(tensorPath)
        reg <- ApplicativeError[Resource[IO, ?], Throwable].fromTry(rs.create(register, wnt))
      } yield reg
    }

  def tensorType(tp: TensorProto): StorageMode =
    dataTypeFromInt(tp.dataType) match {
      case Failure(e) =>
        StorageMode.Error(s"invalid data type for ${tp.name}: ${tp.dataType} ($e)")
      case Success(dt) =>
        val axes: Axes = parseDims(tp.dims)
        tp.rawData match {
          case None =>
            StorageMode.InternalTyped(dt, axes, tp.name, tp)
          case Some(bs) =>
            val bytes = bs.toByteArray
            if (Arrays.equals(bytes, ExternalBytes)) {
              tp.name match {
                case Some(name) => StorageMode.External(dt, axes, name)
                case None       => StorageMode.Error(s"No name for external tensor $tp")
              }
            } else {
              StorageMode.InternalRaw(dt, axes, tp.name, bytes)
            }
        }
    }

  def loadTensor(
      t: TensorProto
  )(tensorPath: String => IO[Path]): Resource[IO, (Register, Tensor.Unknown)] =
    tensorType(t) match {
      case int: StorageMode.Internal =>
        loadInternalTensor(int) match {
          case Success((Some(register), tensor)) => Resource.pure((register, tensor))
          case Success((None, tensor)) =>
            ApplicativeError[Resource[IO, ?], Throwable]
              .raiseError(new IllegalArgumentException("Cannot load InternalTensor without name"))
          case Failure(exc) => ApplicativeError[Resource[IO, ?], Throwable].raiseError(exc)
        }
      case StorageMode.External(dt, axes, name) =>
        loadExternalTensor(dt, axes, name)(tensorPath)
    }

  def loadInternalTensor(tt: StorageMode.Internal): Try[(Option[Register], Tensor.Unknown)] =
    tt match {
      case StorageMode.Error(msg) =>
        Failure(new Exception(msg))
      case StorageMode.InternalRaw(dt, axes, name, bytes) =>
        Tensor.loadBytes(bytes, dt, axes).map(t => (name.map(Register(_)), t))
      case StorageMode.InternalTyped(dt, axes, name, tp) =>
        Try((name.map(Register(_)), Tensor(dt, axes.asRowMajorDims)(toStorage(dt, tp))))
    }

  def loadExternalTensor(dt: DataType, axes: Axes, name: String)(
      tensorPath: String => IO[Path]
  ): Resource[IO, (Register, Tensor.Unknown)] =
    Resource
      .liftF(tensorPath(name))
      .flatMap(Tensor.load(_, dt, axes))
      .map(t => (Register(name), t))

  def toStorage(dt: DataType, tp: TensorProto): Storage[dt.Elem] = {
    implicit val ct = dt.classTag
    val alloc = StorageAllocator.forDataType(dt)

    def conv[A](src: DataType)(seq: Seq[A])(f: A => src.Elem): Storage[dt.Elem] =
      DataType.maybeElemIs(src, dt) match {
        case Some(isa) =>
          alloc.toArrayStorage(isa.substitute[Seq](seq.map(f)).toArray, 0)
        case None => sys.error(s"unreachable: $src was not $dt")
      }

    dt match {
      case d: DataType.Uint8.type    => conv(d)(tp.int32Data)(_.toByte)
      case d: DataType.Uint16.type   => conv(d)(tp.int32Data)(_.toShort)
      case d: DataType.Int8.type     => conv(d)(tp.int32Data)(_.toByte)
      case d: DataType.Int16.type    => conv(d)(tp.int32Data)(_.toShort)
      case d: DataType.Int32.type    => conv(d)(tp.int32Data)(x => x)
      case d: DataType.Int64.type    => conv(d)(tp.int64Data)(x => x)
      case d: DataType.BFloat16.type => conv(d)(tp.int32Data)(x => (x & 0xffff).toShort)
      case d: DataType.Float16.type  => conv(d)(tp.int32Data)(x => (x & 0xffff).toShort)
      case d: DataType.Float32.type  => conv(d)(tp.floatData)(x => x)
      case d: DataType.Float64.type  => conv(d)(tp.doubleData)(x => x)
      case d: DataType.String.type   => conv(d)(tp.stringData)(x => x)
    }
  }

  def nameToPath(modelPath: Path, name: String): Path =
    Option(modelPath.getParent) match {
      case Some(b) => b.resolve(name)
      case None    => Paths.get(name)
    }

  def loadFromPaths(protoPath: Path)(tensorPath: String => IO[Path]): Resource[IO, Model] =
    for {
      mp <- Resource.liftF(Agate.loadModel(protoPath))
      graph = mp.graph.getOrElse(sys.error(s"no graph found in $mp"))
      regs <- parseInitializers(graph)(tensorPath)
      (irs, inputs) <- ApplicativeError[Resource[IO, ?], Throwable].fromTry(parseModelInputs(graph))
      output <- ApplicativeError[Resource[IO, ?], Throwable].fromTry(parseModelOutput(graph))
      ops <- ApplicativeError[Resource[IO, ?], Throwable].fromTry(parseOps(graph))
    } yield {
      Model(regs, irs, inputs, ops, output)
    }

  /**
   *
   */
  def load(path: Path): Resource[IO, Model] =
    loadFromPaths(path) { n =>
      IO.delay(nameToPath(path, n))
    }

  sealed abstract class StorageMode {
    def assertInternal: StorageMode.Internal =
      this match {
        case ext @ StorageMode.External(_, _, _) =>
          StorageMode.Error(s"storage mode $ext is not internal")
        case int: StorageMode.Internal =>
          int
      }
  }

  object StorageMode {
    sealed abstract class Internal extends StorageMode

    case class External(dt: DataType, axes: Axes, name: String) extends StorageMode
    case class Error(msg: String) extends Internal
    case class InternalRaw(dt: DataType, axes: Axes, name: Option[String], bytes: Array[Byte])
        extends Internal
    case class InternalTyped(dt: DataType, axes: Axes, name: Option[String], tp: TensorProto)
        extends Internal
  }
}

case class Model(
    staticRegisters: Registers,
    inputRegisters: Vector[Register],
    inputs: Map[Register, (DataType, Axes)],
    ops: List[Operation],
    output: (Register, (DataType, Axes))
) {
  def nonStaticRegisters: Vector[Register] =
    inputRegisters.filterNot(staticRegisters.registers.keySet)

  /**
   * Apply the inputs and validate that the types match the
   * declared inputs
   */
  def validateInputs(registers: Registers): Try[Registers] =
    for {
      combined <- staticRegisters.merge(registers)
      _ <- combined.validateTypes(inputs)
    } yield combined

  /**
   * validateInputs then runAll operations
   */
  def strictRun(registers: Registers): Try[Registers] =
    for {
      combined <- validateInputs(registers)
      res <- Operation.runAll(ops)(combined)
    } yield res

  /**
   * overwrite any static registers then coerce all
   * the types to those declared in inputs
   */
  def coerceInputs(registers: Registers): Try[Registers] = {
    val combined = staticRegisters.replaceAll(registers)
    combined.coerceTypes(inputs)
  }

  /**
   * This is a lax version of strictRun that overwrites
   * staticRegisters as needed and coerces types
   */
  def run(registers: Registers): Try[Registers] =
    coerceInputs(registers).flatMap { registers =>
      Operation.runAll(ops)(registers)
    }

  def extractResult(regs: Registers): Try[Tensor.Unknown] =
    regs.get(output._1).flatMap { t0 =>
      if (t0.axes == output._2._2) Success(t0)
      else Failure(new Exception(s"wrong output axes: ${t0.axes} vs ${output._2}"))
    }

  def runAndExtract(registers: Registers): Try[Tensor.Unknown] =
    for {
      results <- run(registers)
      t <- extractResult(results)
    } yield t
}
