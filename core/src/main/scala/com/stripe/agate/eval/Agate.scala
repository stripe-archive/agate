package com.stripe.agate.eval

import cats.Monoid
import cats.data.NonEmptyList
import cats.effect.{ExitCode, IO, Resource}
import com.monovore.decline.{Command, Opts}
import com.google.protobuf.ByteString
import java.io.FileInputStream
import java.nio.file.{Files, Path}
import onnx.onnx.{GraphProto, ModelProto, TensorProto, TensorShapeProto, TypeProto, ValueInfoProto}
import org.typelevel.paiges.Doc

import cats.implicits._

object Agate {
  def loadModel(path: Path): IO[ModelProto] =
    Resource
      .make(IO(new FileInputStream(path.toFile)))(fis => IO(fis.close()))
      .use(fis => IO(ModelProto.parseFrom(fis)))

  trait Mode {
    def run: IO[ExitCode]
  }

  case class Show(paths: NonEmptyList[Path]) extends Mode {
    def showShape(tsp: TensorShapeProto): Doc = {
      import TensorShapeProto.Dimension
      import Dimension.Value._

      def showDim(d: Dimension): Doc =
        d.value match {
          case Empty       => Doc.text("<empty>")
          case DimValue(l) => Doc.str(l)
          case DimParam(v) => Doc.text(v)
        }

      Doc.intercalate(Doc.text(" x "), tsp.dim.map(showDim))
    }

    def showElemType(tpe: Int): Doc =
      Doc.text(tpe match {
        case 0 => "UNDEFINED"
        case 1 => "FLOAT"
        case 2 => "UINT8"
        case 3 => "INT8"
        case 4 => "UINT16"
        case 5 => "INT16"
        case 6 => "INT32"
        case 7 => "INT64"
        case 8 => "STRING"
        case 9 => "BOOL"
        // IEEE754 half-precision floating-point format (16 bits wide).
        // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
        case 10 => "FLOAT16"

        case 11 => "DOUBLE"
        case 12 => "UINT32"
        case 13 => "UINT64"
        case 14 => "COMPLEX64"
        case 15 => "COMPLEX128"
        // Non-IEEE floating-point format based on IEEE754 single-precision
        // floating-point number truncated to 16 bits.
        // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
        case 16 => "BFLOAT16"
        case uk => s"_$uk"
      })

    def showRaw(bytes: ByteString): Doc = {
      def showSize = Doc.text(s"<${bytes.size} literal bytes>")
      if (bytes.size < 30) {
        var ascii = true
        val it = bytes.iterator
        while (ascii && it.hasNext) {
          val b = it.next
          ascii = 32 <= b && b < 127
        }
        if (ascii) {
          Doc.text(bytes.toStringUtf8)
        } else showSize
      } else showSize
    }

    def showTensor(t: TensorProto): Doc = {
      val sep = Doc.comma + Doc.space
      val col = Doc.char(':') + Doc.space
      val col2 = Doc.text("::") + Doc.space
      val eql = Doc.text(" = ")
      val tensorDoc = t.rawData match {
        case Some(rd) => eql + showRaw(rd)
        case None     => Doc.empty
      }
      Doc.text("Tensor(") + Doc.text(t.name.getOrElse("<anonymous>")) + col +
        t.dataType.map(showElemType(_)).getOrElse(Doc.text("<missing>")) + col2 +
        Doc.intercalate(sep, t.dims.map(Doc.str(_))) +
        tensorDoc + Doc.char(')')
    }

    def showType(tpe: TypeProto.Value): Doc = {
      import TypeProto.Value._
      import TypeProto.Tensor

      tpe match {
        case TensorType(Tensor(elemtpe, shape)) =>
          elemtpe.fold(Doc.text("?type"))(showElemType) +
            Doc.text(" :: ") +
            shape.fold(Doc.text("?shape"))(showShape)
        case other => Doc.str(other)
      }
    }

    def showValue(value: ValueInfoProto): Doc = {
      val nd = Doc.text(value.name.getOrElse("?name"))
      val td = value.`type` match {
        case None    => Doc.text("?type")
        case Some(v) => showType(v.value)
      }
      nd + Doc.text(": ") + td
    }

    case class Summary(opCount: Map[String, Int]) {
      def combine(that: Summary): Summary =
        Summary(Monoid[Map[String, Int]].combine(opCount, that.opCount))

      def toDoc: Doc = {
        val sz = opCount.size
        val tot = opCount.iterator.map(_._2).sum
        val supportedKeys = Model.supportedOps.keySet
        val unsupportedCount = opCount.keys.filterNot(supportedKeys).size
        val unsupportedWeight = opCount.iterator.collect { case (k, v) if !supportedKeys(k) => v }.sum
        val unsupDistPer = "%2.1f".format((unsupportedCount.toDouble / sz.toDouble) * 100.0)
        val unsupTotPer = "%2.1f".format((unsupportedWeight.toDouble / tot.toDouble) * 100.0)
        val summary =
          Doc.text(
            s"($sz distinct ops, $tot total ops, $unsupportedCount unsupported(!) distinct ops (${unsupDistPer}%), $unsupportedWeight unsupported total (${unsupTotPer}%)):"
          )
        val pairs: List[Doc] = opCount.toList.sortBy(-_._2).map {
          case (n, c) =>
            val unsup = if (Model.supportedOps.contains(n)) Doc.empty else Doc.char('!')
            (Doc.text(n) + unsup) & Doc.text(s"($c)")
        }
        summary + (Doc.line + Doc.text("  ") +
          Doc.intercalate(Doc.comma + Doc.lineOrSpace, pairs).nested(4))
      }
    }

    sealed trait Result {
      def path: Path
      def toDoc: Doc
      def code: ExitCode
    }
    case class Found(path: Path, model: ModelProto, graph: GraphProto) extends Result {
      val summary: Summary = Summary(opsFromGraph(graph))
      def code = ExitCode.Success
      def toDoc: Doc = {
        val header = Doc.text(path.toString) + Doc.char(':')
        val iz = graph.input.size
        val oz = graph.output.size
        val sep = Doc.line
        val table = Doc.tabulate(
          '.',
          "...",
          List(
            "nodes" -> summary.toDoc,
            "initializers" -> Doc.intercalate(sep, graph.initializer.map(showTensor)),
            s"inputs ($iz)" -> Doc.intercalate(sep, graph.input.map(showValue)),
            s"outputs ($oz)" -> Doc.intercalate(sep, graph.output.map(showValue))
          )
        )

        header + (Doc.line + table).nested(2)
      }
    }
    case class ParseFailure(path: Path, err: Throwable) extends Result {
      def code = ExitCode.Error
      def toDoc =
        Doc
          .text(path.toString)
          .space(
            Doc
              .text("failed with message:")
              .lineOrSpace(Doc.text(s"${err.getClass.getSimpleName}: ${err.getMessage}"))
              .nested(4)
          )
    }
    case class MissingGraph(path: Path, model: ModelProto) extends Result {
      def code = ExitCode.Error
      def toDoc = Doc.text(path.toString) :& "had no graph defined"
    }

    def run: IO[ExitCode] = {
      val read: IO[NonEmptyList[Result]] =
        paths.traverse { path =>
          loadModel(path).attempt.map {
            case Left(err) => ParseFailure(path, err)
            case Right(model) =>
              model.graph match {
                case None    => MissingGraph(path, model)
                case Some(g) => Found(path, model, g)
              }
          }
        }

      /*
       * return if there are 2 or more Found
       */
      def getTotal(ne: NonEmptyList[Result]): Option[Summary] = {
        val founds = ne.toList.collect { case f @ Found(_, _, _) => f }

        founds match {
          case many @ (_ :: _ :: _) =>
            Some(many.iterator.map(_.summary).reduce(_.combine(_)))
          case _ => None
        }
      }

      def processResults(nel: NonEmptyList[Result]): NonEmptyList[(Doc, ExitCode)] =
        nel.sortBy(_.path.toString).map { r =>
          (r.toDoc, r.code)
        }

      def show(items: NonEmptyList[Doc]): IO[Unit] = {
        val ilist = items.toList
        val res = Doc.intercalate(Doc.line + Doc.line, ilist)
        val resStr = res.render(80)
        IO(println(resStr))
      }

      def finalCode(cs: NonEmptyList[ExitCode]): ExitCode =
        cs.toList.reduce { (c1, c2) =>
          if (c1.code == 0) c2 else c1
        }

      for {
        rs <- read
        sum = getTotal(rs)
        ps = processResults(rs)
        docs = ps.map(_._1)
        docs1 = sum match {
          case None    => docs
          case Some(d) => docs :+ (Doc.text("TOTAL: ") + d.toDoc)
        }
        _ <- show(docs1)
        code = finalCode(ps.map(_._2))
      } yield code
    }

    def opsFromGraph(graph: GraphProto): Map[String, Int] =
      graph.node
        .flatMap(_.opType.toList)
        .groupBy(identity)
        .iterator
        .map { case (k, vs) => (k, vs.size) }
        .toMap
  }

  val show: Command[Show] = {
    val paths = Opts.arguments[Path]("path-to-proto")
    val opts = paths.map(Show(_))
    Command("show", "tool to list operations used in models")(opts)
  }

  val tool: Command[Mode] =
    Command("agate", "a tool to work with onnx model files")(
      Opts
        .subcommand(show)
        .orElse(Opts.subcommand(Dump.command))
        .orElse(Opts.subcommand(Score.command))
    )

  def main(args: Array[String]): Unit =
    tool.parse(args.toVector) match {
      case Left(help) =>
        println(help.toString)
        System.exit(ExitCode.Error.code)
      case Right(mode) =>
        System.exit(mode.run.unsafeRunSync.code)
    }
}
