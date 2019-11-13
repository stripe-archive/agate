package com.stripe.agate.eval

import com.stripe.agate.GraphViz

import cats.data.{State, Validated}
import cats.effect.{ExitCode, IO}
import com.monovore.decline.{Argument, Command, Opts}
import com.google.protobuf.ByteString
import java.nio.file.Path
import onnx.onnx.{AttributeProto, GraphProto, NodeProto}
import org.typelevel.paiges.Doc

import AttributeProto.AttributeType
import Dump.OutputMode
import cats.implicits._

case class Dump(path: Path, output: OutputMode) extends Agate.Mode {

  def decodeAttribute(a: AttributeProto): (String, AttributeType, String) = {
    import AttributeType._
    val name = a.name.getOrElse("<anonymous>")
    val typ: AttributeType = a.`type`.getOrElse(Unrecognized(-1))
    val value = typ match {
      case UNDEFINED => "n/a"
      case FLOAT     => a.f.get
      case INT       => a.i.get
      case STRING =>
        a.s.get match {
          case bs: ByteString => bs.toStringUtf8()
          case s              => s
        }
      case TENSOR          => a.t.get
      case GRAPH           => a.g.get
      case FLOATS          => a.floats
      case INTS            => a.ints
      case STRINGS         => a.strings
      case TENSORS         => a.tensors
      case GRAPHS          => a.graphs
      case Unrecognized(_) => "n/a"
    }
    (name, typ, value.toString)
  }

  def dumpNode(node: NodeProto): Doc = {

    val commaSep = Doc.comma + Doc.lineOrSpace
    val equalsSep = Doc.space + Doc.char('=') + Doc.lineOrSpace

    def parenthesized(ds: Iterable[Doc]): Doc =
      Doc.char('(') + Doc.intercalate(commaSep, ds) + Doc.char(')')

    val attrs = parenthesized(node.attribute.map { a =>
      val (name, typ, value) = decodeAttribute(a)
      Doc.text(name) + Doc.text(": ") + Doc.str(typ) + Doc.text(" = ") + Doc.text(value)
    })

    val inputs = parenthesized(node.input.map(Doc.text(_)))
    val outputs = parenthesized(node.output.map(Doc.text(_)))
    val o = Doc.text(node.opType.getOrElse("<unknown>"))
    (Doc.text("Node(input") + equalsSep + inputs + commaSep +
      Doc.text("output") + equalsSep + outputs + commaSep +
      Doc.text("opType") + equalsSep + o + commaSep +
      Doc.text("attrs") + equalsSep + attrs + Doc.char(')')).nested(4)
  }

  def dumpGraph(graph: GraphProto): Doc =
    output match {
      case OutputMode.Text =>
        Doc.intercalate(Doc.line, graph.node.map(dumpNode(_)))
      case OutputMode.GraphViz =>
        import GraphViz.{Attribute, Element}

        val allIOs =
          graph.node.toList
            .flatMap { n =>
              n.input.iterator ++ n.output.iterator
            }
            .distinct
            .traverse { n =>
              GraphViz.checkNode(n).map(_.withAttr("label", Attribute.Quoted(n)))
            }

        /*
         * Compute all the edges
         */
        val elems: State[GraphViz.NameState, List[Element]] =
          graph.node.zipWithIndex.toList.flatTraverse {
            case (n, idx) =>
              val nm = n.name.getOrElse(s"<unknown $idx>")
              val node = GraphViz
                .checkNode(nm)
                .map { gn =>
                  gn.withAttrs(Attribute.Shape.Box)
                    .withAttr("label", Attribute.Quoted(n.opType.getOrElse("<unknown op>")))
                }

              // the outputs depend on this node
              val outs = n.output.toList.traverse { o =>
                GraphViz.checkEdge(o, nm)
              }

              val ins = n.input.toList.traverse { i =>
                GraphViz.checkEdge(nm, i)
              }

              for {
                nodeElem <- node
                inEs <- ins
                outEs <- outs
              } yield nodeElem :: inEs ::: outEs
          }

        GraphViz.finalizeNames {
          for {
            ios <- allIOs
            es <- elems
          } yield GraphViz.digraph(ios ::: es).toDoc
        }
    }

  def graph: IO[GraphProto] =
    Agate.loadModel(path).map(_.graph.get)

  def run: IO[ExitCode] =
    graph.attempt.flatMap {
      case Left(err) =>
        IO {
          System.err.println(err)
          ExitCode.Error
        }
      case Right(model) =>
        val doc = dumpGraph(model)
        IO {
          println(doc.render(80))
          ExitCode.Success
        }
    }
}

object Dump {
  sealed abstract class OutputMode
  object OutputMode {
    case object Text extends OutputMode
    case object GraphViz extends OutputMode

    implicit val argOutputMode: Argument[OutputMode] =
      new Argument[OutputMode] {
        val defaultMetavar = "output-mode (text or graphviz)"
        def read(string: String) =
          string match {
            case "text"     => Validated.valid(Text)
            case "graphviz" => Validated.valid(GraphViz)
            case unknown    => Validated.invalidNel(s"unknown output type: $unknown")
          }
      }
  }

  val command: Command[Dump] = {
    val path = Opts.argument[Path]("path-to-proto")
    val om = Opts
      .option[OutputMode]("output-mode", "the mode to print output as. text or graphviz", "o")
      .withDefault(OutputMode.Text)

    val opts = (path, om).mapN(Dump(_, _))
    Command("dump", "print the structure of the compute graph")(opts)
  }
}
