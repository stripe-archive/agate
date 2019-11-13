package com.stripe.agate

import cats.data.State
import cats.data.NonEmptyList
import org.typelevel.paiges.Doc

object GraphViz {
  sealed abstract class GraphType(val asString: String)
  object GraphType {
    case object Graph extends GraphType("graph")
    case object Digraph extends GraphType("digraph")
  }

  sealed trait Attribute {
    def toDoc: Doc
  }
  object Attribute {
    case class Quoted(str: String) extends Attribute {
      def toDoc =
        Doc.text("\"" + str.flatMap {
          case '"' => "\\\""
          case c   => c.toString
        } + "\"")
    }
    case class Attr(str: String) extends Attribute {
      def toDoc = Doc.text(str)
    }

    object Shape {
      val key: String = "shape"

      val Box: Map[String, Attribute] =
        Map(key -> Attribute.Attr("box"))
    }

    def mapToDoc(m: Map[String, Attribute]): Option[Doc] =
      if (m.isEmpty) None
      else
        Some {
          Doc.char('[') +
            Doc.intercalate(Doc.text(", "), m.toList.sortBy(_._1).map {
              case (k, attr) =>
                Doc.text(k) + Doc.char('=') + attr.toDoc
            }) +
            Doc.char(']')
        }
  }

  sealed trait Element {
    def toDoc: Doc
  }

  sealed trait AttrElement[E <: Element] extends Element {
    def withAttr(k: String, at: Attribute): E
    def withAttrs(attrs: Map[String, Attribute]): E
  }

  object Element {
    case class Node(name: String, attr: Map[String, Attribute]) extends AttrElement[Node] {
      def withAttr(k: String, at: Attribute): Node =
        copy(attr = attr.updated(k, at))

      def withAttrs(attrs: Map[String, Attribute]): Node =
        copy(attr = attr ++ attrs)

      def toDoc = {
        val nd = Doc.text(name)
        val ad = Attribute.mapToDoc(attr) match {
          case None    => Doc.empty
          case Some(d) => Doc.space + d
        }
        nd + ad + Doc.char(';')
      }
    }

    object Node {
      def apply(name: String): Node = Node(name, Map.empty)
    }

    private[this] val arrow = Doc.text(" -> ")

    case class Edge(name: String, dests: NonEmptyList[String], attr: Map[String, Attribute])
        extends AttrElement[Edge] {
      def withAttr(k: String, at: Attribute): Edge =
        copy(attr = attr.updated(k, at))

      def withAttrs(attrs: Map[String, Attribute]): Edge =
        copy(attr = attr ++ attrs)

      def toDoc = {
        val nd = Doc.text(name)
        val targetD = dests match {
          case NonEmptyList(n2, Nil) => Doc.text(n2)
          case twoOrMore =>
            val items = Doc.intercalate(Doc.text("; "), twoOrMore.toList.map(Doc.text(_)))
            Doc.char('{') + items + Doc.char('}')
        }

        val ad = Attribute.mapToDoc(attr) match {
          case None    => Doc.empty
          case Some(d) => Doc.space + d
        }
        nd + arrow + targetD + ad + Doc.char(';')
      }
    }

    object Edge {
      def apply(from: String, to: String): Edge =
        Edge(from, NonEmptyList(to, Nil), Map.empty)
    }
  }

  case class Graph(gtype: GraphType, elems: List[Element]) {
    def toDoc: Doc = {
      def block(lines: Iterable[Doc]): Doc =
        Doc.text(gtype.asString) space Doc.char('{') +
          (Doc.line + Doc.intercalate(Doc.line, lines)).nested(2) +
          Doc.line +
          Doc.char('}')

      block(elems.map(_.toDoc))
    }
  }

  def graph(elems: List[Element]): Graph =
    Graph(GraphType.Graph, elems)

  def digraph(elems: List[Element]): Graph =
    Graph(GraphType.Digraph, elems)

  sealed trait NameState

  private case class ST(id: Map[String, String], next: Long) extends NameState

  private val SimpleName = "^([_a-zA-Z0-9])+$".r

  def checkName(name: String): State[NameState, String] =
    name match {
      case SimpleName(_) => State.pure(name)
      case _ =>
        State { ns: NameState =>
          val ST(ids, next) = ns
          ids.get(name) match {
            case None =>
              val safe = s"__anon__$next"
              (ST(ids.updated(name, safe), next + 1L), safe)
            case Some(safe) =>
              (ns, safe)
          }
        }
    }

  def checkNode(n: String): State[NameState, Element.Node] =
    checkName(n).map(Element.Node(_))

  def checkEdge(from: String, to: String): State[NameState, Element.Edge] =
    for {
      fn <- checkName(from)
      tn <- checkName(to)
    } yield Element.Edge(fn, tn)

  def finalizeNames[A](st: State[NameState, A]): A =
    st.runA(ST(Map.empty, 0L)).value
}
