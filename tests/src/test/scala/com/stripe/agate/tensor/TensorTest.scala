package com.stripe.agate
package tensor

import cats.effect.IO
import java.nio.file.Files
import org.scalacheck.{Gen, Prop, Properties}
import org.typelevel.claimant.Claim
import shapeless.HNil
import scala.util.{Success, Try}

import com.stripe.agate.laws.Check._
import Prop.{forAllNoShrink => forAll}
import Shape.{Axes, Axis, Coord, Coords, Dim, Dims, Empty, NonEmpty}
import TensorParser.Interpolation
import TestImplicits._

object TensorTest extends Properties("TensorTest") {

  val transposeCase: Gen[(Axes, Long, Long)] =
    for {
      axes <- genAxes
      g = axisFromShape(axes)
      axis1 <- g
      axis2 <- g
    } yield (axes, axis1, axis2)

  property("transpose") = forAll(transposeCase) {
    case (axes0, a1, a2) =>
      val o = for {
        axes1 <- axes0.transpose(a1, a2)
        axes2 <- axes1.transpose(a1, a2)
      } yield axes2
      Claim(o == Some(axes0))
  }

  property("we can print all empty matrices") = forAll { (t: Tensor.U) =>
    val _ = t.toDoc
    Claim(true)
  }

  property("matrix transposition") = forAll { (m: Matrix) =>
    val t0 = m.tensor.slice(0, 0)
    val t1 = m.tensor.transpose(0, 1).get.slice(1, 0)
    val t2 = m.tensor.slice(1, 0)
    val t3 = m.tensor.transpose(0, 1).get.slice(0, 0)
    Claim(t0 == t1 && t2 == t3)
  }

  property("transpose laws are followed for pairwise transpose") = {
    case class LawArgs(t: Tensor.U, axis0: Long, axis1: Long)
    val gen: Gen[LawArgs] = genTensorU.flatMap { ten =>
      val axis: Gen[Long] =
        if (ten.rank == 0) Gen.const(0L) else Gen.choose(0L, ten.rank.toLong - 1L)
      Gen.zip(axis, axis).map { case (a0, a1) => LawArgs(ten, a0, a1) }
    }

    val p1 = forAll(gen) {
      case LawArgs(t, a0, a1) =>
        // this should always succeed
        val t1 = t.transpose(a0, a1).get
        t.axes.coords.foldLeft(Claim(true)) { (c, coord) =>
          val coord1 = coord.transpose(a0, a1).get
          Claim(t(coord) == t1(coord1))
        }
    }

    val p2 = forAll(gen) {
      case LawArgs(t, a0, a1) =>
        val allAxes = (0 until t.rank).toList.map { a =>
          if (a == a0) a1 else if (a == a1) a0 else a
        }
        Claim(t.dims.transpose(a0, a1) == t.dims.transpose(allAxes).toOption)
    }

    p1 && p2
  }

  property("transpose laws are followed for general transpose") = {

    case class LawArgs(t: Tensor.U, axes: List[Long])
    val gen: Gen[LawArgs] = genTensorU.flatMap { ten =>
      val axes: Gen[List[Long]] = genPerm(ten.rank.toLong)
      axes.map { perm =>
        LawArgs(ten, perm)
      }
    }

    forAll(gen) {
      case LawArgs(t, perm) =>
        // this should always succeed
        val t1 = t.transpose(perm).get
        t.axes.coords.foldLeft(Claim(true)) { (c, coord) =>
          val coord1 = coord.transpose(perm).get
          Claim(t(coord) == t1(coord1))
        }
    }
  }

  property("transposeDefault is idempotent") = forAll { ten: Tensor.U =>
    Claim(ten.transposeDefault.transposeDefault == ten)
  }

  property("transpose 2x2") = forAll { (x0: Float, x1: Float, x2: Float, x3: Float) =>
    val data0 = Array(x0, x1, x2, x3)
    val tensor0 = Tensor.matrix(DataType.Float32)(data0, 0, 2, 2)

    val tensor1 = tensor0.transpose(0, 1).get

    val data2 = Array(x0, x2, x1, x3)
    val tensor2 = Tensor.matrix(DataType.Float32)(data2, 0, 2, 2)

    val tensor3 = tensor0.transposeDefault

    Claim(tensor1 == tensor2) &&
    Claim(tensor3 == tensor1)
  }

  property("Tensor builder works") = {
    val t0 = Tensor((((1f, 2f), (3f, 4f)), ((5f, 6f), (7f, 8f))))
    val dims0 = Shape.axes(2, 2, 2).asRowMajorDims
    val dims1 = Shape.dims((2, 4), (2, 2), (2, 1))
    val t1 = Tensor(DataType.Float32)(Array(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f), 0, dims1)
    Claim(t0 == t1 && dims0 == dims1)
  }

  val coordsMatchIndexing: Gen[(Tensor.U, Coords)] =
    for {
      t <- genTensorU
      cs <- coordsFromDims(t.dims)
    } yield (t, cs)

  property("coords match indexing") = forAll(coordsMatchIndexing) {
    case (t, coords) =>
      val i = t.dims.coordsToIndex(coords)
      val x0 = t.storage(i)
      val x1 = t(coords)
      Claim(x0 == x1)
  }

  property("writeInto #1") = forAll(genTensorU, Gen.choose(0, 20)) { (t0, start) =>
    val t1 = t0.asDataTyped
    val dt = t1.dataType
    val n = t1.dims.totalSize
    implicit val alloc = StorageAllocator.forDataType(dt)
    val data = alloc.allocate(n + start)
    t0.writeInto(data, start.toLong)
    val t2 = Tensor(dt, t1.dims.asRowMajorDims)(data.toStorage.slice(start.toLong))
    Claim(t2 == t1)
  }

  property("rowMajorDims(axes).isRowMajor = true") = {
    forAll { (axes: Axes) =>
      val dims = Shape.rowMajorDims(axes)
      Claim(dims.isRowMajor)
    }
  }

  property("rowMajorDims(axes) satisfies Dim(1) invariant") = forAll { (axes: Axes) =>
    def recur(d: Dims): Prop =
      d match {
        case Empty =>
          Claim(true)
        case NonEmpty(1, Dim(_, stride), ds) =>
          Claim(stride == 1L) && recur(ds)
        case NonEmpty(_, _, ds) =>
          recur(ds)
      }
    recur(axes.asRowMajorDims)
  }

  property("rowMajorDims(axes).transpose(0, 1).isRowMajor = false") =
    forAll(Gen.choose(2, 5), Gen.choose(2, 5)) { (x, y) =>
      val axes0 = Shape.axes(x, y).asRowMajorDims
      val axes1 = axes0.transpose(0, 1).get
      Claim(axes0.isRowMajor && !axes1.isRowMajor)
    }

  property("rowsByCols") = forAll { (axes: Axes) =>
    val o = axes.rowsByCols
    val n = axes.rank
    Claim(o.isDefined == (n == 2))
  }

  property("axes.coords.forall(p) = axes.forall(p)") = forAll { (axes: Axes, p: Coords => Boolean) =>
    val lhs = axes.coords.forall(p)
    val rhs = axes.forall(p)
    Claim(lhs == rhs)
  }

  val genDimTuples: Gen[List[(Long, Long)]] =
    Gen.listOf(Gen.zip(Gen.choose(1L, 10L), Gen.choose(1L, 10L)))

  def verify(dims: Dims): Boolean =
    dims match {
      case Empty                              => true
      case NonEmpty(1, Dim(_, n), _) if n > 1 => false
      case NonEmpty(_, _, rest)               => verify(rest)
    }

  property("length-1 dims have stride 1") = forAll(genDimTuples) { (pairs: List[(Long, Long)]) =>
    val dims = Shape.dims(pairs: _*)
    Claim(verify(dims))
  }

  property("tensor != string") = {
    val t = tensor"[[1 2] [3 4]]"
    Claim(t != "horse")
  }

  property("(x = y) ~ (x.hashCode = y.hashCode)") = forAll { (x: Tensor.U, y: Tensor.U) =>
    Claim((x == y) == (x.hashCode == y.hashCode))
  }

  property("scalar invalid on vectors") = forAll { (m: Matrix) =>
    val vector = m.tensor.slice(0, 0)
    val res = Try(vector.scalar)
    vector.dims.components match {
      case List(1) => Claim(res.isSuccess)
      case List(_) => Claim(res.isFailure)
      case xs      => sys.error(s"unexpected axes: $xs")
    }
  }

  property("scalars is in rowMajorDims order") = forAll { (t: Tensor.U) =>
    implicit val ct = t.dataType.classTag
    val values = t.scalars.toArray
    Claim(Tensor(t.dataType)(values, 0, t.dims.asRowMajorDims) == t)
  }

  property("scalar sees through singletons") = {
    // [ [[1]] [[2]] [[3]] ]
    val t = Tensor(((1f :: HNil) :: HNil, (2f :: HNil) :: HNil, (3f :: HNil) :: HNil))
    Claim(t.slice(0, 1).scalar == 2f)
  }

  property("t.slices(big).isEmpty") = forAll { (t: Tensor.U) =>
    val xs = t.slices(999999)
    Claim(xs.isEmpty)
  }

  property("axes.insert(axis, n, a).last._1 = axes") = forAll { (axes: Axes) =>
    val axis = axes.rank
    val n = 999
    val got = axes.insert(axis, n, Axis).last
    val expected = (axes, n, Axis)
    Claim(got == expected)
  }

  property("axes.last") = forAll { (tail: Axes, n0: Int) =>
    val axes = NonEmpty(n0, Axis, tail)
    val (s, n1, _) = axes.last
    val got = s.insert(s.rank, n1, Axis)
    Claim(got == axes)
  }

  property("reordering inserts") = forAll(transposeCase) {
    case (axes, i0, j0) =>
      val (i1, j1) = if (i0 < j0) (i0, j0 - 1) else (i0 + 1, j0)
      val lhs = axes.insert(i0, 888888, Axis).insert(j0, 999999, Axis)
      val rhs = axes.insert(j1, 999999, Axis).insert(i1, 888888, Axis)
      Claim(lhs == rhs)
  }

  property("test unsqueeze example from Operators.md") = {
    val axes0 = Shape.axes(3, 4, 5)
    val claim1 = Claim(
      axes0.unsqueeze(3).flatMap(_.unsqueeze(0)) == Some(Shape.axes(1, 3, 4, 5, 1))
    ) &&
      Claim(axes0.unsqueeze(0).flatMap(_.unsqueeze(4)) == Some(Shape.axes(1, 3, 4, 5, 1)))

    val dims0 = Shape.axes(3, 4, 5).asRowMajorDims
    val claim2 = Claim(
      dims0.unsqueeze(3).flatMap(_.unsqueeze(0)) == Some(Shape.axes(1, 3, 4, 5, 1).asRowMajorDims)
    ) &&
      Claim(
        dims0.unsqueeze(0).flatMap(_.unsqueeze(4)) == Some(Shape.axes(1, 3, 4, 5, 1).asRowMajorDims)
      )

    claim1 && claim2
  }

  property("unsqueeze then squeeze") = forAll(transposeCase) {
    case (axes, axis, _) =>
      val dims = axes.asRowMajorDims
      val got = dims.unsqueeze(axis).flatMap(_.squeeze(axis))
      Claim(got == Some(dims))
  }

  property("dims.unsqueeze(9999) = None") = forAll { (axes: Axes) =>
    val dims = axes.asRowMajorDims
    val got = dims.unsqueeze(9999)
    Claim(got == None)
  }

  property("dims(2, 1).squeeze(0) = None") = {
    val dims = Shape.axes(2, 1).asRowMajorDims
    Claim(dims.squeeze(0) == None)
  }

  property("dims(2, 1).squeeze(1) = Some(dims(2))") = {
    val dims0 = Shape.axes(2, 1).asRowMajorDims
    val dims1 = Shape.axes(2).asRowMajorDims
    Claim(dims0.squeeze(1) == Some(dims1))
  }

  property("dims(2, 1).squeeze(2) = Some(dims(2, 1))") = {
    val dims = Shape.axes(2, 1).asRowMajorDims
    Claim(dims.squeeze(2) == Some(dims))
  }

  property("axes.insert(999, n, Axis) = error") = forAll { (axes: Axes) =>
    val got = Try(axes.insert(999, 12, Axis))
    Claim(got.isFailure)
  }

  property("Empty.last = error") = {
    val got = Try(Empty.last)
    Claim(got.isFailure)
  }

  property("dims.squeeze(9999) = None") = forAll { (axes: Axes) =>
    val dims = axes.asRowMajorDims
    val got = dims.squeeze(10)
    Claim(got == None)
  }

  import cats.implicits._

  property("coordsToIndex(coords) catches invalid cases") = forAll { (axes0: Axes, axes1: Axes) =>
    val dims = axes0.asRowMajorDims
    val coords = axes1.as(Coord)
    val got = Try(Shape.coordsToIndex(dims, coords))
    Claim(got.isSuccess == (dims.rank == coords.rank))
  }

  property("dims.traverse") = {
    val axes = Shape.axes(3, 2)
    val dims = axes.asRowMajorDims
    val p = (x: Dim) => x.stride % 2 == 0
    Claim(
      dims.traverse(_ => None) == None &&
        dims.traverse(d => Option(d)) == Some(dims) &&
        dims.traverse(d => Option(d).filter(p)) == None
    )
  }

//not ok
  property("dims.foldRight") = {
    import cats.Eval
    val axes = Shape.axes(2, 2, 2, 2)
    val dims = axes.asRowMajorDims
    val got = dims.foldRight(Eval.now(1)) { (d, prod) =>
      prod.map(_ * d.stride.toInt)
    }
    Claim(got.value == (1 * 2 * 4 * 8))
  }

  property("Tensor[Float](data, 0, dims).squeeze(axis).map(_.dims) = dims.squeeze(axis)") = forAll {
    (axes: Axes, n: Int) =>
      val axis = n & 0xff
      val dims = axes.asRowMajorDims
      val data = new Array[Float](dims.totalSize.toInt)
      val got = Tensor(DataType.Float32)(data, 0, dims).squeeze(axis).map(_.dims)
      val expected = dims.squeeze(axis)
      Claim(got == expected)
  }

//not ok
  property("save -> load works") = forAll { (t0: Tensor.U) =>
    val io = for {
      path <- IO(Files.createTempFile("tensor", "test"))
      _ <- IO(path.toFile.deleteOnExit())
      _ <- Tensor.save(path, t0)
      prop <- Tensor
        .load(path, t0.dataType, t0.axes)
        .use { t1 =>
          IO.pure(Claim(t1 == t0))
        }
    } yield prop

    io.unsafeRunSync
  }

  property("save -> loadMapped works") = forAll { (t0: Tensor.F) =>
    val io = for {
      path <- IO(Files.createTempFile("tensor", "test"))
      _ <- IO(path.toFile.deleteOnExit())
      _ <- Tensor.save(path, t0)
      prop <- Tensor
        .loadMappedRowMajorTensor(DataType.Float32, path, t0.axes)
        .use { t1 =>
          IO.pure(Claim(t1 == t0))
        }
    } yield prop

    io.unsafeRunSync
  }

  def testUnicast(t0: Tensor.U, axes: Axes, goal: Option[Tensor.U]): Prop = {
    val got = t0.broadcastTo(axes)
    goal match {
      case Some(t2) => Claim(got == Success(t2))
      case None     => Claim(got.isFailure)
    }
  }

  def testUnicast(start: Axes, goal: Axes, passes: Boolean): Prop = {
    val got = start.asRowMajorDims.broadcastTo(goal)
    if (passes) {
      Claim(got.map(_.as(Axis)) == Success(goal))
    } else {
      Claim(got.isFailure)
    }
  }

  property("unidirectional broadcast #1") = testUnicast(
    tensor"[9]", // have: 1
    Shape.axes(3), // want: 3
    Some(tensor"[9 9 9]")
  ) // got:  3

  property("unidirectional broadcast #2a") = testUnicast(
    tensor"[2 3]", // have:     2
    Shape.axes(3, 2), // want: 3 x 2
    Some(tensor"[[2 3] [2 3] [2 3]]")
  ) // got:  3 x 2

  property("unidirectional broadcast #2b") = testUnicast(
    tensor"[[2 3]]", // have: 1 x 2
    Shape.axes(3, 2), // want: 3 x 2
    Some(tensor"[[2 3] [2 3] [2 3]]")
  ) // got:  3 x 2

  property("unidirectional broadcast #3") = testUnicast(
    start = Shape.axes(15, 1, 5), // have 15 x 1 x 15
    goal = Shape.axes(15, 3, 5), // want 15 x 3 x 15
    passes = true
  )

  property("unidirectional broadcast #4") =
    testUnicast(start = Shape.axes(1, 2), goal = Shape.axes(3, 2, 1), passes = false)

  property("unidirectional broadcast #5") =
    testUnicast(start = Shape.axes(4, 2, 1), goal = Shape.axes(3, 1, 2, 1), passes = false)

  def testMulticast(lhs: Axes, rhs: Axes, goal: Option[Axes]): Prop = {
    val res = Shape.broadcast(lhs.asRowMajorDims, rhs.asRowMajorDims)
    val got = res.map { case (x, y) => (x.as(Axis), y.as(Axis)) }
    goal match {
      case Some(o) =>
        Claim(got.map(_._1) == Success(o)) && Claim(got.map(_._2) == Success(o))
      case None =>
        Claim(got.isFailure)
    }
  }

  // see https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

  property("multidirectional broadcast #1") = testMulticast(
    Shape.axes(256, 256, 3), // 256 x 256 x 3
    Shape.axes(3), //             3
    Some(Shape.axes(256, 256, 3))
  ) // 256 x 256 x 3

  property("multidirectional broadcast #2") = testMulticast(
    Shape.axes(8, 1, 6, 1), // 8 x 1 x 6 x 1
    Shape.axes(7, 1, 5), //     7 x 1 x 5
    Some(Shape.axes(8, 7, 6, 5))
  ) // 8 x 7 x 6 x 5

  property("multidirectional broadcast #3") = testMulticast(
    Shape.axes(5, 4), // 5 x 4
    Shape.axes(1), //     1
    Some(Shape.axes(5, 4))
  ) // 5 x 4

  property("multidirectional broadcast #4") = testMulticast(
    Shape.axes(5, 4), // 5 x 4
    Shape.axes(4), //     4
    Some(Shape.axes(5, 4))
  ) // 5 x 4

  property("multidirectional broadcast #5") = testMulticast(
    Shape.axes(15, 3, 5), // 15 x 3 x 5
    Shape.axes(15, 1, 5), // 15 x 1 x 5
    Some(Shape.axes(15, 3, 5))
  ) // 15 x 3 x 5

  property("multidirectional broadcast #6") = testMulticast(
    Shape.axes(15, 3, 5), // 15 x 3 x 5
    Shape.axes(3, 5), //      3 x 5
    Some(Shape.axes(15, 3, 5))
  ) // 15 x 3 x 5

  property("multidirectional broadcast #7") = testMulticast(
    Shape.axes(15, 3, 5), // 15 x 3 x 5
    Shape.axes(3, 1), //      3 x 1
    Some(Shape.axes(15, 3, 5))
  ) // 15 x 3 x 5

  property("multidirectional broadcast #8") = testMulticast(
    Shape.axes(3), //    3
    Shape.axes(4), //    4
    None
  ) // fail

  property("multidirectional broadcast #9") = testMulticast(
    Shape.axes(2, 1), //     2 x 1
    Shape.axes(8, 4, 3), // 8 x 4 x 3
    None
  ) //      fail

  val stackGen: Gen[(Tensor.Unknown, Int)] =
    for {
      t <- genTensorU
      n = t.dims.rank
      gen = if (n < 1) Gen.const(0) else Gen.choose(0, n)
      axis <- gen
    } yield (t, axis)

  property("stack(t.dataType, axis)(t.slices(axis)) = t") = forAll(stackGen) {
    case (t, axis) =>
      val slices = t.slices(axis).map(_._2)
      if (slices.nonEmpty) {
        val got = Tensor.stack(t.dataType, axis)(slices)
        val expected = Success(t)
        Claim(got == expected)
      } else {
        Claim(true)
      }
  }

  property("chunk example 1") = {
    val t0 = tensor"[0 1 2 3]"
    val res = t0.chunk(axis = 0, size = 2).get
    val expected = Seq(tensor"[0 1]", tensor"[2, 3]")
    val c1 = Claim(res == expected)
    val c2 = Claim(Tensor.unchunk(t0.dataType, 0)(res).get == t0)
    c1 && c2
  }
  property("chunk example 2") = {
    val t0 = tensor"[[0 1 2 3 4 5]]"
    val res = t0.chunk(axis = 1, size = 3).get
    val expected = Seq(tensor"[[0 1 2]]", tensor"[[3, 4, 5]]")
    val c1 = Claim(res == expected)
    val c2 = Claim(Tensor.unchunk(t0.dataType, 1)(res).get == t0)
    c1 && c2
  }
  property("chunk example 3") = {
    val t0 = tensor"[[0 1 2 3], [4 5 6 7]]"
    val res = t0.chunk(axis = 1, size = 2).get
    val expected = Seq(tensor"[[0 1], [4 5]]", tensor"[[2, 3], [6, 7]]")
    val c1 = Claim(res == expected)
    val c2 = Claim(Tensor.unchunk(t0.dataType, 1)(res).get == t0)
    c1 && c2
  }

  property("chunking broadcast works") = {
    val t0 = Tensor.const(DataType.Float32)(3f, Shape.axes(9, 9))
    val res = t0.chunk(axis = 1, size = 3).get
    val t1 = Tensor.const(DataType.Float32)(3f, Shape.axes(9, 3))
    val expected = Seq(t1, t1, t1)
    Claim(res == expected)
  }

  property("reshape 24 to 2, 3, 4") = {
    val t0 = tensor"[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]"
    val t1 = t0.reshape(Shape.axes(2L, 3L, 4L)).get
    val expected = tensor"""[[[ 0  1  2  3]
                              [ 4  5  6  7]
                              [ 8  9 10 11]]

                             [[12 13 14 15]
                              [16 17 18 19]
                              [20 21 22 23]]]"""
    val c1 = Claim(t1 == expected)

    val t2 = t0.reshape(Shape.axes(-1L, 3L, 4L)).get
    val c2 = Claim(t2 == expected)

    val c3 = Claim(t2.reshape(Shape.axes(24L)).get == t0)
    c1 && c2 && c3
  }

  property("we can always reshape an axes if the totalSize is the same") = forAll {
    (axes0: Axes, axes1: Axes) =>
      val leftToRight = axes0.inferReshapeTo(axes1)
      val rightToLeft = axes1.inferReshapeTo(axes0)
      val c1 = Claim(leftToRight.isSuccess == rightToLeft.isSuccess)
      val c2 = if (axes0.totalSize == axes1.totalSize) {
        Claim(leftToRight == Success(axes1)) &&
        Claim(rightToLeft == Success(axes0))
      } else {
        Claim(leftToRight.isFailure && rightToLeft.isFailure)
      }
      // if the size of axes1 divides axes0, we can add -1 and infer a size
      val c3 = if (axes0.totalSize % axes1.totalSize == 0) {
        val axes11 = Shape.NonEmpty(-1, Shape.Axis, axes1)
        Claim(axes0.inferReshapeTo(axes11).isSuccess)
      } else Claim(true)

      c1 && c2 && c3
  }
  property("we can infer the size of 0 dims in inferReshapeTo") = forAll { (axes: Axes) =>
    val comps = axes.components
    def allComps(c: List[Long]): List[List[Long]] =
      c match {
        case Nil => List(Nil)
        case h :: tail =>
          val rest = allComps(tail)
          rest.flatMap { tail =>
            List(h :: tail, 0L :: tail)
          }
      }
    allComps(comps)
      .map { Shape.axes(_: _*) }
      .foldLeft(Claim(true)) { (c, axes0) =>
        val a1 = axes.inferReshapeTo(axes0)
        c && Claim(a1 == Success(axes))
      }
  }

  property("converting to and from Axes works") = forAll(genAxes) { axes =>
    val t = Tensor.convertAxesToTensor(axes)
    val axes1 = t.convertDataToAxes.get
    Claim(axes1 == axes)
  }

  property("concatenate matches numpy #1") = {
    //see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
    val t1 = tensor"[[1, 2], [3, 4]]"
    val t2 = tensor"[[5, 6]]"
    val t3 = Tensor.concatenate(DataType.Float32, axis = 0L)(List(t1, t2)).get
    val expected = tensor"[[1, 2], [3, 4], [5, 6]]"
    Claim(t3 == expected)
  }

  property("concatenate matches numpy #2") = {
    //see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
    val t1 = tensor"[[1, 2]]"
    val t2 = tensor"[[3, 4]]"
    val t3 = Tensor.concatenate(DataType.Float32, axis = 1L)(List(t1, t2)).get
    val expected = tensor"[[1, 2, 3, 4]]"
    Claim(t3 == expected)
  }

  property("we can concatenate then chunk a single tensor on any axis") = {
    // Make sure to generate rank 1 or greater tensors
    case class Arg(t: Tensor.U, axis: Long, len: Long, copies: Int)
    lazy val argGen: Gen[Arg] =
      genTensorU.flatMap { ten =>
        val axes = ten.axes
        if (axes.nonEmpty) {
          Gen
            .choose(0L, axes.rank - 1L)
            .flatMap { axis =>
              val len = axes.at(axis).fold(0L)(_._1)
              Gen.choose(1, 10).map { copies =>
                Arg(ten, axis, len, copies)
              }
            }
        } else argGen
      }

    forAll(argGen) {
      case Arg(t, axis, len, copies) =>
        //see: https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
        val cat = Tensor.concatenate(t.dataType, axis = axis)(List.fill(copies)(t.asDataTyped)).get
        val chunk = cat.chunk(axis = axis, size = len).get.toList
        chunk.tail.foldLeft(Claim(t == chunk.head)) { (claim, chunkT) =>
          claim && Claim(t == chunkT)
        }
    }
  }

  property("sumAxes tests") = {
    // see the examples at: https://github.com/onnx/onnx/blob/master/docs/Operators.md#reducesum
    val t = tensor"[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]"
    val c1 = {
      val expected = tensor"[[[78]]]"
      val got = t.sumAxesKeep().get
      Claim(got == expected)
    }

    val c2 = {
      val expected = tensor"[[[4 6]], [[12 14]], [[20 22]]]"
      val got = t.sumAxesKeep(1L).get
      Claim(got == expected)
    }

    val c3 = {
      val expected = tensor"[[4 6], [12 14], [20 22]]"
      val got = t.sumAxes(1L).get
      Claim(got == expected)
    }

    val c4 = {
      val expected = tensor"[[[10]], [[26]], [[42]]]"
      val got = t.sumAxesKeep(1L, 2L).get
      Claim(got == expected)
    }

    val c5 = {
      val expected = tensor"[10, 26, 42]"
      val got = t.sumAxes(1L, 2L).get
      Claim(got == expected)
    }

    c1 && c2 && c3 && c4 && c5
  }

  property("maxAxes tests") = {
    val t2d = tensor"[[1, 10, 100], [1, 20, 200], [3, 30, 300]]"

    val c1 = {
      val expected = tensor"[3, 30, 300]"
      val got = t2d.maxAxes(0L).get
      Claim(got == expected)
    }

    val c2 = {
      val expected = tensor"[100, 200, 300]"
      val got = t2d.maxAxes(1L).get
      Claim(got == expected)
    }

    val c3 = {
      val got = t2d.maxAxes(0L, 1L).get
      Claim(got == tensor"300")
    }

    c1 && c2 && c3
  }

  property("sum all axes matches sum") = forAll { ten: Tensor.U =>
    implicit val tolerance = ten.dataType match {
      case DataType.BFloat16 => Epsilon(0.5f)
      case DataType.Float16  => Epsilon(0.2f)
      case DataType.Float32  => Epsilon(1e-2f)
      case _                 => Epsilon(1e-3f)
    }

    if (ten.rank > 0) {
      val allAxes = (0 until ten.rank).map(_.toLong)
      ten.sumAxes(allAxes: _*).get =~= Tensor.scalar(ten.dataType)(ten.sum)
    } else Claim(true)
  }

  property("concatenate then sum is the same as muliplying by a constant") = {
    // Make sure to generate rank 1 or greater tensors
    case class Arg(t: Tensor.U, axis: Long, copies: Int)
    lazy val argGen: Gen[Arg] =
      genTensorU.flatMap { ten =>
        val axes = ten.axes
        if (axes.nonEmpty) {
          Gen
            .choose(0L, axes.rank - 1L)
            .flatMap { axis =>
              Gen.choose(1, 10).map { copies =>
                Arg(ten, axis, copies)
              }
            }
        } else argGen
      }

    forAll(argGen) {
      case Arg(t, axis, copies) =>
        val num = OnnxNumber.forDataType(t.dataType)
        val cat = Tensor.concatenate(t.dataType, axis = axis)(List.fill(copies)(t.asDataTyped)).get
        val copiesTpe = List.fill(copies)(num.one).reduce(num.plus(_, _))

        implicit val tolerance = t.dataType match {
          case DataType.BFloat16 => Epsilon(0.5f)
          case DataType.Float16  => Epsilon(0.2f)
          case DataType.Float32  => Epsilon(1e-3f)
          case _                 => Epsilon(1e-5f)
        }

        val c1 = {
          val sum0 = t.sumAxes(axis).get
          val got = cat.sumAxes(axis).get
          val expected = sum0.map { e =>
            num.times(copiesTpe, e)
          }
          got =~= expected
        }

        val c2 = {
          val sum0 = t.sumAxesKeep(axis).get
          val got = cat.sumAxesKeep(axis).get
          val expected = sum0.map { e =>
            num.times(copiesTpe, e)
          }
          got =~= expected
        }

        c1 && c2
    }
  }

  property("select matches numpy slice behavior") = {
    import Shape.AxisRange

    val c1 = {
      val t0 = tensor"[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]"
      val got = t0.select(AxisRange(0L, 1L, 1L) :: Nil)
      val expected = tensor"[[[1, 2], [3, 4]]]"
      Claim(got == expected)
    }

    val c2 = {
      val t0 = tensor"[[1, 2, 3], [4, 5, 6]]"
      val got = t0.select(AxisRange(0L, 2L, 1L) :: AxisRange(2L, 0L, -1L) :: Nil)
      val expected = tensor"[[3, 2], [6, 5]]"
      Claim(got == expected)
    }

    val c3 = {
      val t0 = tensor"[[1, 2, 3, 4], [5, 6, 7, 8]]"
      val got = t0.select(AxisRange(1L, 2L, 1L) :: AxisRange(0L, 3L, 2L) :: Nil)
      val expected = tensor"[[5, 7]]"
      Claim(got == expected)
    }

    c1 && c2 && c3
  }
}
