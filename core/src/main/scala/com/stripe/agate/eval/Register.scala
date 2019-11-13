package com.stripe.agate.eval

import cats.data.{Validated, ValidatedNel}
import cats.implicits._
import com.stripe.agate.tensor._
import scala.util.{Failure, Success, Try}

import Shape.Axes

class Register private (override val toString: String) {

  override val hashCode: Int =
    toString.hashCode

  override def equals(that: Any): Boolean =
    that match {
      case r: Register => toString eq r.toString
      case _           => false
    }
}

object Register {

  def apply(s: String): Register =
    new Register(s.intern)
}

case class Registers(registers: Map[Register, Tensor.Unknown]) {

  def size: Int = registers.size

  def replaceAll(regs: Registers): Registers =
    regs.registers.foldLeft(this) { case (rs, (r, t)) => rs.replace(r, t) }

  def merge(regs: Registers): Try[Registers] =
    if (this.size < regs.size) regs.merge(this)
    else regs.registers.toList.foldM(this) { case (rs, (r, t)) => rs.create(r, t) }

  def getAndUnify(register: Register, dt: DataType): Try[Tensor[dt.type]] =
    get(register).flatMap(_.assertDataType(dt))

  def get(register: Register): Try[Tensor.Unknown] =
    registers.get(register) match {
      case Some(tensor) => Success(tensor)
      case None         => Try(sys.error(s"missing register $register (available: ${registers.keys})"))
    }

  def replace[D <: DataType](register: Register, tensor: Tensor[D]): Registers =
    Registers(registers.updated(register, tensor))

  def create[D <: DataType](register: Register, tensor: Tensor[D]): Try[Registers] =
    registers.get(register) match {
      case Some(tensor0) =>
        Try(
          sys.error(
            s"writing ${tensor.dataType}, ${tensor.axes} to $register failed (already wrote: ${tensor0.dataType} ${tensor0.axes}"
          )
        )
      case None =>
        Success(Registers(registers.updated(register, tensor)))
    }

  def validateTypes(types: Map[Register, (DataType, Axes)]): Try[Unit] = {
    def valid(r: Register, dt: DataType, ax: Axes): Either[(DataType, Axes), Boolean] =
      registers.get(r) match {
        case None => Right(false)
        case Some(ten) =>
          if ((ten.dataType == dt) && (ten.axes == ax)) Right(true)
          else Left((ten.dataType, ten.axes))
      }

    val invalid = types.iterator.flatMap {
      case (r, (d, a)) =>
        valid(r, d, a) match {
          case Right(true)  => Iterator.empty
          case Right(false) => Iterator.single(s"register $r missing from the register set")
          case Left((gotD, gotA)) =>
            Iterator.single(s"register $r expected $d/$a but found $gotD/$gotA")
        }
    }

    if (invalid.isEmpty) Success(())
    else Failure(new Exception(invalid.mkString("\n")))
  }

  def coerceTypes(types: Map[Register, (DataType, Axes)]): Try[Registers] = {
    def coerce(
        r: Register,
        dt: DataType,
        ax: Axes
    ): ValidatedNel[(Register, DataType, Axes, DataType, Axes), Option[
      (Register, Tensor.Unknown)
    ]] =
      registers.get(r) match {
        case None => Validated.valid(None)
        case Some(ten) =>
          val ten1 =
            if (ten.dataType == dt) ten
            else ten.cast(dt)
          if (ten1.axes == ax) Validated.valid(Some((r, ten1)))
          else {
            ten1.broadcastTo(ax) match {
              case Success(ten2) => Validated.valid(Some((r, ten2)))
              case Failure(_)    => Validated.invalidNel((r, dt, ax, ten.dataType, ten.axes))
            }
          }
      }

    val checked = types.toList.traverse {
      case (r, (d, a)) =>
        coerce(r, d, a)
    }

    val all = checked.map { changes =>
      changes.foldLeft(registers) {
        case (registers, None)         => registers
        case (registers, Some((r, t))) => registers.updated(r, t)
      }
    }
    all match {
      case Validated.Valid(v) => Success(Registers(v))
      case Validated.Invalid(errs) =>
        val msg = errs.toList
          .map {
            case (r, expd, expa, gotd, gota) =>
              s"register: $r could not coerce $gotd/$gota to $expd/$expa"
          }
          .mkString("\n")
        Failure(new Exception(msg))
    }
  }
}

object Registers {
  val Empty: Registers = Registers(Map.empty)
  def empty: Registers = Empty
}
