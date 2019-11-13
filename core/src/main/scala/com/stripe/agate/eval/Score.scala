package com.stripe.agate.eval

import cats.ApplicativeError
import cats.data.Validated
import cats.effect.{ExitCode, IO}
import com.monovore.decline.{Argument, Command, Opts}
import com.stripe.agate.tensor.{Tensor, TensorParser}
import java.nio.file.Path
import scala.util.{Failure, Success, Try}

import cats.implicits._

case class Score(onnxPath: Path, inputs: Registers) extends Agate.Mode {

  def run: IO[ExitCode] =
    Model.load(onnxPath).use { m =>
      for {
        res <- ApplicativeError[IO, Throwable].fromTry(m.run(inputs))
        out = res.registers(m.output._1)
        _ <- IO(println(out.toString))
      } yield ExitCode.Success
    }
}

object Score {

  private val Item = """^([^= ]+)=(.+)$""".r

  implicit val argTensor: Argument[(Register, Tensor.Unknown)] =
    new Argument[(Register, Tensor.Unknown)] {
      val defaultMetavar = "register=tensor"
      def read(string: String) =
        string match {
          case Item(s1, s2) =>
            Try(TensorParser.int64.unsafeFromString(s2)) match {
              case Success(t) =>
                Validated.valid((Register(s1), t))
              case Failure(_) =>
                Validated.invalidNel(s"could not parse tensor in $s2")
            }
          case _ =>
            Validated.invalidNel(s"could not parse register=tensor in $string")
        }
    }

  val command: Command[Score] = {
    val path = Opts.argument[Path]("path-to-proto")
    val ts = Opts.arguments[(Register, Tensor.Unknown)]("input tensors")

    val opts = (path, ts).mapN { (p, ts) =>
      Score(p, Registers(ts.toList.toMap))
    }
    Command("score", "compute the score of an onnx model")(opts)
  }
}
