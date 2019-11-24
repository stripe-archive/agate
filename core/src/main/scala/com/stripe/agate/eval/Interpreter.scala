package com.stripe.agate.eval

import java.nio.file.Paths
import cats.effect.IO

object Interpreter {
  def main(args: Array[String]): Unit =
    Model
      .load(Paths.get(args(0)))
      .use { m =>
        IO.delay(println(m))
      }
      .unsafeRunSync()
}
