lazy val readmeVersion = "0.0.10"

lazy val scalacheckVersion = "1.14.0"

lazy val agateSettings = Seq(

  organization := "com.stripe",
  scalaVersion := "2.12.8",
  crossScalaVersions := Seq("2.11.12", "2.12.8"),

  libraryDependencies ++=
    "com.chuusai" %% "shapeless" % "2.3.3" ::
    "com.stripe" %% "dagon-core" % "0.3.0" ::
    "com.lihaoyi" %% "fastparse" % "1.0.0" ::
    "com.monovore" %% "decline" % "0.4.2" ::
    "org.scalacheck" %% "scalacheck" % scalacheckVersion % "test" ::
    "org.typelevel" %% "claimant" % "0.1.1" % "test" ::
    "org.typelevel" %% "cats-core" % "1.4.0" ::
    "org.typelevel" %% "cats-effect" % "1.2.0" ::
    "org.typelevel" %% "paiges-core" % "0.2.4" ::
    Nil,

  scalacOptions ++= options,
  scalacOptions in (Compile, console) ~= { _.filterNot("-Xlint" == _) },
  scalacOptions in (Test, console) := (scalacOptions in (Compile, console)).value,

  testOptions in Test ++=
    Tests.Argument(TestFrameworks.ScalaCheck, "-verbosity", "1") ::
    Tests.Argument(TestFrameworks.ScalaCheck, "-minSuccessfulTests", "5000") ::
    Nil,

  // TODO: these should go in Publish but i can't figure out how to
  // make that work. ugh. sorry!
  releaseIgnoreUntrackedFiles := true,
  releaseCrossBuild := true,

  // ignore generated files in coverage stats
  coverageExcludedPackages := """onnx\.onnx\..*;onnx\.onnx_operators\..*""",

  // support type lambda syntax
  addCompilerPlugin("org.typelevel" %% "kind-projector" % "0.10.0"),

  // optimize for-comprehensions
  addCompilerPlugin("com.olegpy" %% "better-monadic-for" % "0.3.0-M4"),

  PB.targets in Compile := Seq(
    scalapb.gen() -> (sourceManaged in Compile).value
  ))

lazy val agate = project
  .in(file("."))
  .settings(name := "agate")
  .settings(agateSettings: _*)
  .settings(Publish.skip: _*)
  .aggregate(core, laws, tests, docs, bench)
  .dependsOn(core, laws, tests, docs, bench)

lazy val core = project
  .in(file("core"))
  .settings(name := "agate-core")
  .settings(agateSettings: _*)
  .settings(mainClass in assembly := Some("com.stripe.agate.eval.Agate"))
  .settings(Publish.settings: _*)

lazy val laws = project
  .in(file("laws"))
  .dependsOn(core)
  .settings(name := "agate-laws")
  .settings(agateSettings: _*)
  .settings(libraryDependencies += "org.scalacheck" %% "scalacheck" % scalacheckVersion)
  .settings(Publish.settings: _*)

lazy val tests = project
  .in(file("tests"))
  .dependsOn(core, laws)
  .settings(name := "agate-tests")
  .settings(agateSettings: _*)
  .settings(Publish.skip: _*)

lazy val docs = project
  .in(file("mdoc")) // important: it must not be docs
  .settings(name := "agate-mdoc")
  .settings(agateSettings: _*)
  .settings(Publish.skip: _*)
  .settings(
    mdocVariables := Map("VERSION" -> readmeVersion),
    mdocOut := file("."))
  .dependsOn(core)
  .enablePlugins(MdocPlugin)

lazy val bench = project
  .in(file("bench"))
  .enablePlugins(JmhPlugin)
  .settings(name := "agate-bench")
  .settings(agateSettings: _*)
  .settings(Publish.skip: _*)
  .dependsOn(core)

// scalac options

lazy val options = Seq(
  "-deprecation",                      // Emit warning and location for usages of deprecated APIs.
  "-encoding", "utf-8",                // Specify character encoding used by source files.
  "-explaintypes",                     // Explain type errors in more detail.
  "-feature",                          // Emit warning and location for usages of features that should be imported explicitly.
  "-language:existentials",            // Existential types (besides wildcard types) can be written and inferred
  "-language:experimental.macros",     // Allow macro definition (besides implementation and application)
  "-language:higherKinds",             // Allow higher-kinded types
  "-language:implicitConversions",     // Allow definition of implicit functions called views
  "-unchecked",                        // Enable additional warnings where generated code depends on assumptions.
  "-Xcheckinit",                       // Wrap field accessors to throw an exception on uninitialized access.
  "-Xfatal-warnings",                  // Fail the compilation if there are any warnings.
  "-Xfuture",                          // Turn on future language features.
  "-Xlint:adapted-args",               // Warn if an argument list is modified to match the receiver.
  "-Xlint:by-name-right-associative",  // By-name parameter of right associative operator.
  ///"-Xlint:constant",                   // Evaluation of a constant arithmetic expression results in an error.
  "-Xlint:delayedinit-select",         // Selecting member of DelayedInit.
  "-Xlint:doc-detached",               // A Scaladoc comment appears to be detached from its element.
  "-Xlint:inaccessible",               // Warn about inaccessible types in method signatures.
  "-Xlint:infer-any",                  // Warn when a type argument is inferred to be `Any`.
  "-Xlint:missing-interpolator",       // A string literal appears to be missing an interpolator id.
  "-Xlint:nullary-override",           // Warn when non-nullary `def f()' overrides nullary `def f'.
  "-Xlint:nullary-unit",               // Warn when nullary methods return Unit.
  "-Xlint:option-implicit",            // Option.apply used implicit view.
  "-Xlint:package-object-classes",     // Class or object defined in package object.
  "-Xlint:poly-implicit-overload",     // Parameterized overloaded implicit methods are not visible as view bounds.
  "-Xlint:private-shadow",             // A private field (or class parameter) shadows a superclass field.
  "-Xlint:stars-align",                // Pattern sequence wildcard must align with sequence component.
  "-Xlint:type-parameter-shadow",      // A local type parameter shadows a type already in scope.
  "-Xlint:unsound-match",              // Pattern match may not be typesafe.
  "-Yno-adapted-args",                 // Do not adapt an argument list (either by inserting () or creating a tuple) to match the receiver.
  "-Ypartial-unification",             // Enable partial unification in type constructor inference
  //"-Ywarn-dead-code",                  // Warn when dead code is identified.
  ///"-Ywarn-extra-implicit",             // Warn when more than one implicit parameter section is defined.
  "-Ywarn-inaccessible",               // Warn about inaccessible types in method signatures.
  "-Ywarn-infer-any",                  // Warn when a type argument is inferred to be `Any`.
  "-Ywarn-nullary-override",           // Warn when non-nullary `def f()' overrides nullary `def f'.
  "-Ywarn-nullary-unit",               // Warn when nullary methods return Unit.
  //"-Ywarn-numeric-widen",              // Warn when numerics are widened.
  ///"-Ywarn-unused:implicits",           // Warn if an implicit parameter is unused.
  ///"-Ywarn-unused:imports",             // Warn if an import selector is not referenced.
  ///"-Ywarn-unused:locals",              // Warn if a local definition is unused.
  ///"-Ywarn-unused:params",              // Warn if a value parameter is unused.
  ///"-Ywarn-unused:patvars",             // Warn if a variable bound in a pattern is unused.
  ///"-Ywarn-unused:privates",            // Warn if a private member is unused.
  ////"-Ywarn-unused-imports",             // Warn if an import selector is not referenced.
  "-Ywarn-value-discard"               // Warn when non-Unit expression results are unused.
)
