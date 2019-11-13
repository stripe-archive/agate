# Agate

Agate is a library designed to support evaluation of deep learning
models encoded in [ONNX](https://github.com/onnx/onnx).

## Quick Start

To include Agate in your SBT project, use the following snippet:

```scala
libraryDependencies += "com.stripe" %% "agate-core" % "0.0.10"
```

To include Agate in your Bazel project, add the following to your
`jvm_dependencies.yaml` file (or whatever file you use with
[bazel-deps](https://github.com/johnynek/bazel-deps)):

```
dependencies:
  com.stripe:
    agate-core:
      lang: scala
      version: "0.0.10"
```

## Project Layout

Agate has a few modules:

* `core` contains all the actual library code
* `docs` is used to build type-checked documentation
* `bench` contains all our benchmarks

Currently only `core` is published (as `agate-core`).

## Details

Here's an example of how to create a literal 2x3 matrix (i.e. a matrix
with 2 rows and 3 columns):

```scala
import com.stripe.agate.tensor.Tensor

// build a literal tensor using shapeless
val matrix0 = Tensor(((1F, 2F, 3F), (4F, 5F, 6F)))
// matrix0: Tensor[com.stripe.agate.tensor.DataType.Float32.type] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

import com.stripe.agate.tensor.TensorParser.Interpolation

// build a literal tensor using compile-time parsing
val matrix1 = tensor"[[1 2 3] [4 5 6]]"
// matrix1: Tensor[com.stripe.agate.tensor.DataType.Float32.type] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

matrix0 == matrix1
// res0: Boolean = true
```

## Common Tasks

This section describes how to do many common development tasks in
Agate. Commands are often run from SBT but some commands (`coverage`,
`fmt`, `gendoc`, and `validate`) are instead run from the shell.

* [Launching SBT](#launching-sbt)
* [Building and testing](#building-and-testing)
* [Code formatting](#code-formatting)
* [Doc generation](#doc-generation)
* [Releasing](#releasing)
* [Test coverage](#test-coverage)
* [Benchmarking](#benchmarking)

### Launching SBT

Many of these commands are run from within SBT. You can start SBT with
`sbt` (you may need to do `brew install sbt` if it is not already
installed), which will give you a prompt:

```
erik@icebreaker$ sbt
[info] Loading settings for project global-plugins from javap.sbt ...
[info] Loading global plugins from /Users/erik/.sbt/1.0/plugins
[info] Loading settings for project agate-build from plugins.sbt,scalapb.sbt ...
[info] Loading project definition from /Users/erik/stripe/agate/project
[info] Loading settings for project agate from version.sbt,build.sbt ...
[info] Set current project to agate (in build file:/Users/erik/stripe/agate/)
[info] sbt server started at local:///Users/erik/.sbt/1.0/server/deaf34ab09dd5216286e/sock
sbt:agate>
```

### Building and testing

From within SBT, run `compile` to compile the all code, and run `test`
to run the all the tests.

If you're only working in `core` you can scope the commands to have
them run a bit faster, e.g. `core/compile` and `core/test`.

You can also use `testOnly` with wildcards to only run certain tests.
For example `testOnly *Tensor*` runs `com.stripe.agate.TensorTest` and
`com.stripe.agate.TensorParserTest`. This command can also be scoped
as above.

Finally, if you want to test *everything* that could be tested, you
can use `./runall` to run tests, generate docs, and format the code.

### Code formatting

We use [Scalafmt](https://scalameta.org/scalafmt/) to format our code.
This is not done automatically, but unformatted code will likely fail
in CI.

To format your code, run `./fmt` from the shell. You can also format
your code from SBT using the `scalafmtAll` command.

### Doc generation

We use [MDoc](https://github.com/scalameta/mdoc) to generate (and
type-check) our documentation. You can generate the documentation from
the shell with `./gendoc`. (You can also run `docs/mdoc` from SBT.)

Markdown files in our project root (such as `README.md`) are generated
and you should not edit them directly. Instead, edit `docs/README.md`
and then generate the documentation as above. The `./gendoc` command
will try to catch simple cases where you edit the generated files by
mistake.

We generate documentation to ensure that the code examples compile,
and to allow us to consistently template things like version numbers.

### Releasing

To release, run `release` from within SBT.

You'll be prompted to confirm the current version and select a new
version (suffixed with `SNAPSHOT`). In general, incrementing the last
number (the *patch* version) is the right thing to do, although for
breaking changes incremental the middle number (*minor* version) or
the first number (*major* version) would be preferred.

The command will package the release for all supported Scala versions,
publish the jars, tag the release, and push the results to Git.

After `release` finishes, you should also edit `build.sbt` and update
the `readmeVersion` val. This should point to the most
recently-released version of Agate.

### Test coverage

To measure code coverage, run `./coverage` from the shell.

This command will clean the build, then build and run the full test
suite under Scala 2.12. On completion it will open an HTML summary of
the results in your web browser.

### Benchmarking

We use [JMH](https://openjdk.java.net/projects/code-tools/jmh/) for
benchmarking, and put our benchmarks in the `bench` subproject. We use
the SBT command `bench/jmh:run` to run them.

For example, to run the `MatrixMultBench` benchmark you would run:

```
bench/jmh:run com.stripe.agate.bench.MatrixMultBench -i 3 -wi 3 -f1 -t1
```

The options here are:

* `-wi` is the number of "warmup" runs (before benchmarking)
* `-i` is the number of runs to benchmark
* `-f` is the number of processes to use (usually `1` is fine)
* `-t` is the number of threads to use (often `1` is fine)

Higher numbers for `-i` and `-wi` will take longer, but result in less
uncertain benchmaring results.

## Authors

Agate was written by Erik Osheim, Oscar Boykin, Tom Switzer, and Rob Story
