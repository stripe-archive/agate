#!/usr/bin/env bash

# whether to show verbose SBT output or not.
#
# set this to 0 to hide output for successful commands. this makes it
# easier to visually parse the jenkins logs and find steps that failed.
#
# set this to 1 to unconditionally generate full build output for SBT
# commands. you can use this to debug cases where you're worried that
# SBT commands are failing without returning a non-zero exit status.
VERBOSE=0

echo ""
echo "=============== AGATE BUILD STARTS HERE ====================="
echo ""

# exit immediately when a command fails
set -e

# when the shell exits, execute the following commands
trap 'echo ""; echo "=============== AGATE BUILD ENDS HERE ====================="; echo ""; ./ci/aggregate_junit' EXIT

# parse the project version out of version.sbt
VERSION=$(sed -n 's/version in ThisBuild := "\(.*\)"/\1/p' version.sbt)

echo "agate version set to: $VERSION"

# Cache the build artifacts... I kind of feel like this is asking for
# trouble with Ivy, since it gets into a bad state so often.
mkdir -p /cache/ivy2

# announce a build failure and exit with non-zero status
die() {
    echo ""
    echo "BUILD FAILED!"
    exit 1
}

# set up sbt options to use
SBTOPTS="-mem 2048"
SBTOPTS="$SBTOPTS -Dsbt.log.noformat=true"
SBTOPTS="$SBTOPTS -Dsbt.ivy.home=/cache/ivy2/"
SBTOPTS="$SBTOPTS -Dsbt.override.build.repos=true"
SBTOPTS="$SBTOPTS -Dsbt.repository.config=ci/repositories"

runcmd() {
    CMD=$1
    shift
    OUTPUT=`$CMD "$@" 2>&1`
    RES=$?
    if [ $RES -ne 0 ]; then
        echo "ERROR: $CMD $@ failed ($RES)!"
        echo ""
        echo "$OUTPUT"
        die
    else
        NUM=`echo "$OUTPUT" | awk 'END{print NR}'`
        echo "OK: $CMD passed ($NUM lines of output)"
        # only show the SBT output for passing steps if VERBOSE is set
        # to a non-zero value.
        if ( [ -n "$VERBOSE" ] && [ "$VERBOSE" -ne 0 ] ); then
            echo ""
            echo "$OUTPUT"
        fi
    fi
}

# run the given SBT command using the default options.
runsbt() {
    echo ""
    echo "running sbt $@ ..."
    runcmd sbt $SBTOPTS "$@"
}

# do not exit immediately when a command fails. since we are buffering
# SBT command output, we want to display error output before exiting.
set +e

# we prepend tasks with `+` to run them across all scala versions.
# in some cases (mdoc and scalafmt) this is not needed.
runsbt +clean
runsbt +compile
runsbt +test
runsbt +doc
runsbt docs/mdoc
#runsbt scalafmtCheckAll # see https://github.com/scalameta/scalafmt/issues/1399
runsbt core/assembly

# now copy the assembly to /build. we set -e because we want to fail
# immediately if something goes wrong. 
set -e
echo ""
echo "copying assembly to build/"
mkdir -p /build
cp "core/target/scala-2.12/agate-core-assembly-$VERSION.jar" /build/agate-assembly.jar
echo "OK"

# go back to set +e because we want to let runcmd terminate if things go wrong.
set +e
echo ""
echo "running native-image ..."
mkdir -p /build/bin
runcmd /tmp/graal/bin/native-image --static -jar /build/agate-assembly.jar /build/bin/agate

# we want to see if docs/mdoc generated any changes. if so, we should
# fail the build so the user commits the modified documentation.
set -e
echo ""
echo "checking uncommitted files..."
UNCOMMITTED=`git status -s -uno`
if [ -n "$UNCOMMITTED" ]; then
    echo "ERROR: uncommitted files found!"
    echo ""
    echo "$UNCOMMITTED"
    die
fi

# alright, we finished!
echo "OK: no uncommitted files found"
echo ""
echo "build completed successfully!"
