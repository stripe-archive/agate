import sbt._
import sbt.Keys._
import com.jsuereth.sbtpgp.PgpKeys._
import aether.AetherKeys._
import sbtrelease.ReleasePlugin.autoImport.releasePublishArtifactsAction

object Publish {
  val nexus = "https://oss.sonatype.org/"

  def overrideSnapshotRepo = for {
    name <- sys.props.get("agate.snapshots.name")
    url <- sys.props.get("agate.snapshots.url")
  } yield (name at url)

  def useAether: Boolean = sys.props.get("agate.snapshots.aether") match {
    case Some("true") => true
    case Some("false") | None => false
    case Some(other) =>
      throw new IllegalArgumentException(s"agate.snapshots.aether: expected true or false, found $other")
  }

  def getPublishTo(snapshot: Boolean) = {
    if (snapshot) {
      overrideSnapshotRepo.orElse(Some("Snapshots" at nexus + "content/repositories/snapshots"))
    } else {
      Some("Releases" at nexus + "service/local/staging/deploy/maven2")
    }
  }

  lazy val settings = Seq(
    homepage := Some(url("https://github.com/stripe/agate")),
    publishMavenStyle := true,
    (Test / publishArtifact) := false,
    pomIncludeRepository := Function.const(false),
    publish := {
      if (isSnapshot.value && useAether) {
        (aetherDeploy.value: @sbtUnchecked)
      } else {
        (publish.value: @sbtUnchecked)
      }
    },
    publishTo := getPublishTo(isSnapshot.value),
    (Test / publishArtifact) := false,
    publishConfiguration := publishConfiguration.value.withOverwrite(true),
    pomIncludeRepository := Function.const(false),
    releasePublishArtifactsAction := publishSigned.value,
    pomExtra := (
      <licenses>
        <license>
          <name>Apache 2</name>
          <url>http://www.apache.org/licenses/LICENSE-2.0.txt</url>
          <distribution>repo</distribution>
          <comments>A business-friendly OSS license</comments>
        </license>
      </licenses>
      <developers>
        <developer>
          <name>Erik Osheim</name>
          <email>erik@stripe.com</email>
          <organization>Stripe</organization>
          <organizationUrl>https://stripe.com</organizationUrl>
        </developer>
        <developer>
          <name>Oscar Boykin</name>
          <email>oscar@stripe.com</email>
          <organization>Stripe</organization>
          <organizationUrl>https://stripe.com</organizationUrl>
        </developer>
      </developers>)
  )

  lazy val skip = Seq(
    publishTo := getPublishTo(isSnapshot.value),
    publish := (()),
    publishLocal := (()),
    publishArtifact := false
  )
}
