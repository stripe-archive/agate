import sbt._
import sbt.Keys._
import com.jsuereth.sbtpgp.PgpKeys._
import aether.AetherKeys._
import sbtrelease.ReleasePlugin.autoImport.releasePublishArtifactsAction

object Publish {
  val nexus = "https://oss.sonatype.org/"

  def getPublishTo(snapshot: Boolean) = {
    if (snapshot) {
      Some("Snapshots" at nexus + "content/repositories/snapshots")
    } else {
      Some("Releases" at nexus + "service/local/staging/deploy/maven2")
    }
  }

  lazy val settings = Seq(
    homepage := Some(url("https://github.com/stripe/agate")),
    publishMavenStyle := true,
    publishArtifact in Test := false,
    pomIncludeRepository := Function.const(false),
    publishTo := getPublishTo(isSnapshot.value),
    publishArtifact in Test := false,
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
