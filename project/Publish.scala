import sbt._
import sbt.Keys._
import com.typesafe.sbt.pgp.PgpKeys._
import aether.AetherKeys._

object Publish {
  val snapshotsUrl = "https://nexus-content-v2.northwest.corp.stripe.com:446/nexus/repository/maven-snapshots/"
  val releasesUrl = "https://nexus-content-v2.northwest.corp.stripe.com:446/nexus/repository/maven-releases/"

  def getPublishTo(snapshot: Boolean) = {
    if (snapshot) {
      val url = sys.props.get("stripe.snapshots.url").getOrElse(snapshotsUrl)
      Some("stripe-nexus-snapshots" at url)
    } else {
      val url = sys.props.get("stripe.releases.url").getOrElse(releasesUrl)
      Some("stripe-nexus-releases" at url)
    }
  }

  lazy val settings = Seq(
    homepage := Some(url("http://github.com/stripe-internal/agate")),
    publishMavenStyle := true,
    publish := aetherDeploy.value,
    publishTo := getPublishTo(isSnapshot.value),
    publishArtifact in Test := false,
    pomIncludeRepository := Function.const(false),
    pomExtra := (
      <scm>
        <url>git@github.com:stripe-internal/agate.git</url>
        <connection>scm:git:git@github.com:stripe-internal/agate.git</connection>
      </scm>
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
