import com.stripe.rainier.compute._
import com.stripe.rainier.core.{Normal, RandomVariable, _}
import com.stripe.rainier.sampler._

/**
  * Builds a simple 1-way Anova model in Scala using Rainier version 0.2.2
  */
object OneWayAnova {

  def anova: Unit = {
    implicit val rng = ScalaRNG(3)
    val n = 50 // groups
    val N = 250 // obs per group
    val mu1 = 5.0 // overall mean
    val sigE = 2.0 // random effect SD
    val sigD = 3.0 // obs SD
    val effects = Vector.fill(n)(sigE * rng.standardNormal)
    val data = effects map (e =>
      Vector.fill(N)(mu1 + e + sigD * rng.standardNormal))
    println(data)

    // Define parameters and priors
    val prior = for {
      mu <- Normal(0, 100).param // overall mean
      sdObs <- LogNormal(0, 10).param // observational sd
      sdEff <- LogNormal(1, 5).param // random effect sd
    } yield Map("mu" -> mu, "sdObs" -> sdObs, "sdEff" -> sdEff) // save to a Map

    // Update by observing the data per group
    def addEffect(current: Map[String, Real], i: Int): RandomVariable[Real] =
      for {
        gm <- Normal(0, current("sdEff")).param
        _ <- Normal(current("mu") + gm, current("sdObs")).fit(data(i))
      } yield gm

    // Apply to all groups
    val model = for {
      current <- prior
      _ <- RandomVariable.traverse((0 until n) map (addEffect(current, _)))
    } yield current

    // Fit the model
    val iterations = 10000
    val thin = 1
    val lfrogStep = 50
    val output = model.sample(HMC(lfrogStep), warmupIterations = 1000, iterations *
      thin, thin)
    println(output)
  }

  // Calculation of the execution time
  def time[A](f: => A): A = {
    val s = System.nanoTime
    val ret = f
    val execTime = (System.nanoTime - s) / 1e6
    println("time: " + execTime + "ms")
    ret
  }
  def main(args: Array[String]): Unit = {
    println("main starting")
    time(anova)
    println("main finishing")
  }
}
