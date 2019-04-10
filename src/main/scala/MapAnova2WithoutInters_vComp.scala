import java.io.File
import breeze.linalg._
import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._
import scala.annotation.tailrec

object MapAnova2WithoutInters_vComp {

  def main(args: Array[String]): Unit = {
    val rng = ScalaRNG(3)
    val (data, n1, n2) = dataProcessing()
    mainEffects(data, rng, n1, n2)
  }

  /**
    * Process data read from input file
    */
  def dataProcessing(): (Map[(Int,Int), Seq[Double]], Int, Int) = {
    val data = csvread(new File("/home/antonia/ResultsFromBessel/CompareRainier/simulNoInter.csv"))
    val sampleSize = data.rows
    val y = data(::, 0).toArray
    val alpha = data(::, 1).map(_.toInt)
    val beta = data(::, 2).map(_.toInt)
    val nj = alpha.toArray.distinct.length //the number of levels for the first variable
    val nk = beta.toArray.distinct.length //the number of levels for the second variable
    val l= alpha.length
    var dataList = Seq[(Int, Int)]()

    for (i <- 0 until l){
      dataList = dataList :+ (alpha(i),beta(i))
    }

    val dataMap = (dataList zip y).groupBy(_._1).map{ case (k,v) => ((k._1-1,k._2-1) ,v.map(_._2))}
    (dataMap, nj, nk)
  }

  /**
    * Use Rainier for modelling the main effects only, without interactions
    */
  def mainEffects(dataMap: Map[(Int, Int), Seq[Double]], rngS: ScalaRNG, n1: Int, n2: Int): Unit = {

    implicit val rng = rngS
    val n = dataMap.size //No of groups

    val prior = for {
      mu <- Normal(0, 0.0001).param
      sigE1 <- Gamma(1, 0.0001).param
      sigE2 <- Gamma(1, 0.0001).param
      sigD <- Gamma(1, 0.0001).param
      eff11= List.fill(n1) { Normal(0, sigE1).param.value }
      eff22 = List.fill(n2) { Normal(0, sigE2).param.value }
    } yield Map("mu" -> List(mu), "eff1" -> eff11, "eff2" -> eff22, "sigE1" -> List(sigE1), "sigE2" -> List(sigE2), "sigD" -> List(sigD))

    //  /**
    //    * Add the main effects of alpha
    //    */
    //  def addAplha(current: RandomVariable[Map[String, List[Real]]], i: Int): RandomVariable[Map[String, List[Real]]] = {
    //    for {
    //      cur <- current
    //      gm_1 <- Normal(0, cur("sigE1")(0)).param
    //      newEff1 = gm_1::cur("eff1")
    //    } yield  Map("mu" -> current("mu"), "eff1" -> newEff1, "eff2" -> current("eff2"), "sigE1" -> current("sigE1"), "sigE2" -> current("sigE2"), "sigD" -> current("sigD"))
    //  }
    //
    //    /**
    //    * Add the main effects of beta
    //    */
    //    def addBeta(current: RandomVariable[Map[String, List[Real]]], i: Int): RandomVariable[Map[String, List[Real]]] = {
    //      for {
    //        cur <- current
    //        gm_2 <- Normal(0, cur("sigE2")(0)).param
    //        newEff2 = gm_2::cur("eff1")
    //      } yield  Map("mu" -> current("mu"), "eff1" -> current("eff1"), "eff2" -> newEff2, "sigE1" -> current("sigE1"), "sigE2" -> current("sigE2"), "sigD" -> current("sigD"))
    //    }

    /**
      * Fit to the data per group
      */
    def addGroup(current: RandomVariable[Map[String, List[Real]]], i: Int, j: Int, dataMap: Map[(Int, Int), Seq[Double]]): RandomVariable[Map[String, List[Real]]] = {
      for {
        cur <- current
        gm = cur("mu")(0) + cur("eff1")(i) + cur("eff2")(j)
        _ <- Normal(gm, cur("sigD")(0)).fit(dataMap(i, j))
      } yield cur
    }

    /**
      * Add all the main effects for each group. Version: Recursion
      */
    @tailrec def addAllEffRecursive(alphabeta: RandomVariable[Map[String, List[Real]]], dataMap: Map[(Int, Int), Seq[Double]], i: Int, j: Int): RandomVariable[Map[String, List[Real]]] = {
      val temp = addGroup(alphabeta, i, j, dataMap)

      if (i == n1 - 1 && j == n2 - 1) {
        temp
      } else {
        val nextJ = if (j < n2 - 1) j + 1 else 0
        val nextI = if (j < n2 - 1) i else i + 1
        addAllEffRecursive(temp, dataMap, nextI, nextJ)
      }
    }

    /**
      * Add all the main effects for each group. Version: Loop
      */
    def addAllEffLoop(alphabeta: RandomVariable[Map[String, List[Real]]], dataMap: Map[(Int, Int), Seq[Double] ]) : RandomVariable[Map[String, List[Real]]] = {
      var tempAlphaBeta: RandomVariable[Map[String, List[Real]]] = alphabeta

      for (i <- 0 until n1) {
        for (j <- 0 until n2) {
          tempAlphaBeta = addGroup(tempAlphaBeta, i, j, dataMap)
        }
      }
      tempAlphaBeta
    }

    /**
      * Add all the main effects for each group.
      * We would like something like: val result = (0 until n1)(0 until n2).foldLeft(alphabeta)(addGroup(_,(_,_)))
      * But we can't use foldLeft with double loop so we use an extra function either with loop or recursively
      */
    def fullModel(alphabeta: RandomVariable[Map[String, List[Real]]], dataMap: Map[(Int, Int), Seq[Double]]) : RandomVariable[Map[String, List[Real]]] = {
      //addAllEffLoop(alphabeta, dataMap)
      addAllEffRecursive(alphabeta, dataMap, 0, 0)
    }

    //    val alpha = (0 until n1).foldLeft(prior)(addAplha(_, _))
    //    val alphabeta = (0 until n2).foldLeft(alpha)(addBeta(_, _))

    val fullModelRes = fullModel(prior, dataMap)

    val model = for {
      mod <- fullModelRes
    } yield Map("mu" -> mod("mu"),
      "eff1" -> mod("eff1"),
      "eff2" -> mod("eff2"),
      "sigE1" -> mod("sigE1"),
      "sigE2" -> mod("sigE2"),
      "sigD" -> mod("sigD"))

    // sampling
    println("Model built. Sampling now (will take a long time)...")
    val thin = 200
    val out = model.sample(HMC(5), 10000, 10 * thin, thin)
    println("Sampling finished.")

    def printResults(out: List[Map[String, Seq[Double]]]) = {

      def flattenSigMu(vars: String, grouped: Map[String, List[Seq[Double]]]):  Iterable[Double] ={
        grouped.filter((t) => t._1 == vars).map { case (k, v) => v }.flatten.flatten
      }

      def flattenEffects(vars: String, grouped: Map[String, List[Seq[Double]]]):  Iterable[Seq[Double]] ={
        grouped.filter((t) => t._1 == vars).map { case (k, v) => v }.flatten
      }

      //Separate the parameters
      val grouped = out.flatten.groupBy(_._1).mapValues(_.map(_._2))
      val effects1 = flattenEffects("eff1", grouped)
      val effects2 = flattenEffects("eff2", grouped)
      val sigE1 = flattenSigMu("sigE1", grouped)
      val sigE2 = flattenSigMu("sigE2", grouped)
      val sigD = flattenSigMu("sigD", grouped)
      val mu = flattenSigMu("mu", grouped)

      // Find the averages
      def mean(list: Iterable[Double]): Double =
        if (list.isEmpty) 0 else list.sum / list.size

      val sigE1A = mean(sigE1)
      val sigE2A = mean(sigE2)
      val sigDA = mean(sigD)
      val muA = mean(mu)
      val eff1A = effects1.transpose.map(x => x.sum / x.size.toDouble)
      val eff2A = effects2.transpose.map(x => x.sum / x.size.toDouble)

      //Print the average
      println(s"sigE1: ", sigE1A)
      println(s"sigE2: ", sigE2A)
      println(s"sigD: ", sigDA)
      println(s"mu: ", muA)
      println(s"effects1: ", eff1A)
      println(s"effects2: ", eff2A)
    }
    printResults(out)
  }

}
