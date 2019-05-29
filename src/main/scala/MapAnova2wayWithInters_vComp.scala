import java.io.File

import breeze.linalg.{DenseMatrix, DenseVector, csvread}
import com.stripe.rainier.compute._
import com.stripe.rainier.core.{Normal, _}
import com.stripe.rainier.sampler._

import scala.annotation.tailrec
import scala.math.sqrt

object MapAnova2wayWithInters_vComp {

  def main(args: Array[String]): Unit = {
    val rng = ScalaRNG(3)
    val (data, n1, n2) = dataProcessing()
    mainEffectsAndInters(data, rng, n1, n2)
  }

  /**
    * Process data read from input file
    */
  def dataProcessing(): (Map[(Int,Int), List[Double]], Int, Int) = {
    val data = csvread(new File("/home/antonia/ResultsFromCloud/CompareRainier/try/inter.csv"))
    val sampleSize = data.rows
    val y = data(::, 0).toArray
    val alpha = data(::, 1).map(_.toInt)
    val beta = data(::, 2).map(_.toInt)
    val nj = alpha.toArray.distinct.length //the number of levels for the first variable
    val nk = beta.toArray.distinct.length //the number of levels for the second variable
    val l= alpha.length
    var dataList = List[(Int, Int)]()

    for (i <- 0 until l){
      dataList = dataList :+ (alpha(i),beta(i))
    }

    val dataMap = (dataList zip y).groupBy(_._1).map{ case (k,v) => ((k._1-1,k._2-1) ,v.map(_._2))}
    (dataMap, nj, nk)
  }

  /**
    * Use Rainier for modelling the main effects only, without interactions
    */
  def mainEffectsAndInters(dataMap: Map[(Int, Int), List[Double]], rngS: ScalaRNG, n1: Int, n2: Int): Unit = {
    implicit val rng = rngS
    val n = dataMap.size //No of groups

    val prior = for {
      mu <- Normal(0, 0.0001).param
      // Sample tau, estimate sd to be used in sampling from Normal the effects for the 1st variable
      tauE1RV = Gamma(1, 10000).param
      tauE1 <- tauE1RV
      sdE1LD = tauE1RV.sample(1)
      sdE1= sqrt(1/sdE1LD(0))

      // Sample tau, estimate sd to be used in sampling from Normal the effects for the 2nd variable
      tauE2RV = Gamma(1, 10000).param
      tauE2 <- tauE2RV
      sdE2LD = tauE2RV.sample(1)
      sdE2= sqrt(1/sdE2LD(0))

      // Sample tau, estimate sd to be used in sampling from Normal the effects for the 2nd variable
      tauGRV = Gamma(1, 10000).param
      tauG <- tauGRV
      sdGLD = tauGRV.sample(1)
      sdG= sqrt(1/sdGLD(0))

      // Sample tau, estimate sd to be used in sampling from Normal for fitting the model
      tauDRV = Gamma(1, 10000).param
      tauD <- tauDRV
      sdDLD = tauDRV.sample(1)
      sdDR= Real(sqrt(1/sdDLD(0)))

      sigEg <- Gamma(1, 0.0001).param
      sigD <- Gamma(1, 0.0001).param
      eff11= Vector.fill(1){Vector.fill(n1) { Normal(0, sdE1).param.value }}
      eff22 = Vector.fill(1){Vector.fill(n2) { Normal(0, sdE2).param.value }}
      effinter = Vector.fill(n1) { Vector.fill(n2) { Normal(0, sdG).param.value } }
    } yield Map("mu" -> Vector.fill(1,1)(mu), "eff1" -> eff11, "eff2" -> eff22, "effg" -> effinter, "tauE1" -> Vector.fill(1,1)(tauE1), "tauE2" -> Vector.fill(1,1)(tauE2), "tauInter" ->Vector.fill(1,1)(tauG),  "sigD" -> Vector.fill(1,1)(sdDR))

    /**
      * Fit to the data per group
      */
    def addGroup(current: RandomVariable[Map[String, Vector[Vector[Real]]]], i: Int, j: Int, dataMap: Map[(Int, Int), List[Double]]): RandomVariable[Map[String, Vector[Vector[Real]]]] = {
      for {
        cur <- current
        gm = cur("mu")(0)(0) + cur("eff1")(0)(i) + cur("eff2")(0)(j) + cur("effg")(i)(j)
        _ <- Normal(gm, cur("sigD")(0)(0)).fit(dataMap(i, j))
      } yield cur
    }

    /**
      * Add all the main effects for each group. Version: Recursion
      */
    @tailrec def addAllEffRecursive(alphabeta: RandomVariable[Map[String, Vector[Vector[Real]]]], dataMap: Map[(Int, Int), List[Double]], i: Int, j: Int): RandomVariable[Map[String, Vector[Vector[Real]]]] = {

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
    def addAllEffLoop(alphabeta: RandomVariable[Map[String, Vector[Vector[Real]]]], dataMap: Map[(Int, Int), List[Double] ]) : RandomVariable[Map[String, Vector[Vector[Real]]]] = {

      var tempAlphaBeta: RandomVariable[Map[String, Vector[Vector[Real]]]] = alphabeta

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
    def fullModel(alphabeta: RandomVariable[Map[String, Vector[Vector[Real]]]], dataMap: Map[(Int, Int), List[Double]]) : RandomVariable[Map[String, Vector[Vector[Real]]]] = {
      //addAllEffLoop(alphabeta, dataMap)
      addAllEffRecursive(alphabeta, dataMap, 0, 0)
    }

    val fullModelRes = fullModel(prior, dataMap)

    val model = for {
      mod <- fullModelRes
    } yield Map("mu" -> mod("mu"),
      "eff1" -> mod("eff1"),
      "eff2" -> mod("eff2"),
      "effg" -> mod("effg"),
      "tauE1" -> mod("tauE1"),
      "tauE2" -> mod("tauE2"),
      "tauInter" -> mod("tauInter"),
      "sigD" -> mod("sigD"))

    // Sampling
    println("Model built. Sampling now (will take a long time)...")
    val thin = 10
    val out = model.sample(HMC(5), 10000, 10 * thin, thin)
    println("Sampling finished.")

    //Print the results
    val outList = out.map{ v=> v.map { case (k,v) => (k, v.toList.map(l => l.toList))}}
    printResults(outList)

    /**
      * Takes the result of the sampling and processes and prints the results
      */
    def printResults (out: List[Map[String, List[List[Double]]]] ) = {

      def flattenSigMu(vars: String, grouped:  Map[String, List[List[List[Double]]]]):  List[Double] ={
        grouped.filter((t) => t._1 == vars).map { case (k, v) => v }.flatten.flatten.flatten.toList
      }

      def flattenEffects(vars: String, grouped:  Map[String, List[List[List[Double]]]]):  List[List[Double]] ={
        grouped.filter((t) => t._1 == vars).map { case (k, v) => v }.flatten.flatten.toList
      }

      //Separate the parameters
      val grouped = out.flatten.groupBy(_._1).mapValues(_.map(_._2))
      val effects1 = flattenEffects("eff1", grouped)
      val effects2 = flattenEffects("eff2", grouped)
      val sigE1 = flattenSigMu("sigE1", grouped)
      val sigE2 = flattenSigMu("sigE2", grouped)
      val sigD = flattenSigMu("sigD", grouped)
      val sigInter = flattenSigMu("sigInter", grouped)
      val mu = flattenSigMu("mu", grouped)


      // Find the averages
      def mean(list: List[Double]): Double =
        if (list.isEmpty) 0 else list.sum / list.size

      val sigE1A = mean(sigE1)
      val sigE2A = mean(sigE2)
      val sigDA = mean(sigD)
      val sigInterA = mean(sigInter)
      val muA = mean(mu)
      val eff1A = effects1.transpose.map(x => x.sum / x.size.toDouble)
      val eff2A = effects2.transpose.map(x => x.sum / x.size.toDouble)

      //Interaction coefficients
      val effectsInter = grouped.filter((t) => t._1 == "effg").map { case (k, v) => v }.flatten
      val effsum = effectsInter.reduce((a, b) => a.zip(b).map { case (v1, v2) => v1.zip(v2).map { case (x, y) => (x + y) } }).map(z => z.map(z1 => z1 / effectsInter.size))

      val interEff = effectsInter.map(l1 => l1.reduce((a,b)=> a++b)).toList //List[List[Double]] the inner lists represent the iterations. This will have to be stored in a DenseMatrix


      //Print the average
      println(s"sigE1: ", sigE1A)
      println(s"sigE2: ", sigE2A)
      println(s"sigD: ", sigDA)
      println(s"sigInter: ", sigInterA)
      println(s"mu: ", muA)
      println(s"effects1: ", eff1A)
      println(s"effects2: ", eff2A)
      println(s"effects: ", effsum)

      //Save results to csv
      val effects1Mat = DenseMatrix(effects1.map(_.toArray):_*) //make it a denseMatrix to concatenate later
      val effects2Mat = DenseMatrix(effects2.map(_.toArray):_*) //make it a denseMatrix to concatenate later
      val interEffMat =  DenseMatrix(interEff.map(_.toArray):_*) //make it a denseMatrix to concatenate later
      val sigE1dv = new DenseVector[Double](sigE1.toArray)
      val sigE2dv = new DenseVector[Double](sigE2.toArray)
      val sigDdv = new DenseVector[Double](sigD.toArray)
      val sigInterdv = new DenseVector[Double](sigInter.toArray)
      val mudv = new DenseVector[Double](mu.toArray)
      val sigmasMu = DenseMatrix(sigE1dv, sigE2dv)
      println(sigE1dv.length)
      println(sigDdv.length)
      println(mudv.length)
//      val results = DenseMatrix.horzcat(effects1Mat, effects2Mat,  sigmasMu.t)
//      val outputFile = new File("/home/antonia/ResultsFromCloud/CompareRainier/FullResultsRainierWithInter.csv")
//      breeze.linalg.csvwrite(outputFile, results, separator = ',')


    }
  }
}
