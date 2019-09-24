import java.io.File

import breeze.linalg._
import breeze.numerics.sqrt
import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._

import scala.annotation.tailrec
import com.stripe.rainier.ir
import com.cibo.evilplot.geometry.Extent
import com.cibo.evilplot.plot.LinePlot
import com.stripe.rainier.plot.EvilTracePlot._
import com.cibo.evilplot.plot.aesthetics.DefaultTheme._
import com.cibo.evilplot.numeric.Point
import com.cibo.evilplot._
import com.cibo.evilplot.plot._
import com.cibo.evilplot.plot.aesthetics.DefaultTheme._
import com.cibo.evilplot.numeric.Point
import com.cibo.evilplot.plot._
import com.cibo.evilplot.colors._
import com.cibo.evilplot.plot.aesthetics.DefaultTheme._
import com.cibo.evilplot.numeric.Point
import java.awt.Image.SCALE_SMOOTH

object MapAnova2WithoutInters_vComp {

  def main(args: Array[String]): Unit = {
    val rng = ScalaRNG(3)
    val (data, n1, n2) = dataProcessing()
    mainEffects(data, rng, n1, n2)
  }

  /**
    * Process data read from input file
    */
  def dataProcessing(): (Map[(Int,Int), List[Double]], Int, Int) = {
    val data = csvread(new File("/home/antonia/ResultsFromCloud/CompareRainier/040619/withoutInteractions/simulNoInter040619.csv"))
    val sampleSize = data.rows
    val y = data(::, 0).toArray
    val alpha = data(::, 1).map(_.toInt)
    val beta = data(::, 2).map(_.toInt)
    val nj = alpha.toArray.distinct.length //the number of levels for the first variable
    val nk = beta.toArray.distinct.length //the number of levels for the second variable
    val l= alpha.length // the number of observations
    var dataList = List[(Int, Int)]()

    for (i <- 0 until l){
      dataList = dataList :+ (alpha(i),beta(i))
    }
    //println(dataList)

    val dataMap = (dataList zip y).groupBy(_._1).map{ case (k,v) => ((k._1-1,k._2-1) ,v.map(_._2))} //Bring the data to the map format
    //println(dataMap)
    (dataMap, nj, nk)

  }

  /**
    * Use Rainier for modelling the main effects only, without interactions
    */
  def mainEffects(dataMap: Map[(Int, Int), List[Double]], rngS: ScalaRNG, n1: Int, n2: Int): Unit = {

    // Implementation of sqrt for Real
    def sqrtR(x: Real): Real = {
      val lx = (Real(0.5) * x.log).exp
      lx
    }

    implicit val rng = rngS
    val n = dataMap.size //No of groups
    // All prior values for the unknown parameters, defined as follows, are stored in lists, to be able to process and print the results at the end.
    val prior = for {
      mu <- Normal(0, 100).param

      // Sample tau, estimate sd to be used in sampling from Normal the effects for the 1st variable
      tauE1RV = Gamma(1, 10000).param
      tauE1 <- tauE1RV
      sdE1= sqrtR(Real(1.0)/tauE1)

      // Sample tau, estimate sd to be used in sampling from Normal the effects for the 2nd variable
      tauE2RV = Gamma(1, 10000).param
      tauE2 <- tauE2RV
      sdE2LD = tauE2RV.sample(1)
      sdE2= sqrtR(Real(1.0)/tauE2)

      // Sample tau, estimate sd to be used in sampling from Normal for fitting the model
      tauDRV = Gamma(1, 10000).param
      tauD <- tauDRV
      sdDR= sqrtR(Real(1.0)/tauD)

    } yield Map("mu" -> List(mu), "eff1" -> List[Real]() , "eff2" -> List[Real](), "sdE1" -> List(sdE1), "sdE2" -> List(sdE2), "sdD" -> List(sdDR))

    /**
      * Add the main effects of alpha
      */
    def addAplha(current: RandomVariable[Map[String, List[Real]]], i: Int): RandomVariable[Map[String, List[Real]]] = {
      for {
        cur<- current
        gm_1 <- Normal(0, cur("sdE1")(0)).param
      } yield Map("mu" -> cur("mu"), "eff1" -> (gm_1::cur("eff1")) , "eff2" -> cur("eff2"), "sdE1" -> cur("sdE1"), "sdE2" -> cur("sdE2"), "sdD" -> cur("sdD"))
    }

    /**
      * Add the main effects of beta
      */
    def addBeta(current: RandomVariable[Map[String, List[Real]]], j: Int): RandomVariable[Map[String, List[Real]]] = {
      for {
        cur <- current
        gm_2 <- Normal(0, cur("sdE2")(0)).param
      } yield Map("mu" -> cur("mu"), "eff1" -> cur("eff1") , "eff2" -> (gm_2::cur("eff2")), "sdE1" -> cur("sdE1"), "sdE2" -> cur("sdE2"), "sdD" -> cur("sdD"))
    }
    /**
      * Fit to the data per group
      */
    def addGroup(current: RandomVariable[Map[String, List[Real]]], i: Int, j: Int, dataMap: Map[(Int, Int), List[Double]]): RandomVariable[Map[String, List[Real]]] = {
      for {
        cur <- current
        tauDR = cur("sdD")(0)
        gm = cur("mu")(0) + cur("eff1")(i) + cur("eff2")(j)
        _ <- Normal(gm, tauDR).fit(dataMap(i, j))
      } yield cur
    }

    /**
      * Add all the main effects for each group. Version: Recursion
      */
    @tailrec def addAllEffRecursive(alphabeta: RandomVariable[Map[String, List[Real]]], dataMap: Map[(Int, Int), List[Double]], i: Int, j: Int): RandomVariable[Map[String, List[Real]]] = {
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
    def addAllEffLoop(alphabeta: RandomVariable[Map[String, List[Real]]], dataMap: Map[(Int, Int), List[Double] ]) : RandomVariable[Map[String, List[Real]]] = {
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
    def fullModel(alphabeta: RandomVariable[Map[String, List[Real]]], dataMap: Map[(Int, Int), List[Double]]) : RandomVariable[Map[String, List[Real]]] = {
      //addAllEffLoop(alphabeta, dataMap)
      addAllEffRecursive(alphabeta, dataMap, 0, 0)
    }
    val alpha = (0 until n1).foldLeft(prior)(addAplha(_, _))

    val alphabeta = (0 until n2).foldLeft(alpha)(addBeta(_, _))

    val fullModelRes = fullModel(alphabeta, dataMap)

    val model = for {
      mod <- fullModelRes
    } yield Map("mu" -> mod("mu"),
      "eff1" -> mod("eff1"),
      "eff2" -> mod("eff2"),
      "sdE1" -> mod("sdE1"),
      "sdE2" -> mod("sdE2"),
      "sdD" -> mod("sdD"))

    // Sampling
    println("Model built. Sampling now (will take a long time)...")
    val thin = 100
    val out = model.sample(HMC(200), 1000, 10000 * thin, thin)
    println("Sampling finished.")

    def printResults(out: List[Map[String, List[Double]]]) = {

      def flattenSigMu(vars: String, grouped: Map[String, List[List[Double]]]):  List[Double] ={
        grouped.filter((t) => t._1 == vars).map { case (k, v) => v }.flatten.flatten.toList
      }

      def flattenEffects(vars: String, grouped: Map[String, List[List[Double]]]):  List[List[Double]] ={
        grouped.filter((t) => t._1 == vars).map { case (k, v) => v }.flatten.toList
      }

      //Separate the parameters
      val grouped = out.flatten.groupBy(_._1).mapValues(_.map(_._2))
      val effects1 = flattenEffects("eff1", grouped)
      val effects2 = flattenEffects("eff2", grouped)
      val sigE1 = flattenSigMu("sdE1", grouped)
      val sigE2 = flattenSigMu("sdE2", grouped)
      val sigD = flattenSigMu("sdD", grouped)
      val mu = flattenSigMu("mu", grouped)

      //Save results to csv
      val effects1Mat = DenseMatrix(effects1.map(_.toArray):_*) //make it a denseMatrix to concatenate later
      val effects2Mat = DenseMatrix(effects2.map(_.toArray):_*) //make it a denseMatrix to concatenate later
      val sigE1dv = new DenseVector[Double](sigE1.toArray)
      val sigE2dv = new DenseVector[Double](sigE2.toArray)
      val sigDdv = new DenseVector[Double](sigD.toArray)
      val mudv = new DenseVector[Double](mu.toArray)
      val sigmasMu = DenseMatrix(sigE1dv, sigE2dv, sigDdv, mudv)
      val results = DenseMatrix.horzcat(effects1Mat, effects2Mat, sigmasMu.t)

      val outputFile = new File("/home/antonia/ResultsFromCloud/CompareRainier/040619/withoutInteractions/FullResultsRainierWithoutInterHMC200New.csv")
      breeze.linalg.csvwrite(outputFile, results, separator = ',')

      val mudata = Seq.tabulate(mudv.length) { i =>
        Point(i.toDouble, mudv(i))
      }


      val plot = LinePlot(mudata).
        xAxis().yAxis().frame().
        xLabel("x").yLabel("y").render()


      // Find the averages
      def mean(list: List[Double]): Double =
        if (list.isEmpty) 0 else list.sum / list.size

      val sigE1A = mean(sigE1)
      val sigE2A = mean(sigE2)
      val sigDA = mean(sigD)
      val muA = mean(mu)
      val eff1A = effects1.transpose.map(x => x.sum / x.size.toDouble)
      val eff2A = effects2.transpose.map(x => x.sum / x.size.toDouble)

      //Print the average
      println(s"tauE1: ", sigE1A)
      println(s"tauE2: ", sigE2A)
      println(s"sdD: ", sigDA)
      println(s"mu: ", muA)
      println(s"effects1: ", eff1A)
      println(s"effects2: ", eff2A)
    }
    val outList = out.map{ v=> v.map { case (k,v) => (k, v.toList)}}
    printResults(outList)
  }

}
