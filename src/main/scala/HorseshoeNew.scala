import java.io.{BufferedWriter, File, FileWriter}
import breeze.linalg.{*, DenseMatrix, DenseVector, csvread}
import breeze.stats.mean
import com.stripe.rainier.compute._
import com.stripe.rainier.core.{Normal, RandomVariable, _}
import com.stripe.rainier.sampler._
import scala.collection.immutable.ListMap
import scala.annotation.tailrec


object HorseshoeNew {
  def main(args: Array[String]): Unit = {
    val rng = ScalaRNG(3)
    val (data, n1, n2) = dataProcessing()
    mainEffectsAndInters(data, rng, n1, n2)
  }

  /**
    * Process data read from input file
    */
  def dataProcessing(): (Map[(Int, Int), List[Double]], Int, Int) = {
    val data = csvread(new File("./SimulatedDataAndTrueCoefs/simulDataWithInters.csv"))
    val sampleSize = data.rows
    val y = data(::, 0).toArray
    val alpha = data(::, 1).map(_.toInt)
    val beta = data(::, 2).map(_.toInt)
    val nj = alpha.toArray.distinct.length //the number of levels for the first variable
    val nk = beta.toArray.distinct.length //the number of levels for the second variable
    val l = alpha.length
    var dataList = List[(Int, Int)]()

    for (i <- 0 until l) {
      dataList = dataList :+ (alpha(i), beta(i))
    }

    val dataMap = (dataList zip y).groupBy(_._1).map { case (k, v) => ((k._1 - 1, k._2 - 1), v.map(_._2)) }
    (dataMap, nj, nk)
  }

  /**
    * Use Rainier for modelling the main effects only, without interactions
    */
  def mainEffectsAndInters(dataMap: Map[(Int, Int), List[Double]], rngS: ScalaRNG, n1: Int, n2: Int): Unit = {
    implicit val rng = rngS
    val n = dataMap.size //No of groups

    // Implementation of sqrt for Real
    def sqrtF(x: Real): Real = {
      val lx = (Real(0.5) * x.log).exp
      lx
    }

    def updatePrior(mu: Real, sdE1: Real, sdE2: Real, sdG: Real, sdDR: Real): scala.collection.mutable.Map[String, Map[(Int, Int), Real]] = {
      val myMap = scala.collection.mutable.Map[String, Map[(Int, Int), Real]]()

      myMap("mu") = Map((0, 0) -> mu)
      myMap("eff1") = Map[(Int, Int), Real]()
      myMap("eff2") = Map[(Int, Int), Real]()
      myMap("effg") = Map[(Int, Int), Real]()
      myMap("sigE1") = Map((0, 0) -> sdE1)
      myMap("sigE2") = Map((0, 0) -> sdE2)
      myMap("sigInter") = Map((0, 0) -> sdG)
      myMap("sigD") = Map((0, 0) -> sdDR)

      myMap
    }

    //Define the prior
    //For jags we had: mu~dnorm(0,0.0001) and jags uses precision, so here we use sd = sqrt(1/tau)
    val prior = for {
      mu <- Normal(0, 100).param
      // Sample tau, estimate sd to be used in sampling from Normal the effects for the 1st variable
      tauE1RV = Gamma(1, 10000).param //RandomVariable[Real]
      tauE1 <- tauE1RV //Real
      sdE1 = sqrtF(Real(1.0) / tauE1) //Real. Without Real() it is Double

      // Sample tau, estimate sd to be used in sampling from Normal the effects for the 2nd variable
      tauE2RV = Gamma(1, 10000).param
      tauE2 <- tauE2RV
      sdE2 = sqrtF(Real(1.0) / tauE2)

      // Sample tau, estimate sd to be used in sampling from Normal the interaction effects
      tauGRV = Gamma(1, 10000).param
      tauG <- tauGRV
      sdG = sqrtF(Real(1.0) / tauG)

      // Sample tau, estimate sd to be used in sampling from Normal for fitting the model
      tauDRV = Gamma(1, 10000).param
      tauD <- tauDRV
      sdDR = sqrtF(Real(1.0) / tauD)
      //scala.collection.mutable.Map("mu" -> Map((0, 0) -> mu), "eff1" -> Map[(Int, Int), Real](), "eff2" -> Map[(Int, Int), Real](), "effg" -> Map[(Int, Int), Real](), "sigE1" -> Map((0, 0) -> sdE1), "sigE2" -> Map((0, 0) -> sdE2), "sigInter" -> Map((0, 0) -> sdG), "sigD" -> Map((0, 0) -> sdDR))
    } yield updatePrior(mu, sdE1, sdE2, sdG, sdDR)

    /**
      * Helper function to update the values for the main effects of the Map
      */
    def updateMap(myMap: scala.collection.mutable.Map[String, Map[(Int, Int), Real]], index: Int, key: String, addedValue: Real): scala.collection.mutable.Map[String, Map[(Int, Int), Real]] = {
      myMap(key) += ((0, index) -> addedValue)
      myMap
    }

    /**
      * Add the main effects of alpha
      */
    def addAplha(current: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]], i: Int): RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = {

      for {
        cur <- current
        gm_1 <- Normal(0, cur("sigE1")(0, 0)).param
      } yield updateMap(cur, i, "eff1", gm_1)
    }

    /**
      * Add the main effects of beta
      */
    def addBeta(current: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]], j: Int): RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = {
      for {
        cur <- current
        gm_2 <- Normal(0, cur("sigE2")(0, 0)).param
      } yield updateMap(cur, j, "eff2", gm_2)
    }

    /**
      * Helper function to update the values for the interaction effects of the Map
      */
    def updateMapGammaEff(myMap: scala.collection.mutable.Map[String, Map[(Int, Int), Real]], i: Int, j: Int, key: String, addedValue: Real): scala.collection.mutable.Map[String, Map[(Int, Int), Real]] = {
      myMap(key) += ((i, j) -> addedValue)
      myMap
    }

    /**
      * Add the interaction effects of beta
      */
    def addGammaEff(current: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]], i: Int, j: Int): RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = {
      for {
        cur <- current
        gm_inter <- Normal(0, cur("sigInter")(0, 0)).param
        //yield Map("mu" -> cur("mu"), "eff1" -> cur("eff1"), "eff2" -> (cur("eff2") += ((0, j) -> gm_2)), "sdE1" -> cur("sdE1"), "sdE2" -> cur("sdE2"), "sdD" -> cur("sdDR"))
      } yield updateMapGammaEff(cur, i, j, "effg", gm_inter)
    }

    /**
      * Add the interaction effects
      */
    def addGamma(current: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]]): RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = {
      var tempAlphaBeta: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = current

      for (i <- 0 until n1) {
        for (j <- 0 until n2) {
          tempAlphaBeta = addGammaEff(tempAlphaBeta, i, j)
        }
      }
      tempAlphaBeta
    }

    /**
      * Fit to the data per group
      */
    def addGroup(current: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]], i: Int, j: Int, dataMap: Map[(Int, Int), List[Double]]): RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = {
      for {
        cur <- current
        gm = cur("mu")(0, 0) + cur("eff1")(0, i) + cur("eff2")(0, j) + cur("effg")(i, j)
        _ <- Normal(gm, cur("sigD")(0, 0)).fit(dataMap(i, j))
      } yield cur
    }

    /**
      * Add all the main effects for each group. Version: Recursion
      */
    @tailrec def addAllEffRecursive(alphabeta: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]], dataMap: Map[(Int, Int), List[Double]], i: Int, j: Int): RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = {

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
    def addAllEffLoop(alphabeta: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]], dataMap: Map[(Int, Int), List[Double]]): RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = {

      var tempAlphaBeta: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = alphabeta

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
    def fullModel(alphabeta: RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]], dataMap: Map[(Int, Int), List[Double]]): RandomVariable[scala.collection.mutable.Map[String, Map[(Int, Int), Real]]] = {
      addAllEffRecursive(alphabeta, dataMap, 0, 0)
    }

    // Add the effects sequentially
    val alpha = (0 until n1).foldLeft(prior)(addAplha(_, _))
    val alphabeta = (0 until n2).foldLeft(alpha)(addBeta(_, _))
    val alphabetagamma = addGamma(alphabeta)
    val fullModelRes = fullModel(alphabetagamma, dataMap)

    val model = for {
      mod <- fullModelRes
    } yield Map("mu" -> mod("mu"),
      "eff1" -> mod("eff1"),
      "eff2" -> mod("eff2"),
      "effg" -> mod("effg"),
      "sigE1" -> mod("sigE1"),
      "sigE2" -> mod("sigE2"),
      "sigInter" -> mod("sigInter"),
      "sigD" -> mod("sigD"))

    // Calculation of the execution time
    def time[A](f: => A): A = {
      val s = System.nanoTime
      val ret = f
      val execTime = (System.nanoTime - s) / 1e6
      println("time: " + execTime + "ms")
      val bw = new BufferedWriter(new FileWriter(new File("./SimulatedDataAndTrueCoefs/results/RainierResWithInterHMC300-1mTime.txt")))
      bw.write(execTime.toString)
      bw.close()
      ret
    }

    // Sampling
    println("Model built. Sampling now (will take a long time)...")
    val thin = 100
    val out = model.sample(HMC(300), 1000, 10000 * thin, thin)
    println("Sampling finished.")
    printResults(out)

  }

  /**
    * Takes the result of the sampling and processes and prints the results
    */
  def printResults(out: scala.List[Map[String, Map[(Int, Int), Double]]]) = {

    def variableDM(varName: String):  DenseMatrix[Double] ={

      // Separate the data for the specific variable of interest
      val sepVariableData = out
        .flatMap{ eff1ListItem => eff1ListItem(varName) }
        .groupBy(_._1)
        .map { case (k, v) => k -> v.map(_._2) }

      // If the map contains more than one keys, they need to be sorted out to express the effects sequentially.
      // This is necessary for comparing the results from Scala and R in R
      val tempData= {
        varName match {
          case "mu" | "tau" | "sigInter" | "sigE1" | "sigE2" | "sigD" => sepVariableData
          case "eff1" | "eff2" => ListMap(sepVariableData.toSeq.sortBy(_._1._2):_*)
          case "effg" => ListMap(sepVariableData.toSeq.sortBy(_._1._2).sortBy(_._1._1):_*)
        }
      }
      val tempList = tempData.map{case (k,listDb) => (listDb)}.toList
      DenseMatrix(tempList.map(_.toArray): _*).t
    }
    println("----------------mu ------------------")
    val muMat = variableDM("mu")
    println(mean(muMat(::,*)))

    println("----------------eff1 ------------------")
    val effects1Mat = variableDM("eff1")
    println(mean(effects1Mat(::,*)))

    println("----------------eff2 ------------------")
    val effects2Mat = variableDM("eff2")
    println(mean(effects2Mat(::,*)))

    println("----------------effg (1,1),(2,1), (3,1) etc... ------------------")
    val effgMat = variableDM("effg")
    println(mean(effgMat(::,*)))

    println("----------------sigInter ------------------")
    val sigInterMat = variableDM("sigInter")
    println(mean(sigInterMat(::,*)))

    println("----------------sigΕ1 ------------------")
    val sigE1Mat = variableDM("sigE1")
    println(mean(sigE1Mat(::,*)))

    println("----------------sigΕ2 ------------------")
    val sigE2Mat = variableDM("sigE2")
    println(mean(sigE2Mat(::,*)))

    println("----------------sigD ------------------")
    val sigDMat = variableDM("sigD")
    println(mean(sigDMat(::,*)))

    val results = DenseMatrix.horzcat(effects1Mat, effects2Mat, effgMat, muMat, sigDMat, sigE1Mat, sigE2Mat, sigInterMat)

    val outputFile = new File("./SimulatedDataAndTrueCoefs/results/RainierResWithInterHMC300-1m.csv")
    breeze.linalg.csvwrite(outputFile, results, separator = ',')

  }
}
