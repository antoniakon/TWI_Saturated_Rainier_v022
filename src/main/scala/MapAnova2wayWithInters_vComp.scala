import java.io.File

import breeze.linalg.csvread
import com.stripe.rainier.compute._
import com.stripe.rainier.core.{Normal, _}
import com.stripe.rainier.sampler._

import scala.annotation.tailrec

object MapAnova2wayWithInters_vComp {

   def main(args: Array[String]): Unit = {
     val rng = ScalaRNG(3)
     val (data, n1, n2) = dataProcessing()
     mainEffectsAndInters(data, rng, n1, n2)
    }

    /**
      * Process data read from input file
      */
    def dataProcessing(): (Map[(Int,Int), Seq[Double]], Int, Int) = {
      val data = csvread(new File("/home/antonia/ResultsFromBessel/CompareRainier/simulInter.csv"))
      val sampleSize = data.rows
      val y = data(::, 0).toArray
      val alpha = data(::, 1).map(_.toInt)
      val beta = data(::, 2).map(_.toInt)
      val nj = alpha.toArray.distinct.length //the number of levels for the first variable
      val nk = beta.toArray.distinct.length //the number of levels for the second variable
      //val tryMap = y.map(obs => alpha.map(a=> beta.map(b=>((a,b),y))))
      val l= alpha.length
      var dataList = Seq[(Int, Int)]()

      for (i <- 0 until l){
        dataList = dataList :+ (alpha(i),beta(i))
      }

      val dataMap = (dataList zip y).groupBy(_._1).map{ case (k,v) => ((k._1-1,k._2-1) ,v.map(_._2))}
      (dataMap, nj, nk)
    }


  // build and fit model
    //case class MyStructure(mu: Real, eff1: List[Real], eff2: List[Real], effg: Vector[Vector[Real]], sigE1: Real, sigE2: Real, sigEg: Real, sigD: Real)

    private val prior = for {
      mu <- Normal(5, 10).param
      sigE1 <- LogNormal(1, 0.2).param
      sigE2 <- LogNormal(1, 0.2).param
      sigEg <- LogNormal(1, 0.2).param
      sigD <- LogNormal(1, 4).param
      eff11= Vector.fill(1){Vector.fill(5) { Normal(0, sigE1).param.value }}
      eff22 = Vector.fill(1){Vector.fill(10) { Normal(0, sigE2).param.value }}
      effinter = Vector.fill(5) { Vector.fill(10) { Normal(0, sigEg).param.value } }
    } yield Map("mu" -> Vector.fill(1,1)(mu), "eff1" -> eff11, "eff2" -> eff22, "effg" -> effinter, "sigE1" -> Vector.fill(1,1)(sigE1), "sigE2" -> Vector.fill(1,1)(sigE2), "sigInter" ->Vector.fill(1,1)(sigEg),  "sigD" -> Vector.fill(1,1)(sigD))

    /**
      * Fit to the data per group
      */
    def addGroup(current: RandomVariable[Map[String, Vector[Vector[Real]]]], i: Int, j: Int, dataMap: Map[(Int, Int), Seq[Double]]): RandomVariable[Map[String, Vector[Vector[Real]]]] = {
      for {
        cur <- current
        gm = cur("mu")(0)(0) + cur("eff1")(0)(i) + cur("eff2")(0)(j) + cur("effg")(i)(j)
        _ <- Normal(gm, cur("sigD")(0)(0)).fit(dataMap(i, j))
      } yield cur

    }


    /**
      * Add all the main effects for each group. Version: Recursion
      */
    @tailrec def addAllEffRecursive(alphabeta: RandomVariable[Map[String, Vector[Vector[Real]]]], dataMap: Map[(Int, Int), Seq[Double]], i: Int, j: Int): RandomVariable[Map[String, Vector[Vector[Real]]]] = {

      val n1 = 5
      val n2 = 10

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
    def addAllEffLoop(alphabeta: RandomVariable[Map[String, Vector[Vector[Real]]]], dataMap: Map[(Int, Int), Seq[Double] ]) : RandomVariable[Map[String, Vector[Vector[Real]]]] = {
      val n1 = 5
      val n2 = 10

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
    def fullModel(alphabeta: RandomVariable[Map[String, Vector[Vector[Real]]]], dataMap: Map[(Int, Int), Seq[Double]]) : RandomVariable[Map[String, Vector[Vector[Real]]]] = {
      //addAllEffLoop(alphabeta, dataMap)
      addAllEffRecursive(alphabeta, dataMap, 0, 0)
    }

    /**
      * Use Rainier for modelling the main effects only, without interactions
      */
    def mainEffectsAndInters(dataMap: Map[(Int, Int), Seq[Double]], rngS: ScalaRNG, n1: Int, n2: Int): Unit = {
      implicit val rng = rngS
      val n = dataMap.size //No of groups

      //val alpha = (0 until n1).foldLeft(prior)(addAplha(_, _))

      //val alphabeta = (0 until n2).foldLeft(alpha)(addBeta(_, _))

      val fullModelRes = fullModel(prior, dataMap)

      //sealed abstract class Thing
      //case class obj(ob: Real ) extends Thing
      //case class vec(ve: List[Real]) extends Thing

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

      // sampling
      println("Model built. Sampling now (will take a long time)...")
      val thin = 200
      val out = model.sample(HMC(5), 100, 100 * thin, thin)


      //println(grouped)
      println(out)
      println("Sampling finished.")

      //Separate the parameters
      val grouped = out.flatten.groupBy(_._1).mapValues(_.map(_._2))
      val effects1 = grouped.filter((t) => t._1 =="eff1").map{case (k,v) => v}.flatten.flatten
      val effects2 = grouped.filter((t) => t._1 =="eff2").map{case (k,v) => v}.flatten.flatten
      val sigE1 = grouped.filter((t) => t._1 =="sigE1").map{case (k,v) => v}.flatten.flatten.flatten
      val sigE2 = grouped.filter((t) => t._1 =="sigE2").map{case (k,v) => v}.flatten.flatten.flatten
      val sigD = grouped.filter((t) => t._1 =="sigD").map{case (k,v) => v}.flatten.flatten.flatten
      val mu = grouped.filter((t) => t._1 =="mu").map{case (k,v) => v}.flatten.flatten.flatten

      // Find the averages
      def mean(list:Iterable[Double]):Double =
        if(list.isEmpty) 0 else list.sum/list.size

      val sigE1A = mean(sigE1)
      val sigE2A = mean(sigE2)
      val sigDA = mean(sigD)
      val muA = mean(mu)
      val eff1A = effects1.transpose.map(x => x.sum/x.size.toDouble)
      val eff2A = effects2.transpose.map(x => x.sum/x.size.toDouble)

      //    val avg = grouped.map(l => (l._1, l._2.sum / out.length))

      //Print the average
      println(s"sigE1: ", sigE1A)
      println(s"sigE2: ", sigE2A)
      println(s"sigD: ", sigDA)
      println(s"mu: ", muA)
      println(s"effects1: ", eff1A)
      println(s"effects2: ", eff2A)

    }
}
