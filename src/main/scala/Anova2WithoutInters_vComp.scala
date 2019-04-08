import java.io.File
import breeze.linalg._
import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._
import annotation.tailrec

object Anova2WithoutInters_vComp {

  def main(args: Array[String]): Unit = {
    val rng = ScalaRNG(3)
    val (data, n1, n2) = dataProcessing()
    println(n1)
    println(data)
    //mainEffects(dataMap, rng, n1, n2)

  }

  /**
    * Generate data with and without interactions
    */
  //def dataProcessing(): Map[(Int, Int), Vector[Double]] = {
  def dataProcessing(): (Map[(Int,Int), Seq[Double]], Int, Int) = {
    val data = csvread(new File("/home/antonia/ResultsFromBessel/CompareRainier/simulNoInter.csv"))
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

    val dataMap = (dataList zip y).groupBy(_._1).map{ case (k,v) => (k,v.map(_._2))}
    (dataMap, nj, nk)
  }

// // build and fit model
//  case class MyStructure(mu: Real, eff1: List[Real], eff2: List[Real], sigE1: Real, sigE2: Real, sigD: Real)
//
//  private val prior = for {
//    mu <- Normal(5, 10).param
//    sigE1 <- LogNormal(1, 0.2).param
//    sigE2 <- LogNormal(1, 0.2).param
//    sigD <- LogNormal(1, 4).param
//  } yield (MyStructure(mu, Nil, Nil, sigE1, sigE2, sigD))
//
//  /**
//    * Add the main effects of alpha
//    */
//  def addAplha(current: RandomVariable[MyStructure], i: Int): RandomVariable[MyStructure] = {
//    for {
//      cur <- current
//      gm_1 <- Normal(0, cur.sigE1).param
//    } yield MyStructure(cur.mu, gm_1 :: cur.eff1, cur.eff2, cur.sigE1, cur.sigE2, cur.sigD)
//  }
//
//  /**
//    * Add the main effects of beta
//    */
//  def addBeta(current: RandomVariable[MyStructure], j: Int): RandomVariable[MyStructure] = {
//    for {
//      cur <- current
//      gm_2 <- Normal(0, cur.sigE2).param
//    } yield MyStructure(cur.mu, cur.eff1, gm_2 :: cur.eff2,cur.sigE1,cur.sigE2, cur.sigD)
//  }
//
//  /**
//    * Fit to the data per group
//    */
//  def addGroup(current: RandomVariable[MyStructure], i: Int, j: Int, dataMap: Map[(Int, Int), Vector[Double]]): RandomVariable[MyStructure] = {
//    for {
//      cur <- current
//      _ <- Normal(cur.mu + cur.eff1(i) + cur.eff2(j), cur.sigD).fit(dataMap(i, j))
//    } yield cur
//
//  }
//
//  /**
//    * Add all the main effects for each group. Version: Recursion
//    */
//  @tailrec def addAllEffRecursive(alphabeta: RandomVariable[MyStructure], dataMap: Map[(Int, Int), Vector[Double]], i: Int, j: Int): RandomVariable[MyStructure] = {
//
//    val n1 = alphabeta.value.eff1.length
//    val n2 = alphabeta.value.eff2.length
//
//    val temp = addGroup(alphabeta, i, j, dataMap)
//
//    if (i == n1 - 1 && j == n2 - 1) {
//      temp
//    } else {
//      val nextJ = if (j < n2 - 1) j + 1 else 0
//      val nextI = if (j < n2 - 1) i else i + 1
//      addAllEffRecursive(temp, dataMap, nextI, nextJ)
//    }
//  }
//
//  /**
//    * Add all the main effects for each group. Version: Loop
//    */
//  def addAllEffLoop(alphabeta: RandomVariable[MyStructure], dataMap: Map[(Int, Int), Vector[Double] ]) : RandomVariable[MyStructure] = {
//    val n1 = alphabeta.value.eff1.length
//    val n2 = alphabeta.value.eff2.length
//
//    var tempAlphaBeta: RandomVariable[MyStructure] = alphabeta
//
//    for (i <- 0 until n1) {
//      for (j <- 0 until n2) {
//        tempAlphaBeta = addGroup(tempAlphaBeta, i, j, dataMap)
//      }
//    }
//    tempAlphaBeta
//  }
//
//  /**
//    * Add all the main effects for each group.
//    * We would like something like: val result = (0 until n1)(0 until n2).foldLeft(alphabeta)(addGroup(_,(_,_)))
//    * But we can't use foldLeft with double loop so we use an extra function either with loop or recursively
//    */
//  def fullModel(alphabeta: RandomVariable[MyStructure], dataMap: Map[(Int, Int), Vector[Double]]) : RandomVariable[MyStructure] = {
//    //addAllEffLoop(alphabeta, dataMap)
//    addAllEffRecursive(alphabeta, dataMap, 0, 0)
//  }
//
//  /**
//    * Use Rainier for modelling the main effects only, without interactions
//    */
//  def mainEffects(dataMap: Map[(Int, Int), Vector[Double]], rngS: ScalaRNG, n1: Int, n2: Int): Unit = {
//    implicit val rng = rngS
//    val n = dataMap.size //No of groups
//
//    val alpha = (0 until n1).foldLeft(prior)(addAplha(_, _))
//
//    val alphabeta = (0 until n2).foldLeft(alpha)(addBeta(_, _))
//
//    val fullModelRes = fullModel(alphabeta, dataMap)
//
//    //sealed abstract class Thing
//    //case class obj(ob: Real ) extends Thing
//    //case class vec(ve: List[Real]) extends Thing
//
//    val model = for {
//      mod <- fullModelRes
//    } yield Map("mu" -> mod.mu,
//      "eff1" -> mod.eff1(0),
//      "eff2" -> mod.eff2(0),
//      "sigE1" -> mod.sigE1,
//      "sigE2" -> mod.sigE2,
//      "sigD" -> mod.sigD)
//
//    // sampling
//    println("Model built. Sampling now (will take a long time)...")
//    val thin = 200
//    val out = model.sample(HMC(5), 100000, 10000 * thin, thin)
//
//    //Average parameters
//    val grouped = out.flatten.groupBy(_._1).mapValues(_.map(_._2))
//    val avg = grouped.map(l => (l._1, l._2.sum / out.length))
//
//    //println(grouped)
//    println(avg)
//    println("Sampling finished.")
//
//  }

}
