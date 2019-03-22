import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._
import scala.collection.breakOut
import annotation.tailrec

object Anova2way{

  def main(args: Array[String]): Unit = {
    val n1 = 3 // levels of var1
    val n2 = 4 // levels of var2
    val rng = ScalaRNG(3)
    val (dataWithoutInter, dataWithInter) = dataGeneration(rng, n1, n2)
    mainEffects(dataWithoutInter, rng, n1, n2)
  }

  /**
    *
    * Generate data with and without interactions
    */
  def dataGeneration(rng: ScalaRNG, n1: Int, n2: Int): (Map[(Int, Int), Vector[Double]], Vector[Vector[Double]]) = {

    val n = n1 * n2 //no of groups
    val N = 5 // obs per group
    val mu = 5.0 // overall mean
    val sigE1 = 2.0 // random effect var1 SD
    val sigE2 = 1.5 //random effect var2 SD
    val sigInter = 1.0 //random effect interactions SD
    val sigD = 3.0 // obs SD
    val effects1 = Vector.fill(n1)(sigE1 * rng.standardNormal) // Effects1
    val effects2 = Vector.fill(n2)(sigE2 * rng.standardNormal) // Effects2
    val inters = Vector.fill(n)(sigInter * rng.standardNormal) // Interaction effects

    val pairedgroups = for (i <- (1 to n1); j <- (1 to n2)) yield (i, j) // Create a vector with all the combinations of levels: Vector((1,1), (1,2), ..., (2,1), (2,2),...)
    val pairedEff = for (i <- effects1; j <- effects2) yield (i, j) // Pairing the effects for the groups as defined from pairedgroups
    val data = pairedEff map { e => Vector.fill(N)(mu + e._1 + e._2 + sigD * rng.standardNormal) } // Create the "random" N observations per group by using the effects for each variable + mu + error. Result: Vector(Vector(...N obs for group (1,1)..., Vector(...N obs for group (1,2)..., ...)

    val dataMap = (pairedgroups zip data).toMap
    val dataWithInter = inters.flatMap(i => data.map(v => v.map(elem => i + elem))) //Add the interactions to the data
    val dataWithInterMap = (pairedgroups zip dataWithInter).toMap

    println(effects1)
    println(effects2)
//    println(pairedgroups)
//    println(pairedEff)
//    println(data)
//    println(inters)
//    println(dataWithInter)
//    println(data.length)
//    println(dataMap(1,1))
//    println(dataMap)
//    println(dataWithInterMap)

    (dataMap, dataWithInter)
  }


  // build and fit model
  case class MyStructure(mu: Real, eff1: List[Real], eff2: List[Real], sigE1: Real, sigE2: Real, sigD: Real)

  private val prior = for {
    mu <- Normal(5, 10).param
    eff1 <- Normal(1, 0.2).param
    eff2 <- Normal(1, 0.2).param
    sigE1 <- LogNormal(0, 2).param
    sigE2 <- Normal(1,0.2).param
    sigD <- LogNormal(1, 4).param
  } yield (MyStructure(mu, List(eff1), List(eff2), sigE1, sigE2, sigD))


  def addAplha(current: RandomVariable[MyStructure], i: Int): RandomVariable[MyStructure]= {
    for {
      cur <- current
      gm_1 <- Normal(0, cur.sigE1).param
    } yield (MyStructure(cur.mu, gm_1::cur.eff1, cur.eff2, cur.sigE1, cur.sigE2, cur.sigD))
  }

  def addBeta(current: RandomVariable[MyStructure], j: Int): RandomVariable[MyStructure]= {
    for {
      cur <- current
      gm_2 <- Normal(0, cur.sigE2).param
    } yield (MyStructure(cur.mu, cur.eff1, gm_2::cur.eff2, cur.sigE1, cur.sigE2, cur.sigD))
  }


  def addGroup(current: RandomVariable[MyStructure], i: Int, j:Int, dataMap: Map[(Int, Int), Vector[Double] ] ): RandomVariable[MyStructure] = {
    for {
      cur <- current
      _<- Normal(cur.mu + cur.eff1(i) + cur.eff2(j), cur.sigD).fit(dataMap(i,j))
    } yield (MyStructure(cur.mu, cur.eff1, cur.eff2, cur.sigE1, cur.sigE2, cur.sigD))

  }

  @tailrec def addAllEffRecursive(alphabeta: RandomVariable[MyStructure], dataMap: Map[(Int, Int), Vector[Double]], i: Int, j: Int ) : RandomVariable[MyStructure] = {

    val n1 = alphabeta.value.eff1.length
    val n2 = alphabeta.value.eff2.length

    val temp = addGroup(alphabeta, i, j, dataMap)

    if (i == n1-1 && j == n2-1) {
      temp
    } else {
      val nextJ = if (j < n2-1) j+1 else 0
      val nextI = if (j < n2-1) i else i+1
      addAllEffRecursive(temp, dataMap, nextI, nextJ)
    }
  }

  def fullModel(alphabeta: RandomVariable[MyStructure], dataMap: Map[(Int, Int), Vector[Double]]) : RandomVariable[MyStructure] = {
    //    meLoop(alphabeta, dataMap)
    addAllEffRecursive(alphabeta, dataMap, 0, 0)
  }

  def addAllEffLoop(alphabeta: RandomVariable[MyStructure], dataMap: Map[(Int, Int), Vector[Double] ]) : RandomVariable[MyStructure] = {
    val n1 = alphabeta.value.eff1.length
    val n2 = alphabeta.value.eff2.length

    var tempAlphaBeta: RandomVariable[MyStructure] = alphabeta

    for (i <- 0 until n1) {
      for (j <- 0 until n2) {
        tempAlphaBeta = addGroup(tempAlphaBeta, i, j, dataMap)
      }
    }

    tempAlphaBeta
  }

  /**
    *
    * Use Rainier for modelling the main effects only, without interactions
    */
  def mainEffects(dataMap: Map[(Int, Int), Vector[Double] ], rngS: ScalaRNG, n1: Int, n2:Int): Unit= {
    implicit val rng= rngS
    val n= dataMap.size //No of groups


    val alpha=(0 until n1).foldLeft(prior)(addAplha(_,_))
    println(alpha.value.mu)

    val alphabeta= (0 until n2).foldLeft(alpha)(addBeta(_,_))

    //      val result= (0 until n).foldLeft(alphabeta)(addGroup(_)
    val fullModelRes = fullModel(alphabeta, dataMap)

    sealed abstract class Thing
    case class obj(ob: Real ) extends Thing
    case class vec(ve: List[Real]) extends Thing

          val model = for {
            mod <- fullModelRes
          } yield
            Map("mu" -> mod.mu,
              "eff1" -> mod.eff1(1),
              "eff2" -> mod.eff2(1),
              "sigE1" -> mod.sigE2,
              "sigE2" -> mod.sigE2,
              "sigD" -> mod.sigD)

          // sampling
          println("Model built. Sampling now (will take a long time)...")
          val thin = 200
          val out = model.sample(HMC(5), 100000, 100000 * thin, thin)

    println(out)
          //Average parameters
          //val grouped= out.flatten.groupBy(_._1).mapValues(_.map(_._2))
          //val avg= grouped.map(l=> (l._1,l._2.sum/out.length)) //For the SP we don't need an average

          //println(grouped)
          //println(avg)
          println("Sampling finished.")

  }
}