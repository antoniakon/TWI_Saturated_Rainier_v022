import com.stripe.rainier.compute._
import com.stripe.rainier.core._
import com.stripe.rainier.sampler._
import annotation.tailrec

object Anova2wayWithInters_v2 {

    def main(args: Array[String]): Unit = {
      val n1 = 3 // levels of var1
      val n2 = 4 // levels of var2
      val rng = ScalaRNG(3)
      val (dataMap, dataWithInterMap) = dataGeneration(rng, n1, n2)
      mainEffectsAndInters(dataWithInterMap, rng, n1, n2)
    }

    /**
      * Generate data with and without interactions
      */
    def dataGeneration(rng: ScalaRNG, n1: Int, n2: Int): (Map[(Int, Int), Vector[Double]], Map[(Int, Int), Vector[Double]]) = {

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

      val pairedgroups = for (i <- (0 until n1); j <- (0 until n2)) yield (i, j) // Create a vector with all the combinations of levels: Vector((1,1), (1,2), ..., (2,1), (2,2),...)
      val pairedEff = for (i <- effects1; j <- effects2) yield (i, j) // Pairing the effects for the groups as defined from pairedgroups
      val data = pairedEff map { e => Vector.fill(N)(mu + e._1 + e._2 + sigD * rng.standardNormal) } // Create the "random" N observations per group by using the effects for each variable + mu + error. Result: Vector(Vector(...N obs for group (1,1)..., Vector(...N obs for group (1,2)..., ...)

      val dataMap = (pairedgroups zip data).toMap
      val dataWithInter = inters.flatMap(i => data.map(v => v.map(elem => i + elem))) //Add the interactions to the data
      val dataWithInterMap = (pairedgroups zip dataWithInter).toMap

      println(s"effects1: ", effects1)
      println(s"effects2: ", effects2)
      //    println(pairedgroups)
      //    println(pairedEff)
      //    println(data)
      //    println(inters)
      //    println(dataWithInter)
      //    println(data.length)
      //    println(dataMap(1,1))
      //  println(dataMap.keys)
      //    println(dataWithInterMap)

      (dataMap, dataWithInterMap)
    }

    // build and fit model
    case class MyStructure(mu: Real, eff1: List[Real], eff2: List[Real], effg: Vector[Vector[Real]], sigE1: Real, sigE2: Real, sigEg: Real, sigD: Real)

    private val prior = for {
      mu <- Normal(5, 10).param
      sigE1 <- LogNormal(1, 0.2).param
      sigE2 <- LogNormal(1, 0.2).param
      sigEg <- LogNormal(1, 0.2).param
      sigD <- LogNormal(1, 4).param
    } yield MyStructure(mu, Nil, Nil, Vector(Vector(0)), sigE1, sigE2, sigEg, sigD)

    /**
      * Add the main effects of alpha
      */
    def addAplha(current: RandomVariable[MyStructure], i: Int): RandomVariable[MyStructure] = {
      for {
        cur <- current
        gm_1 <- Normal(0, cur.sigE1).param
      } yield MyStructure(cur.mu, gm_1 :: cur.eff1, cur.eff2, cur.effg, cur.sigE1, cur.sigE2, cur.sigEg, cur.sigD)
    }

    /**
      * Add the main effects of beta
      */
    def addBeta(current: RandomVariable[MyStructure], j: Int): RandomVariable[MyStructure] = {
      for {
        cur <- current
        gm_2 <- Normal(0, cur.sigE2).param
      } yield MyStructure(cur.mu, cur.eff1, gm_2 :: cur.eff2,cur.effg, cur.sigE1,cur.sigE2, cur.sigEg , cur.sigD)
    }

  /**
    * Add the interaction effects
    */
  def addGamma(current: RandomVariable[MyStructure]): RandomVariable[MyStructure] = {
    val n1 = current.value.eff1.length
    val n2 = current.value.eff2.length

    val tempGamma = Vector.fill(n1, n2)(Normal(0, current.value.sigEg).param.value)
// Not necessary
//    for (i <- 0 until n1) {
//      for (j <- 0 until n2) {
//        tempGamma(i)(j) = Normal(0, current.value.sigEg).param.value
//      }
//    }

    for {
      cur <- current
    }yield MyStructure (cur.mu, cur.eff1, cur.eff2, tempGamma, current.value.sigE1, current.value.sigE2, current.value.sigEg, current.value.sigD)
  }


  def addAlpha1(current: RandomVariable[MyStructure], n1:Int) = {
    val eff1 = List.fill(n1) { Normal(0, current.value.sigE1).param.value }

    for {
      cur <- current
    } yield MyStructure(cur.mu, eff1, cur.eff2, cur.effg, cur.sigE1, cur.sigE2, cur.sigEg, cur.sigD)
  }


    /**
      * Fit to the data per group
      */
    def addGroup(current: RandomVariable[MyStructure], i: Int, j: Int, dataMap: Map[(Int, Int), Vector[Double]]): RandomVariable[MyStructure] = {
      for {
        cur <- current
        _ <- Normal(cur.mu + cur.eff1(i) + cur.eff2(j)+ cur.effg(i)(j), cur.sigD).fit(dataMap(i, j))
      } yield cur

    }


    /**
      * Add all the main effects for each group. Version: Recursion
      */
    @tailrec def addAllEffRecursive(alphabeta: RandomVariable[MyStructure], dataMap: Map[(Int, Int), Vector[Double]], i: Int, j: Int): RandomVariable[MyStructure] = {

      val n1 = alphabeta.value.eff1.length
      val n2 = alphabeta.value.eff2.length

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
      * Add all the main effects for each group.
      * We would like something like: val result = (0 until n1)(0 until n2).foldLeft(alphabeta)(addGroup(_,(_,_)))
      * But we can't use foldLeft with double loop so we use an extra function either with loop or recursively
      */
    def fullModel(alphabeta: RandomVariable[MyStructure], dataMap: Map[(Int, Int), Vector[Double]]) : RandomVariable[MyStructure] = {
      //addAllEffLoop(alphabeta, dataMap)
      addAllEffRecursive(alphabeta, dataMap, 0, 0)
    }

    /**
      * Use Rainier for modelling the main effects only, without interactions
      */
    def mainEffectsAndInters(dataMap: Map[(Int, Int), Vector[Double]], rngS: ScalaRNG, n1: Int, n2: Int): Unit = {
      implicit val rng = rngS
      val n = dataMap.size //No of groups

      val alpha = (0 until n1).foldLeft(prior)(addAplha(_, _))

      val alphabeta = (0 until n2).foldLeft(alpha)(addBeta(_, _))

      val fullModelRes = fullModel(addGamma(alphabeta), dataMap)

      //sealed abstract class Thing
      //case class obj(ob: Real ) extends Thing
      //case class vec(ve: List[Real]) extends Thing

      val model = for {
        mod <- fullModelRes
      } yield Map("mu" -> mod.mu,
        "eff1" -> mod.eff1(0),
        "eff2" -> mod.eff2(0),
        "sigE1" -> mod.sigE1,
        "sigE2" -> mod.sigE2,
        "sigD" -> mod.sigD)

      // sampling
      println("Model built. Sampling now (will take a long time)...")
      val thin = 200
      val out = model.sample(HMC(5), 100000, 10000 * thin, thin)

      //Average parameters
      val grouped = out.flatten.groupBy(_._1).mapValues(_.map(_._2))
      val avg = grouped.map(l => (l._1, l._2.sum / out.length))

      //println(grouped)
      println(avg)
      println("Sampling finished.")

    }


}