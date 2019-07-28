package org.optree4s

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.slf4j.LoggerFactory

import scala.io.Source
import scala.util.Random

class LocalSearch extends Serializable {

  private val logger = LoggerFactory.getLogger(this.getClass)
  private val bigM: Double = 1e8

  def localSearch(tree: Tree, data: RDD[(Array[Double], Double)]): Tree = {
    logger.info("Caching the training data...")
    data.cache()

    var errorPrevious = this.bigM
    var errorCurrent = tree.loss(data)
    logger.info(s"The initial error is: $errorCurrent")

    while (errorCurrent != errorPrevious) {
      if (errorPrevious > errorCurrent) {
        errorPrevious = errorCurrent
      }

      for (t <- Random.shuffle(tree.parentNodes)) {
        logger.info(s"Visiting node $t...")
        val subtree = tree.subtree(t)
        val subdata = tree.subdata(t)
        if (!subdata.isEmpty()) {
          subtree.optimizeRoot(subdata)
          val newParameters = Map(t -> (subtree.a(t), subtree.b(t)))
          val originalParameters = Map(t -> (tree.a(t), tree.b(t)))
          if (newParameters != originalParameters) {
            val originalGlobalError = errorCurrent
            val updatedGlobalError = tree.update(newParameters).loss(data)
            errorCurrent = updatedGlobalError

            logger.info(s"originalGlobalError: $originalGlobalError, updatedGlobalError: $updatedGlobalError")

            // if the new parameters cannot improve the tree's global loss, fallback to the original parameters.
            if (updatedGlobalError >= originalGlobalError) {
              tree.update(originalParameters)
              errorCurrent = originalGlobalError
            }
          }
        }
      }
    }
    tree
  }
}


