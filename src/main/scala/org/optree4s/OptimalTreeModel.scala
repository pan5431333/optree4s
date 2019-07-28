package org.optree4s

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.slf4j.LoggerFactory

abstract class AbstractOptimalTreeModel {

  private val logger = LoggerFactory.getLogger(this.getClass)

  val xCols: Array[String]
  val yCol: String
  val treeDepth: Int
  val localSearcher: LocalSearch
  private var internalTree: Option[Tree] = None

  def train(data: DataFrame)(implicit spark: SparkSession): Unit = {
    require(xCols.nonEmpty, "Given `xCols` is empty! ")
    require(treeDepth >= 1, "Given treeDepth should be >= 1! ")

    import spark.implicits._
    internalTree = Some(initializeTree(depth = treeDepth, featureDimension = xCols.length))
    val tree = internalTree.get

    val trainData = data.map { row =>
      val x = xCols.map(row.getAs[Double])
      val y = row.getAs[Double](yCol)
      x -> y
    }.rdd

    localSearcher.localSearch(tree, trainData)
    logger.info("Training done.")
  }

  def predict(data: DataFrame)(implicit spark: SparkSession): DataFrame = internalTree.map { tree =>
    import spark.implicits._
    val testData = data.map { row =>
      val x = xCols.map(row.getAs[Double])
      x
    }.rdd
    val predRes = tree.predict(testData).map(_._2)
    val columns = data.schema.add("prediction", "double")
    spark.createDataFrame(data.rdd.zip(predRes).map { case (row, prediction) =>
      Row.fromSeq(row.toSeq :+ prediction)
    }, columns)
  }.getOrElse {
    throw new IllegalArgumentException("Model has not been trained! Please train the model first using `.train` method")
    null
  }

  def initializeTree(depth: Int, featureDimension: Int): Tree
}

case class OptimalTreeModel(override val xCols: Array[String],
                            override val yCol: String,
                            override val treeDepth: Int
                           ) extends AbstractOptimalTreeModel {
  override val localSearcher: LocalSearch = new LocalSearch()

  override def initializeTree(depth: Int, featureDimension: Int): Tree = Tree.initRandomBinary(1, depth, featureDimension)
}

object AbstractOptimalTreeModelTest {

  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession.builder().appName("test").master("local[*]").getOrCreate()
    import spark.implicits._

    val data = spark.createDataset(Seq(
      (1.0, 1.0, 1.0),
      (2.0, 2.0, 1.0),
      (2.0, 1.0, -1.0),
      (2.0, 0.0, -1.0),
      (3.0, 1.0, -1.0)
    )).toDF("x1", "x2", "y")
    val testData = spark.createDataset(Seq(
      (1.0, 1.0, 1.0),
      (2.0, 2.0, 1.0),
      (2.0, 1.0, -1.0),
      (2.0, 0.0, -1.0),
      (3.0, 1.0, -1.0),
      (3.0, 0.0, -1.0)
    )).toDF("x1", "x2", "y")
    val model = OptimalTreeModel(xCols = Array("x1", "x2"), yCol = "y", treeDepth = 2)
    model.train(data)
    model.predict(testData).show()
  }
}
