package org.optree4s

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.util.Random

abstract class Tree(val rootNode: Int,
                    val depth: Int,
                    val a: Map[Int, Array[Double]],
                    val b: mutable.Map[Int, Double],
                    val alpha: Float = 0F) extends Serializable {
  require(rootNode >= 1, s"root node must be greater than or equal to 1! The given root node is $rootNode")
  require(depth >= 0, s"depth must be non-negative! The given depth is $depth")
  parentNodes.foreach { node: Int =>
    require(a.keySet.contains(node), s"The given a doesn't contain parent node $node! The given a is $a")
    require(b.keySet.contains(node), s"The given b doesn't contain parent node $node! The given b is $a")
  }

  private var nodeLabelMapping: Map[Int, Double] = Map.empty
  private var nodeDataMapping: RDD[(Int, (Array[Double], Double))] = null

  val depthInACompleteTree: Int = Tree.calculateDepthInACompleteTree(rootNode, depth)

  lazy val nodes: List[Int] = Tree.getNodes(rootNode, depth)

  lazy val parentNodes: List[Int] = Tree.getParentNodes(rootNode, depth)

  lazy val leafNodes: List[Int] = Tree.getLeafNodes(rootNode, depth)

  def optimizeRoot(data: RDD[(Array[Double], Double)]): Unit

  def subtree(subtreeRootNode: Int): Tree = {
    require(nodes.contains(subtreeRootNode), s"The given subtreeRootNode is not contains in this tree! Given: $subtreeRootNode")
    require(depth > 0, "The current tree's depth is 0, which means it's a leaf node, thus you cannot create subtrees for a leaf node! ")

    val subtreeDepth = depth - Tree.calculateRootDepthInACompleteTree(subtreeRootNode)
    val subtreeParentNodes = Tree.getParentNodes(subtreeRootNode, subtreeDepth)
    val subtreeA = a
      .filter { case (node, _) => subtreeParentNodes.contains(node) }
      .map { case (node, aa) =>
        val newArray = Array.fill[Double](aa.length)(0.0)
        aa.copyToArray(newArray)
        node -> newArray
      }
    val subtreeB = b.filter { case (node, _) => subtreeParentNodes.contains(node) }

    this match {
      case _: BinaryTree => BinaryTree(subtreeRootNode, subtreeDepth, subtreeA, subtreeB, alpha)
      case _: HyperTree => HyperTree(subtreeRootNode, subtreeDepth, subtreeA, subtreeB, alpha)
    }
  }

  def children: (Tree, Tree) = {
    assert(depth > 0, "The current tree's depth is 0, which means it's a leaf node, thus you cannot create children for leaf node")
    (subtree(rootNode * 2), subtree(rootNode * 2 + 1))
  }

  def loss(data: RDD[(Array[Double], Double)]): Double = {
    assert(!data.isEmpty(), "The given data to calculate loss is empty! ")

    val dataEvaluated = data.mapPartitions(iter =>
      iter.map { case (x, y) =>
        val leafNodeFallInto = Tree.evaluateSingle(x, a, b, depth, rootNode)
        leafNodeFallInto -> x -> y -> 1
      }
    )

    nodeDataMapping = dataEvaluated.map { case (((node, x), y), _) =>
      (node, (x, y))
    }

    nodeLabelMapping = dataEvaluated
      .map { case (((node, feature), label), cnt) => ((node, label), cnt) }
      .reduceByKey(_ + _)
      .map { case ((node, label), cnt) => (node, (label, cnt)) }
      .reduceByKey { case ((label1, cnt1), (label2, cnt2)) =>
        if (cnt1 >= cnt2) label1 -> cnt1 else label2 -> cnt2
      }.map { case (node, (label, cnt)) => (node, label) }
      .collect().toMap

    val loss = defineLoss(dataEvaluated.map { case (((node, _), y), _) =>
      nodeLabelMapping(node) -> y
    })

    loss
  }

  def update(nodeParametersMapping: Map[Int, (Array[Double], Double)]): this.type = {
    for ((node, (a, b)) <- nodeParametersMapping) {
      if (parentNodes.contains(node)) {
        a.copyToArray(this.a(node))
        this.b(node) = b
      }
    }
    this
  }

  def subdata(t: Int): RDD[(Array[Double], Double)] = {
    if (nodeDataMapping == null) {
      throw new IllegalArgumentException("The tree model has not been trained yet. Please train it first (call .loss() method) before calling .subdata() method! ")
    }

    val leafNodesOfSubtree = subtree(t).leafNodes.toSet

    nodeDataMapping.filter { case (node, _) =>
      leafNodesOfSubtree.contains(node)
    }.map(_._2)
  }

  def predict(data: RDD[Array[Double]]): RDD[(Array[Double], Double)] = {
    if (nodeLabelMapping.isEmpty)
      throw new IllegalArgumentException("The tree model has not been trained yet. Please train it first! (Call the .loss() method)")
    data.mapPartitions { iter =>
      iter.map { x => x -> nodeLabelMapping(Tree.evaluateSingle(x, a, b, depth, rootNode)) }
    }
  }

  def copy(): Tree = {
    val newA = a.map { case (node, aa) =>
      val newArray = Array.fill[Double](aa.length)(0.0)
      aa.copyToArray(newArray)
      node -> newArray
    }
    val newB = mutable.Map(b.toList: _*)

    this match {
      case _: BinaryTree => BinaryTree(rootNode, depth, newA, newB, alpha)
      case _: HyperTree => HyperTree(rootNode, depth, newA, newB, alpha)
    }
  }

  protected def defineLoss(data: RDD[(Double, Double)]): Double = {
    val numRows = data.count()
    data.filter { case (yPredict, yTrue) =>
      yPredict.round.toInt != yTrue.round.toInt
    }.count() / numRows.toDouble
  }
}

case class CompleteTree(override val depth: Int,
                        override val a: Map[Int, Array[Double]],
                        override val b: mutable.Map[Int, Double],
                        override val alpha: Float = 0F)
  extends Tree(1, depth, a, b, alpha) {
  override def optimizeRoot(data: RDD[(Array[Double], Double)]): Unit = ???
}

case class BinaryTree(override val rootNode: Int,
                      override val depth: Int,
                      override val a: Map[Int, Array[Double]],
                      override val b: mutable.Map[Int, Double],
                      override val alpha: Float = 0F)
  extends Tree(rootNode, depth, a, b, alpha) {

  lazy val featureDimension: Int = a.head._2.length
  private val logger = LoggerFactory.getLogger(this.getClass)

  override def optimizeRoot(data: RDD[(Array[Double], Double)]): Unit = {
    var currentLoss = loss(data)
    for (j <- Random.shuffle((0 until featureDimension).toList)) {
      logger.info(s"Optimizing dimension $j...")
      val sorted = data.map(_._1(j)).sortBy(k => k).collect()
      val checkPoints = sorted.scanLeft((0.0, 0.0)) { case ((previous, res), current) =>
        (current, (current + previous) / 2.0)
      }.tail.map(_._2)
      val losses = checkPoints.par.map { checkPoint =>
        val newTree = this.copy()
        restoreA(newTree.a, rootNode)
        newTree.a(rootNode)(j) = 1.0
        newTree.b(rootNode) = checkPoint
        val loss = newTree.loss(data)
        checkPoint -> loss
      }
      val (finalCheckPoint, finalLoss) = losses.minBy(_._2)

      if (finalLoss < currentLoss) {
        b(rootNode) = finalCheckPoint
        restoreA(a, rootNode)
        a(rootNode)(j) = 1.0
        currentLoss = finalLoss
      }
    }
  }

  private def restoreA(a: Map[Int, Array[Double]], node: Int): Unit = {
    for (j <- 0 until featureDimension) {
      a(node)(j) = 0.0
    }
  }
}

case class HyperTree(override val rootNode: Int,
                     override val depth: Int,
                     override val a: Map[Int, Array[Double]],
                     override val b: mutable.Map[Int, Double],
                     override val alpha: Float = 0F)
  extends Tree(rootNode, depth, a, b, alpha) {
  override def optimizeRoot(data: RDD[(Array[Double], Double)]): Unit = ???
}


object Tree {

  /** Factory Creation Methods: methods' names starting with .initXXXX() *****/

  def initRandomHyper(rootNode: Int, depth: Int, featureDimension: Int, alpha: Float = 0): Tree = {
    require(featureDimension >= 1, s"feature's dimension must be positive integer! Got $featureDimension")
    val parentNodes: List[Int] = getParentNodes(rootNode, depth)
    val initA = parentNodes.map { node => node -> (1 to featureDimension).map(_ => Math.random() * 2 - 1).toArray }.toMap
    val initB = mutable.Map(parentNodes.map { node => node -> (Math.random() * 2 - 1) }: _*)
    HyperTree(rootNode, depth, initA, initB, alpha)
  }

  def initRandomHyperCompleteTree(depth: Int, featureDimension: Int, alpha: Float = 0): Tree = {
    initRandomHyper(1, depth, featureDimension, alpha)
  }

  def initRandomBinary(rootNode: Int, depth: Int, featureDimension: Int, alpha: Float = 0): Tree = {
    require(featureDimension >= 1, s"feature's dimension must be positive integer! Got $featureDimension")
    val parentNodes: List[Int] = getParentNodes(rootNode, depth)
    val initA = parentNodes.map { node =>
      val randomIndex = (Math.random() * featureDimension).floor.toInt
      val a = Array.fill[Double](featureDimension)(0.0)
      a(randomIndex) = 1.0
      node -> a
    }.toMap
    val initB = mutable.Map(parentNodes.map { node => node -> 0.0 }: _*)
    BinaryTree(rootNode, depth, initA, initB, alpha)
  }

  def initRandomBinaryCompleteTree(depth: Int, featureDimension: Int, alpha: Float = 0): Tree = {
    initRandomBinary(1, depth, featureDimension, alpha)
  }

  private def getNodes(rootNode: Int, depth: Int): List[Int] = {
    (1 to depth)
      .foldLeft[(List[Int], List[Int])]((List(rootNode), List(rootNode))) {
      case ((nodes, previousDepthNodes), _) =>
        val (newNodes, newPreviousDepthNodes) = previousDepthNodes
          .foldLeft[(List[Int], List[Int])]((nodes, List())) {
          case ((nodes, thisDepthNodes), node) =>
            val leftChild = node * 2
            val rightChild = leftChild + 1
            (leftChild :: rightChild :: nodes, leftChild :: rightChild :: thisDepthNodes)
        }
        (newNodes, newPreviousDepthNodes)
    }._1.sorted
  }

  private def calculateRootDepthInACompleteTree(rootNode: Int): Int = {
    (Math.log(rootNode + 1) / Math.log(2.0)).ceil.toInt - 1
  }

  private def calculateDepthInACompleteTree(rootNode: Int, depth: Int): Int = {
    calculateRootDepthInACompleteTree(rootNode) + depth
  }

  private def getParentLeafSplitNode(depthInACompleteTree: Int): Int = {
    Math.pow(2, depthInACompleteTree).toInt
  }

  private def getParentNodes(rootNode: Int, depth: Int): List[Int] = {
    getNodes(rootNode, depth).filter(_ < getParentLeafSplitNode(calculateDepthInACompleteTree(rootNode, depth)))
  }

  private def getLeafNodes(rootNode: Int, depth: Int): List[Int] = {
    getNodes(rootNode, depth).filter(_ >= getParentLeafSplitNode(calculateDepthInACompleteTree(rootNode, depth)))
  }

  private[optree4s] def convertBinaryAToHyperA(binaryA: Int, featureDimension: Int): Array[Double] = {
    val zeros = Array.fill(featureDimension)(0.0)
    val randomIndex = (Math.random() * featureDimension).floor.toInt
    zeros(randomIndex) = 1.0
    zeros
  }

  def evaluateSingle(x: Array[Double], a: Map[Int, Array[Double]], b: mutable.Map[Int, Double], depth: Int, rootNode: Int): Int = {
    val res = (0 until depth).foldLeft(rootNode) { case (node, _) =>
      val res = x.zip(a(node)).map { case (aa, bb) => aa * bb }.sum
      if (res < b(node)) node * 2 else node * 2 + 1
    }
    res
  }
}


object TreeTest extends App {
  val tree1 = Tree.initRandomHyper(1, 2, 3, 0)
  assert(tree1.nodes == List(1, 2, 3, 4, 5, 6, 7), s"Got nodes: ${tree1.nodes}")
  assert(tree1.parentNodes == List(1, 2, 3), s"Got parent nodes: ${tree1.parentNodes}")
  assert(tree1.leafNodes == List(4, 5, 6, 7), s"Got leaf nodes: ${tree1.leafNodes}")

  val tree2 = Tree.initRandomBinary(2, 2, 3)
  assert(tree2.nodes == List(2, 4, 5, 8, 9, 10, 11), s"Got nodes: ${tree2.nodes}")
  assert(tree2.parentNodes == List(2, 4, 5), s"Got parent nodes: ${tree2.parentNodes}")
  assert(tree2.leafNodes == List(8, 9, 10, 11), s"Got leaf nodes: ${tree2.leafNodes}")

  assert(tree1.subtree(2).nodes == List(2, 4, 5))
  assert(tree1.subtree(4).leafNodes == List(4))
  assert(tree1.subtree(4).parentNodes == List())

  val (leftChild, rightChild) = tree1.children
  assert(leftChild.nodes == List(2, 4, 5))
  assert(rightChild.nodes == List(3, 6, 7))

  val conf = new SparkConf().setMaster("local[*]").setAppName("Test")
  val sc = SparkContext.getOrCreate(conf)
  val testData: RDD[(Array[Double], Double)] = sc.parallelize(Seq(
    Array(1.0, 2.0, 3.0) -> 1.0,
    Array(2.0, 3.0, 4.0) -> 1.0,
    Array(3.0, 4.0, 5.0) -> 2.0
  ))

  // no expected answer here due to random initialization. What we can test here is to make sure the
  // two functions can be executed properly without throwing any unexpected exception.
  tree1.loss(testData)
  tree1.predict(testData.map(_._1)).collect().toSeq.map { case (x, y) => x.toSeq -> y }
}
