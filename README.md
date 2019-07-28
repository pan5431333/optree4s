# optree4s 
OPtimal decision TREE FOR Spark. 

Run Bertsimas's [optimal decision tree](https://dspace.mit.edu/handle/1721.1/110328) distributionally on the world's most popular big data platform Spark. 

This project is designed for being used with Scala. For Python version, please refer to this project [pyoptree](https://github.com/pan5431333/pyoptree). Unfortunately, distributional computation is not supported currently in the Python version. 

Both of the Scala/Spark version and the Python version are under active(?) development. But since my work in Alibaba is so busy, I certainly welcome anyone's fork&pull request. 

### A minimal runnable example 
A minimal runnable example is as follows, please don't hesitate to contact me by (meng.pan95@gmail.com) if you encountered any trouble or have any suggestion. I usually will check my Email every night, and I promise to respond every Email from GitHub~~

```scala
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
``` 

