### Machine-Learning Library - Datasets
### MLLIB template
### DataType - Vectors dense

from __future__ import print_function
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

def main():
    lines   = spark.sparkContext.textFile("/Users/atul/Google Drive/2.atul/Atul-work/spark/regression.txt")
    data    = lines.map(lambda x: x.split(",")).map(lambda x: (float(x[0]), Vectors.dense(float(x[1]))))
    
    ## Convert this RDD to a DataFrame
    colNames    = ["label", "features"]
    df          = data.toDF(colNames)
    
    ## Let's split our data into training data and testing data
    trainTest   = df.randomSplit([0.5, 0.5])
    trainingDF  = trainTest[0]
    testDF      = trainTest[1]
    
    ## Create Linear Regression model
    lir     = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    ## Train model using our training data
    model   = lir.fit(trainingDF)
    
    ## See if we can predict values in our test data.
    ## Generate predictions using our linear regression model for all features in our
    ## test dataframe:
    fullPredictions = model.transform(testDF).cache()

    ## Extract the predictions and the "known" correct labels.
    predictions     = fullPredictions.select("prediction").rdd.map(lambda x: x[0])
    labels          = fullPredictions.select("label").rdd.map(lambda x: x[0])

    ## Zip them together
    predictionAndLabel = predictions.zip(labels).collect()

    # Print out the predicted and actual values for each point
    for prediction in predictionAndLabel:
      print(prediction)


if __name__=='__main__':
    spark = SparkSession.builder.appName("LinearRegression").getOrCreate()
    main()
    spark.stop()
    print("\nDone")
    
spark.stop()