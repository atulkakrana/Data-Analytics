### Most Popular Movie using spark.sql

from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions

def movieDict():
    adict = {}
    fh_in = open("/Users/atul/Work/spark/ml-100k/u.item",'r',encoding="latin1")
    aread = fh_in.readlines()
    fh_in.close()
    
    for aline in aread:
        # print(aline)
        ent = aline.split('|')
        adict[ent[0]] = ent[1]
    
    return adict

def main():
    nameDict    = movieDict()
    lines       = spark.sparkContext.textFile("/Users/atul/Work/spark/ml-100k/u.data")
    
    ## Convert to RDD of row objects
    movies      = lines.map(lambda x: Row(movieID = int(x.split()[1])))
    
    ## Convert to dataFrame/dataSet
    movieDataset= spark.createDataFrame(movies)
    
    ## SQL-query to sort all movies by popularity
    topMovieIDs = movieDataset.groupBy("movieID").count().orderBy("count",ascending=False).cache()
    
    ## See top Ids
    topMovieIDs.show()
    top10 = topMovieIDs.take(10)
    
    
    # Print the results, use dict
    for result in top10:
    # Each row has movieID, count as above.
        print("%s: %d" % (nameDict[result[0]], result[1]))


if __name__=='__main__':
    spark    = SparkSession.builder.appName("PopularMovies").getOrCreate()
    main()
    spark.stop()
    print("\nDone")
    
    
spark.stop()