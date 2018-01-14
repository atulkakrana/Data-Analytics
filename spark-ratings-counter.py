from pyspark import SparkConf, SparkContext
import collections

### Assingment -1
conf    = SparkConf().setMaster("local").setAppName("RatingsHistogram")
sc      = SparkContext(conf = conf)

lines   = sc.textFile("/Users/atul/Work/spark/ml-100k/u.data")  ## Read files and split it into pieces (RDD object)
ratings = lines.map(lambda x: x.split()[2])                     ## RDD transformation; get Ratings (into new RDD object)
result  = ratings.countByValue()                                ## Action on RDD; counts by value [(1,10),(2,97),(3,100)], its a dict
print(result)

sortedResults = collections.OrderedDict(sorted(result.items())) ## Pure python code
print(sortedResults)

for key, value in sortedResults.items():
    print("%s %i" % (key, value))
    
    
### Find most popular movie, i.e. with most entries or rated by most people
###################################################
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

# adict = movieDict()

def main():
    lines   = sc.textFile("/Users/atul/Work/spark/ml-100k/u.data")  ## Read files and split it into pieces (RDD object)
    
    ## Create Movie and count tuple
    movies  = lines.map(lambda x: (x.split()[1],1)) ## Get Movie ID, and find which moview was watched most frequesntly
    
    ## Get total counts for each movie
    counts  = movies.reduceByKey(lambda x,y: x+y)
    
    ## Flip elements to sort on counts
    flipped = counts.map(lambda x: (x[1],x[0])).sortByKey(ascending= False)
    
    namedict        = sc.broadcast(movieDict()) ### Make a dict and broadcast to workers
    movieWithNames  = flipped.map(lambda x:(namedict.value[x[1]],x[0]))  ## Query the dictionary
    results         = movieWithNames.collect()
    
    for ares in results:
        print(ares)
    
    
if __name__=='__main__':
    conf    = SparkConf().setMaster("local").setAppName("MostPopular")
    sc      = SparkContext(conf = conf)
    main()
    sc.stop()
    print("\nDone")
    
    
sc.stop()


    
