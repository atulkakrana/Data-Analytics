from pyspark import SparkConf, SparkContext

conf    = SparkConf().setMaster("local").setAppName("FriendsByAge")
sc      = SparkContext(conf = conf)

def parseLine(line):
    fields      = line.split(',')
    age         = int(fields[2])
    numFriends  = int(fields[3])
    return (age, numFriends)

# lines           = sc.textFile("/Users/atul/Work/spark/fakefriends.csv")
lines           = sc.textFile("/home/atul/Google-Drive/2.atul/Atul-work/spark/fakefriends.csv")
rdd             = lines.map(parseLine) ## Key-value RDD; age is key and number of freinds is value

totalsByAge     = rdd.mapValues(lambda x: (x, 1)).reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) ## Functions are mapped to values
averagesByAge   = totalsByAge.mapValues(lambda x: x[0] / x[1])
results         = averagesByAge.collect()

for result in results:
    print(result)
    
sc.stop()
