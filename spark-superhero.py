### Superhero templates
########################

def countpartners(line):
    ent     = line.split()
    aval    = (ent[0],len(ent)-1)
    return aval

def parsename(line):
    ent = line.split('\"')
    aval = (int(ent[0]),ent[1].replace('/"',"").encode("utf8"))
    return aval

with open("/Users/atul/Google Drive/2.atul/Atul-work/spark/Marvel-Names.txt") as f:
    for line in f:
        print(parsename(line))
        

def main():
    alines  = sc.textFile("/Users/atul/Google Drive/2.atul/Atul-work/spark/Marvel-Names.txt")
    names   = alines.map(parsename)
    
    blines  = sc.textFile("/Users/atul/Google Drive/2.atul/Atul-work/spark/Marvel-Graph.txt")
    partners= blines.map(countpartners)
    
    friends = partners.reduceByKey(lambda x,y:x+y)
    flipped = friends.map(lambda x: (x[1],x[0]))
    
    popular = flipped.max() ### Filter keys with max friends
    print(popular[1])
    
    popularname = names.lookup(int(popular[1])) ## Use RDD as dictionary
    print (popular,popularname)
    

if __name__=='__main__':
    conf    = SparkConf().setMaster("local").setAppName("PopularHero")
    sc      = SparkContext(conf = conf)
    main()
    sc.stop()
    print("\nDone")
    

sc.stop()
