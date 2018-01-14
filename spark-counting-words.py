## Counting words 


def normalize_words(text):
    return re.compile(r'\W+',re.UNICODE).split(text.lower())


def main():
    abook       = sc.textFile("/Users/atul/Google Drive/2.atul/Atul-work/spark/Book.txt")
    words       = abook.flatMap(normalize_words)
    counts      = words.map(lambda x:(x,1)).reduceByKey(lambda x,y: x+y)
    asorted     = counts.map(lambda x: (x[1],x[0])).sortByKey()
    results     = asorted.collect()

    
    for ares in results:
        count = str(ares[0])
        aword = ares[1].encode('ascii','ignore')
        if aword:
            print(aword,count)



if __name__=='__main__':
    conf    = SparkConf().setMaster("local").setAppName("WordCount")
    sc      = SparkContext(conf = conf)
    main()
    sc.stop()
    print("Job Done")

