### Machine-Learning Library
### MLLIB template
## Dataypes - Vector, LabeledPoint, Rating

import sys
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, Rating

userID = 166

def movieDict():
    adict = {}
    fh_in = open("/Users/atul/Work/spark/ml-100k/u.item",'r',encoding='ascii', errors="ignore")
    aread = fh_in.readlines()
    fh_in.close()
    
    for aline in aread:
        ent = aline.split('|')
        # print(ent)
        adict[int(ent[0])] = ent[1]
    
    return adict

def main():
    nameDict    = movieDict()
    adata       = sc.textFile("/Users/atul/Work/spark/ml-100k/u.data")
    ratings     = adata.map(lambda l: l.split()).map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))).cache()
    
    #### TEST CODE ####
    ###################
    # acount = 0
    # for i in ratings.collect():
    #     # print("Rating",i,"User:",i[0])
    #     acount +=1
        
    #     if (i[0] == userID):
    #         print("Rating",i,"User:",i[0])
    #         # break

    print("\nTraining recommendation model...")
    rank            = 10 ## Build the recommendation model using Alternating Least Squares
    numIterations   = 6  ## Lowered numIterations to ensure it works on lower-end systems
    amodel          = ALS.train(ratings, rank, numIterations)
    
    print("\nRatings for user ID " + str(userID) + ":")
    userRatings     = ratings.filter(lambda l: l[0] == userID)
    
    for rating in userRatings.collect():
        # print("Rating",rating)
        print (nameDict[int(rating[1])] + ": " + str(rating[2]))

    print("\nTop 10 recommendations:")
    recommendations = amodel.recommendProducts(userID, 10)
    for recommendation in recommendations:
        print (nameDict[int(recommendation[1])] + " score " + str(recommendation[2]))

if __name__=='__main__':
    conf    = SparkConf().setMaster("local[*]").setAppName("MovieRecommendationsALS")
    sc      = SparkContext(conf = conf)
    main()
    sc.stop()
    print("\nDone")
    
    
sc.stop()
    


