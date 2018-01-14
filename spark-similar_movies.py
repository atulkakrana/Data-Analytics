### Movie-pairs
####################

import sys,time
from pyspark import SparkConf, SparkContext
from math import sqrt

## Settings
scoreThreshold          = 0.97
coOccurenceThreshold    = 50
movieID                 = 50
simstrengthCut          = 200


def loadMovieNames():
    '''
    Makes a dict of movie ids and names
    '''
    movieNames = {}
    with open("/Users/atul/Work/spark/ml-100k/u.item", encoding='ascii', errors='ignore') as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

def filterDuplicates( userRatings ):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return movie1 < movie2
    
def makePairs(userRatings):
    ratings = userRatings[1]
    (movie1, rating1) = ratings[0]
    (movie2, rating2) = ratings[1]
    return ((movie1, movie2), (rating1, rating2))

def computeCosineSimilarity(ratingPairs):
    numPairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in ratingPairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        numPairs += 1

    numerator = sum_xy
    denominator = sqrt(sum_xx) * sqrt(sum_yy)

    score = 0
    if (denominator):
        score = (numerator / (float(denominator)))

    return (score, numPairs)

def main():
    print("\nLoading movie names...")
    nameDict = loadMovieNames()
    print("Movie ID and Name dict made")
    time.sleep(3)
    
    ### Make Key-value pairs
    data    = sc.textFile("/Users/atul/Work/spark/ml-100k/u.data")
    ratings = data.map(lambda l: l.split()).map(lambda l: (int(l[0]), (int(l[1]), float(l[2])))) # Map ratings to key / value pairs: user ID => movie ID, rating
    print("Key-Value pairs made for data")
    time.sleep(3)
    
    ### 
    joinedRatings       = ratings.join(ratings) ## Put all pairs into a list for each user; userID => ((movieID, rating), (movieID, rating))
    uniqueJoinedRatings = joinedRatings.filter(filterDuplicates) # Filter out duplicate pairs
    print("Pairs joined and uniqued for user")
    time.sleep(3)

    moviePairs          = uniqueJoinedRatings.map(makePairs)    ## We now have (movie1, movie2) => (rating1, rating2);
    moviePairRatings    = moviePairs.groupByKey()               ## Now collect all ratings for each movie pair and compute similarity
    print("Pairs joined and uniqued for user")
    time.sleep(3)
    
    # We now have (movie1, movie2) = > (rating1, rating2), (rating1, rating2) ...
    # Can now compute similarities.
    moviePairSimilarities = moviePairRatings.mapValues(computeCosineSimilarity).cache()
    
    filteredResults = moviePairSimilarities.filter(lambda pairSim: (pairSim[0][0] == movieID or pairSim[0][1] == movieID) and pairSim[1][0] > scoreThreshold and pairSim[1][1] > coOccurenceThreshold)
    results         = filteredResults.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(ascending = False).take(10)     # Sort by quality score
    
    print("Top 10 similar movies for " + nameDict[movieID])
    for result in results:
        (sim, pair) = result
        # Display the similarity result that isn't the movie we're looking at
        similarMovieID = pair[0]
        if (similarMovieID == movieID):
            similarMovieID = pair[1]
            
        print(nameDict[similarMovieID] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))



if __name__=='__main__':
    conf    = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities")
    sc      = SparkContext(conf = conf)
    main()
    sc.stop()
    print("\nDone")


sc.stop()