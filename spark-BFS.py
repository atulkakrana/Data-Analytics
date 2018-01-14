### Breadth-First-Search Algorithm
##########################

startCharacterID    = 5306      ## SpiderMan
targetCharacterID   = 14        ## ADAM 3,031 (who?)

hitCounter          = sc.accumulator(0)  ## Our accumulator, used to signal when we find the target character during our BFS traversal.

def convertToBFS(line):
    fields = line.split()
    heroID = int(fields[0])
    connections = []
    
    for connection in fields[1:]:
        connections.append(int(connection))

    color = 'WHITE'
    distance = 9999

    if (heroID == startCharacterID):
        color       = 'GRAY'
        distance    = 0

    return (heroID, (connections, distance, color))
    
def bfsMap(node):
    characterID = node[0]
    data        = node[1]
    connections = data[0]
    distance    = data[1]
    color       = data[2]

    results     = []

    #If this node needs to be expanded...
    if (color == 'GRAY'):
        for connection in connections:
            newCharacterID  = connection
            newDistance     = distance + 1
            newColor        = 'GRAY'
            if (targetCharacterID == connection):
                hitCounter.add(1)

            newEntry = (newCharacterID, ([], newDistance, newColor))
            results.append(newEntry)

        #We've processed this node, so color it black
        color = 'BLACK'

    #Emit the input node so we don't lose it.
    results.append( (characterID, (connections, distance, color)) )
    return results

def bfsReduce(data1, data2):
    edges1      = data1[0]
    edges2      = data2[0]
    distance1   = data1[1]
    distance2   = data2[1]
    color1      = data1[2]
    color2      = data2[2]

    distance    = 9999
    color       = color1
    edges       = []

    # See if one is the original node with its connections.
    # If so preserve them.
    if (len(edges1) > 0):
        edges.extend(edges1)
    if (len(edges2) > 0):
        edges.extend(edges2)

    # Preserve minimum distance
    if (distance1 < distance):
        distance = distance1

    if (distance2 < distance):
        distance = distance2

    # Preserve darkest color
    if (color1 == 'WHITE' and (color2 == 'GRAY' or color2 == 'BLACK')):
        color = color2

    if (color1 == 'GRAY' and color2 == 'BLACK'):
        color = color2

    if (color2 == 'WHITE' and (color1 == 'GRAY' or color1 == 'BLACK')):
        color = color1

    if (color2 == 'GRAY' and color1 == 'BLACK'):
        color = color1

    return (edges, distance, color)

def main():
    atext               = sc.textFile("/Users/atul/Google Drive/2.atul/Atul-work/spark/Marvel-Graph.txt")
    iterationRdd        = atext.map(convertToBFS)
    
    for iteration in range(0, 10):
        print("Running BFS iteration# " + str(iteration+1))
    
        # Create new vertices as needed to darken or reduce distances in the
        # reduce stage. If we encounter the node we're looking for as a GRAY
        # node, increment our accumulator to signal that we're done.
        mapped  = iterationRdd.flatMap(bfsMap)
        results = mapped.collect()
        
        ## Just to evaluate the above function
        print("Processing " + str(mapped.count()) + " values.")
    
        ## If boolean value for hit counter is changed, we are done
        if (hitCounter.value > 0):
            print("Hit the target character! From " + str(hitCounter.value) \
                + " different direction(s).")
            break
        
        iterationRdd = mapped.reduceByKey(bfsReduce)

if __name__=='__main__':
    conf    = SparkConf().setMaster("local").setAppName("PopularHero")
    sc      = SparkContext(conf = conf)
    main()
    sc.stop()
    print("\nDone")
    
    
sc.stop()