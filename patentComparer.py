from uRAE_lib import *
import random
from similarityMatrix import *
from flattenTree import *
from dynamicPooling import *
from embeddingTree import *
from patentNumberClassifier import *
from SRHM import *
import math
import nltk
from datetime import datetime
import cloud

PATENT_DATA = "photovoltaic_05.csv"

def get_single_claim(n):
    pats_reader = csv.reader(open(PATENT_DATA, 'rb'))
    for i in range(n+3):
      pat = pats_reader.next()

    # return the claim with number deleted
    return (n,re.sub(r'^\d+\.\s*', '', pat[5]))

def get_random_integer(done):
	num = random.randint(1,9500)
	if num in done:
		return get_random_integer(done)
	return num

def getPatentData(numberClaims):
  f = open('training_claims_index.txt', 'wb')
  patentData = []
  done = []
  while len(done) < numberClaims:
    randomNumber = get_random_integer(done)
    done.append(randomNumber)
    claim = get_single_claim(randomNumber)
    patentData.append(claim)
    f.write(str(randomNumber)+'\n')
  f.close()
  return patentData


def partitionTrees(trees):
  totalNumber = len(trees)

  random.shuffle(trees)

  numberTraining = max(int(0.66 * totalNumber), 1)
  numberTesting = int(0.34 * totalNumber)

  print numberTraining
  print numberTesting

  trainingTrees = [tree for (x,tree) in trees[ : numberTraining]]
  testingTrees = trees[ numberTraining : ]

  return (trainingTrees, testingTrees)


def patentComparer():
  print 'Started:'
  print datetime.now()
  #wordVectorDir = "C:\\Users\\Deepak\\Dropbox\\6.864 Project\\Word Vectors"
  wordVectorDir = "/Users/aratner/Dropbox/6.864 Project/Word Vectors"

  # Get patent claims
  patentData = getPatentData(200)
  print "Got patent data..."

  # Obtain the tree from the claim using the rule based parser / machine learnt model
  trees = [(x,hm_tree(y,3)) for (x,y) in patentData]
  embeddingTrees = [(x,generateEmbeddingTree(tree, wordVectorDir)) for (x,tree) in trees]

  #print embeddingTrees
  #for embeddingTree in embeddingTrees:
  #  embeddingTree[1].draw()

  print "Generated trees for claims..."

  # Partition trees into two sets - a training set and a testing set
  (trainingTrees, testingTrees) = partitionTrees(embeddingTrees)

  # Run the unfolding recursive auto-encoder on the tree, using picloud
  #params = train_params(trainingTrees, math.pow(10,-5), 0.01, False)
  jid = cloud.call(train_params, trainingTrees, math.pow(10,-5), 0.01, True, 60, _type='c2')
  print 'jid = ' + str(jid)
  params = cloud.result(jid)
  save_params(params, '_finalpicloud')
  print 'RAE training complete, parameters saved, at:'
  print datetime.now()

  print "Done..."



if __name__ == '__main__':
	patentComparer()

