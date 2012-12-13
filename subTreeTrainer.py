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

def generatePhraseTree(tree, wordVectorDir):
	phrases = flatten(tree)
	#print phrases
	# print phrases
	index = getIndex(wordVectorDir)

	subTrees = []
	for phrase in phrases:

		vectors = []
		for wordInPhrase in phrase:
			vector = getVector(wordInPhrase, wordVectorDir, index)
			if (vector != None):
				vectors.append(vector)
				#vectors.append(wordInPhrase)

		intermediateTrees = [nltk.tree.Tree(vector, []) for vector in vectors]
		#print len(intermediateTrees)
		if (len(intermediateTrees) == 2):
			subTree = nltk.tree.Tree("", intermediateTrees)
			subTrees.append(subTree)
		elif (len(intermediateTrees) == 1):
			subTrees.append(Tree(intermediateTrees[0].node, []))
		# elif (len(intermediateTrees) ==3):
		# 	subTree = nltk.tree.Tree("", [intermediateTrees[0], nltk.tree.Tree(intermediateTrees[1], intermediateTrees[2])])
		# 	subTrees.append(subTree)
		# elif (len(intermediateTrees) == 4):
		# 	subTree = nltk.tree.Tree("", [nltk.tree.Tree(intermediateTrees[0], intermediateTrees[1]), nltk.tree.Tree(intermediateTrees[2], intermediateTrees[3])])
		# 	subTrees.append(subTree)
		elif (len(intermediateTrees) != 0):
			tempTrees = []
			subTree = nltk.tree.Tree("", [intermediateTrees[0], intermediateTrees[1]])
			tempTrees.append(subTree)
			for i in xrange(1, (len(intermediateTrees) - 1)):
				subTree = nltk.tree.Tree("", [tempTrees[-1], intermediateTrees[i+1]])
				tempTrees.append(subTree)
			subTrees.append(subTree)

	return subTrees



def trainIntermediateRAE():
  print 'Started:'
  print datetime.now()
  #wordVectorDir = "C:\\Users\\Deepak\\Dropbox\\6.864 Project\\Word Vectors"
  wordVectorDir = "/Users/aratner/Dropbox/6.864 Project/Word Vectors"

  # Get patent claims
  patentData = getPatentData(500)
  print "Got patent data..."

  # Obtain the tree from the claim using the rule based parser / machine learnt model
  trees = [(x,hm_tree(y,3)) for (x,y) in patentData]
  totalTrainingdata = []
  for treeData in trees:
  	tree = treeData[1]
  	#tree.draw()
  	label = treeData[0]
  	totalTrainingdata.extend(generatePhraseTree(tree, wordVectorDir))

  print "Generated trees for claims..."

  # Partition trees into two sets - a training set and a testing set
  #(trainingTrees, testingTrees) = partitionTrees(totalTrainingdata)
  trainingTrees = totalTrainingdata
  testingTrees = totalTrainingdata

  #for tree in trainingTrees:
  # 	if (len(tree) > 1):
  #		tree.draw()

  # Run the unfolding recursive auto-encoder on the tree
  jid = cloud.call(train_params, trainingTrees, math.pow(10,-5), 0.01, True, 60, _type='c2')
  print 'jid ='+str(jid)
  params = cloud.result(jid)
  save_params(params, '_finalpicloud_subtrees')
  print 'RAE training complete, at:'
  print datetime.now()


  print "Done..."

if __name__ == '__main__':
	trainIntermediateRAE()
