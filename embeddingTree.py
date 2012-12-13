from findEmbedding import *
from flattenTree import *
from abstract_SHM import *
import nltk


def generateEmbeddingTree(tree, wordVectorDir):
	phrases = flatten(tree)
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
		if (len(intermediateTrees) == 2):
			subTree = nltk.tree.Tree("", intermediateTrees)
			subTrees.append(subTree)
		elif (len(intermediateTrees) == 1):
			subTrees.append(intermediateTrees[0].node)
		elif (len(intermediateTrees) != 0):
			tempTrees = []
			subTree = nltk.tree.Tree("", [intermediateTrees[0], intermediateTrees[1]])
			tempTrees.append(subTree)
			for i in xrange(1, (len(intermediateTrees) - 1)):
				subTree = nltk.tree.Tree("", [tempTrees[-1], intermediateTrees[i+1]])
				tempTrees.append(subTree)
			subTrees.append(subTree)


	if (len(subTrees) < 2):
		return nltk.tree.Tree("", subTrees)
	else:
		tempTrees = []
		embeddingTree = nltk.tree.Tree("", [subTrees[0], subTrees[1]])
		tempTrees.append(embeddingTree)

		for i in xrange(1, len(subTrees) - 1):
			embeddingTree = nltk.tree.Tree("", [tempTrees[-1], subTrees[i+1]])
			tempTrees.append(embeddingTree)
		return embeddingTree


if __name__ == '__main__':
	abstractParser = AbstractParser()
	abstract = abstractParser.get_single_abstract(1)
	tree = abstractParser.get_tree(abstract)
	tree.draw()

	testingTree = nltk.tree.Tree("racing", [nltk.tree.Tree("wheel", []), nltk.tree.Tree("car", [])])

	embeddingTree = generateEmbeddingTree(tree, "C:\\Users\\Deepak\\Dropbox\\6.864 Project\\Word Vectors")
	embeddingTree.draw()
