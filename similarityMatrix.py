import numpy as np

def computeSimilarityMatrix(phraseVectors1, phraseVectors2):
	output = []
	for phraseVector1 in phraseVectors1:
		row = []
		for phraseVector2 in phraseVectors2:
			distance = euclideanDistance(phraseVector1, phraseVector2)
			row.append(distance)
		output.append(row)

	numpyOutput = np.array(output)
	return numpyOutput


def euclideanDistance(phraseVector1, phraseVector2):
	distance = 0
	if (len(phraseVector1) == len(phraseVector2)):
		for i in xrange(len(phraseVector1)):
			distance = distance + (phraseVector1[i] - phraseVector2[i])**2

	distance = distance ** 0.5
	return distance

