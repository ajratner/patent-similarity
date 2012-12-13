# Helper method that gets the index from the Index file and loads into a dicitonary
def getIndex(wordVectorDir):
	#indexFilePath = wordVectorDir + "\\Index.txt"
	indexFilePath = wordVectorDir + "/Index.txt"
	indexFile = open(indexFilePath)
	indices = {}
	indexLines = indexFile.read().split('\n')
	for index in indexLines:
		indexTokens = index.split()
		if (len(indexTokens) != 2):
			continue
		key = int(indexTokens[0])
		value = indexTokens[1]
		indices[key] = value

	return indices


# Helper method that gets the index of the document which potentially contains the desired word
def findDocument(wordsIndex, word):
	startIndex = 0
	endIndex = len(wordsIndex) - 1
	midIndex = (startIndex + endIndex)/2
	found = False
	while ((startIndex <= endIndex) and (found == False)):

		if (midIndex + 1 == len(wordsIndex)):
			if (word >= wordsIndex[midIndex]):
				found = True
				return midIndex

		elif (midIndex + 1 < len(wordsIndex)):

			if (word >= wordsIndex[midIndex] and word < wordsIndex[midIndex + 1]):
				found = True
				return midIndex
			elif (word > wordsIndex[midIndex]):
				startIndex = midIndex + 1
				midIndex = (startIndex + endIndex)/2
			elif (word < wordsIndex[midIndex]):
				endIndex = midIndex - 1
				midIndex = (startIndex + endIndex)/2

	if (found == False):
		return None 	# None indicates that the word was not found in the index


def binarySearch(lines, wordToken):
	startIndex = 0
	endIndex = len(lines) - 1
	found = False

	while ((startIndex <= endIndex) and (found == False)):
		midIndex = (startIndex + endIndex) / 2
		line = lines[midIndex]

		splitLines = line.split()
		wordMidIndex = 0
		if (len(splitLines) == 0):
			return None

		wordMidIndex = line.split()[0]
		if (wordToken == wordMidIndex):
			words = line.split()
			vector = [float(words[i]) for i in xrange(1,len(words))]
			found = True
			return vector
		elif (wordToken > wordMidIndex):
			startIndex = midIndex + 1
		elif (wordToken < wordMidIndex):
			endIndex = midIndex - 1

	if (found == False):
		return None


def getVector(wordToken, wordVectorDir, wordsIndex):
	wordTokenIndex = findDocument(wordsIndex, wordToken)
	if (wordTokenIndex != None):
		#fileName = wordVectorDir + "//WordVectors_" + str(wordTokenIndex) + ".txt"
		fileName = wordVectorDir + "/WordVectors_" + str(wordTokenIndex) + ".txt"
		file = open(fileName, 'r')
		lines = file.read().split('\n')
		numLines = len(lines)

		return binarySearch(lines, wordToken)

	return None
