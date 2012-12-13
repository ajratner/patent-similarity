import numpy as np
import sys

# Takes in a similarity matrix in the form of a double dimensional numpy array
# Size represents the size of the reduced matrix
# Returns the output in the form of a double dimensional numpy array

def upSample(similarityMatrix, size):
  (dimensionX, dimensionY) = similarityMatrix.shape


  numRepeatX = 0
  numRepeatY = 0
  similarityMatrixCopy = []
  similarityMatrixDoubleCopy = []
  print dimensionX,dimensionY
  if (size > dimensionY):
    numRepeatX = size / dimensionY + 1

    for i in xrange(dimensionX):
      row = []
      for j in xrange(dimensionY):
        for k in xrange(numRepeatX):
          row.append(similarityMatrix[i][j])
      similarityMatrixCopy.append(row)
  else:
    similarityMatrixCopy = similarityMatrix



  numpyMatrix = np.array(similarityMatrixCopy)
  (dimensionX, dimensionY) = numpyMatrix.shape
      
  if (size > dimensionX):
    numRepeatY = size / dimensionX + 1
    
    for i in xrange(dimensionX):
      row = numpyMatrix[i]
      for k in xrange(numRepeatY):
        similarityMatrixDoubleCopy.append(row)
  else:
    similarityMatrixDoubleCopy = similarityMatrix

  finalArray = np.array(similarityMatrixDoubleCopy)
  return finalArray

def dynamicPooling(similarityMatrix, size):
  (dimensionX, dimensionY) = similarityMatrix.shape

  spanningRegionX = 0
  spanningRegionY = 0

  numRepeatX = 0
  numRepeatY = 0

  similarityMatrix = upSample(similarityMatrix, size)

  if (dimensionX % size ==0):
     spanningRegionX = dimensionX / size
  else:
     spanningRegionX = dimensionX / size + 1

  if (dimensionY % size == 0):
     spanningRegionY = dimensionY / size
  else:
     spanningRegionY = dimensionY / size + 1
  
  output = []

  for i in xrange(size):
    row = []
    for j in xrange(size):
      minimumElement = sys.maxint
      for x in xrange(spanningRegionX):
        for y in xrange(spanningRegionY):
          if (i+x < dimensionX and j+y < dimensionY):
            if (minimumElement > similarityMatrix[i+x][j+y]):
              minimumElement = similarityMatrix[i+x][j+y]
      row.append(minimumElement)
    output.append(row)

  numpyOutput = np.array(output)

  return numpyOutput



similarityMatrix = np.array([[1,2,3],[4,5,6],[7,8,9]])
dynamicPooling(similarityMatrix, 10)
