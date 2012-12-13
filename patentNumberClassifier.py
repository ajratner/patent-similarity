import sys
import csv
import re

PATENT_DATA = "photovoltaic_05.csv"


def getIPCNumber(n):
	pats_reader = csv.reader(open(PATENT_DATA, 'rb'))
	for i in range(n+3):
		pat = pats_reader.next()

    # return the claim with number deleted
	return re.sub(r'^\d+\.\s*', '', pat[0])

def compare(list1, list2):
	for i in xrange(len(list1)):
		if (list1[i] != list2[i]):
			return i

	return 4


def convertSymbolToList(symbol):
	symbolList = []
	symbolList.append(symbol[:1])
	symbolList.append(symbol[1:3])
	symbolList.append(symbol[3:4])
	symbolList.append(symbol[4:])

	return symbolList


def compareIPCNumbers(IPCNumber1, IPCNumber2):
	symbolList1 = convertSymbolToList(IPCNumber1)
	symbolList2 = convertSymbolToList(IPCNumber2)

	return compare(symbolList1, symbolList2)

def getMetric(claim1, claim2):
	IPCNumber1 = getIPCNumber(claim1)
	IPCNumber2 = getIPCNumber(claim2)

	return compareIPCNumbers(IPCNumber1, IPCNumber2)


if __name__ == '__main__':

	for i in xrange(0,100):
		for j in xrange(i,100):
			print getMetric(i,j)

