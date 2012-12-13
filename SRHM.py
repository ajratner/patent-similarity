import re
import string
import csv
import nltk
from nltk.tree import Tree
from nltk.stem.porter import PorterStemmer

PATENT_DATA = "photovoltaic_05.csv"

# retrieve a single claim (first) from the patents csv file, for testing
def get_single_claim(n):
  pats_reader = csv.reader(open(PATENT_DATA, 'rb'))
  for i in range(n+3):
    pat = pats_reader.next()

  # return the claim with number deleted
  return re.sub(r'^\d+\.\s*', '', pat[5])

def first_JN_chunk(s, n):

  # chunk sequences of <JJ>*<NN>+ form
  tokens = nltk.word_tokenize(s)
  tagged = nltk.pos_tag(tokens)
  grammar = r'HNP: {<JJ|RB|VBD|VBN|NN.*>*<NN(P|S)?>+}'
  chunker = nltk.RegexpParser(grammar)
  chunk_tree = chunker.parse(tagged)

  # FOR DEV TESTING
  #print chunk_tree

  # get first chunk from tree
  for c in chunk_tree:
    if type(c) == nltk.tree.Tree:
      return [t[0] for t in c[-n:]]

  # default to simply returning the last n words
  return tokens[-n:]

def first_V_chunk(s, n):

  # chunk sequences of <V> form
  tokens = nltk.word_tokenize(s)
  tagged = nltk.pos_tag(tokens)
  grammar = r'MVP: {<VBG><DT>?<JJ.*|RB|VB(D|N)|NN.*|CD>*<NN.*>+}'
  chunker = nltk.RegexpParser(grammar)
  chunk_tree = chunker.parse(tagged)

  # FOR DEV TESTING
  #print chunk_tree

  # get first chunk from tree
  for c in chunk_tree:
    if type(c) == nltk.tree.Tree:
      if len(c) > n:
        return [c[0][0]] + [t[0] for t in c[(1-n):]]
      else:
        return [t[0] for t in c]

  # fall back to JV chunk
  grammar = r'HNP: {<JJ|RB|VBD|VBN|NN.*>*<NN(P|S)?>+}'
  chunker = nltk.RegexpParser(grammar)
  chunk_tree = chunker.parse(tagged)
  for c in chunk_tree:
    if type(c) == nltk.tree.Tree:
      return [t[0] for t in c[-n:]]
  
  # default to simply returning the last n words
  return tokens[-n:]

def get_head_words(s, nwords, ctype):
  #print ctype
  #print s
  
  # first limit to before any commas, semicolons; and remove stop list phrases
  s = re.split(r';,', s)[0]
  remove_list = r'(a\splurality\sof\s|at\sleast|composition\sof|the\ssteps\sof|wherein\s*(?:said)?|first|second|third|(?:[a-z]|\d+)?(?:\)|\.))'
  s = re.sub(remove_list, '', s)

  if ctype == 'device':
    
    # get first ~ <JJ>*<NN>+ chunk
    return first_JN_chunk(s, nwords)
  
  elif ctype == 'method':
    
    # first try to split around "method" (for first parent node)
    msplit1 = re.split(r'method\s(of|for|to)', s)
    if len(msplit1) > 1:
      return first_V_chunk(msplit1[2], nwords)
    msplit2 = re.split(r'method', s)
    if len(msplit2) > 1:
      return first_V_chunk(msplit2[0], nwords)

    # else, get first VBG + its subject if possible
    return first_V_chunk(s, nwords)

def merge_trees(trees):
  seen_nodes = []
  trees_out = []
  for tree in trees:
    if tree.node not in seen_nodes or len(tree) > 0:
      seen_nodes.append(tree.node)
      trees_out.append(tree)
  return trees_out

def hm_tree(p, nwords, split_lvl=0, ctype=''):
  splits = [r';', r',']
  split_lvl = min(split_lvl, len(splits)-1)

  # check for method claim
  if ctype == '':
    if re.search(r'method', p) > 0:
      ctype = 'method'
    else:
      ctype = 'device'
  
  # first split along parent -> [children] line
  if ctype == 'device':
    split_words = r'(compris\w+|has|having|including)'
  else:
    split_words = r'(compris\w+\sthe\ssteps\sof)'
  split_markers = r'.*?(?::|-|\s)(?:\sa\splurality\sof)?'
  parts_rgx = r'^(.*?)' + split_words + split_markers + r'(.*)$'
  parts = re.match(parts_rgx, p)
  if parts:

    # NOTE: could change which words from head chunk are selected here
    parent = get_head_words(parts.group(1), nwords, ctype)

    # then split the [children] array
    children = re.split(splits[split_lvl] + r'(?:\s*and)?', parts.group(3))
    if len(children) == 1:
      children = re.split(r'and', parts.group(3))
    return Tree(parent, merge_trees([hm_tree(child, nwords, split_lvl+1, ctype) for child in children]))
  else:
    
    # try splitting on splitters here
    # NOTE: to do later...? danger of pulling in lots of crap

    return Tree(get_head_words(p, nwords, ctype), [])



def draw_hm_tree(n):
  text = get_single_claim(n)
  hm_tree(text, 3).draw()





# --> PIPELINE STUFF...
def get_holonym_meronyms(tree):
  holonym_meronyms = {}
  currentParent = tree.node
  currentParentCopy = currentParent.strip()
  for node in tree:
    if (type(node) == nltk.tree.Tree):
      nodeDict = get_holonym_meronyms(node)
      root = node.node
      root = root.strip()
      holonym_meronyms = dict(holonym_meronyms.items() + nodeDict.items())
      if (currentParentCopy in holonym_meronyms):
        holonym_meronyms[currentParentCopy].append(root)
      else:
        holonym_meronyms[currentParentCopy] = [root]
    elif (type(node) == list):
      for listElement in node:
        elementCopy = listElement.strip()
        if (currentParentCopy in holonym_meronyms):
          holonym_meronyms[currentParentCopy].append(elementCopy)
        else:
          holonym_meronyms[currentParentCopy] = [elementCopy]
    else:
      nodeCopy = node.strip()
      if (currentParentCopy in holonym_meronyms):
        holonym_meronyms[currentParentCopy].append(nodeCopy)
      else:
        holonym_meronyms[currentParentCopy] = [nodeCopy]
  return holonym_meronyms

def get_holonym_meronym_pairs(tree):
  holonym_meronyms = get_holonym_meronyms(tree)
  newDict = {}
  for key in holonym_meronyms:
    values = holonym_meronyms[key]
    listOfKeys = key.split()
    for newKey in listOfKeys:
      if (newKey in newDict):
        newDict[newKey].extend(values)
      else:
        newDict[newKey] = values

  return newDict

#for i in xrange(20):
#  text = get_single_claim(i)
#  tree = get_tree(text)
#  print "got it"
  #print get_holonym_meronyms(tree)
  #print get_holonym_meronym_pairs(tree)
  #tree.draw()


