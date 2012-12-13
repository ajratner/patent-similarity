import nltk

def flatten(tree):
	if len(tree) == 0:
		if (type(tree) == nltk.tree.Tree):
			return [tree.node]
		else:
			return [tree]
	if (type(tree) == nltk.tree.Tree):
		flattened_vector = [tree.node]
		for child in tree:
			flattened_vector.extend(flatten(child))
		return flattened_vector
	else:
		return [tree]

