# Unfolding recursive auto-encoder as in Manning & Socher NIPS2011
# built for use with nltk.tree Trees
import numpy as np
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as l_bfgs
from nltk.tree import Tree
import nltk
import cloud

def unroll_params(params, encoded, decoded):
  W_e = np.array(params[0:encoded*decoded]).reshape((encoded, decoded))
  W_d = np.array(params[encoded*decoded:2*encoded*decoded]).reshape((decoded, encoded))
  b_e = np.array(params[2*encoded*decoded:(2*encoded*decoded + encoded)])
  b_d = np.array(params[(2*encoded*decoded + encoded):(2*encoded*decoded + encoded + decoded)])
  return W_e, W_d, b_e, b_d

# the recursive encoding function -->
def encode(W_e, b_e, x):
  return np.tanh(np.dot(W_e, x.transpose()) + b_e)

def encoding(W_e, b_e, tree):
  (m, n) = W_e.shape
  if type(tree) == nltk.tree.Tree:
    if len(tree) == n/m:
      d = [encoding(W_e, b_e, child) for child in tree]
      x_concat = np.array([di.node for di in d]).flatten()
      encoded = encode(W_e, b_e, x_concat)

      # store all a_e results in tree structure
      encoding_tree = Tree(encoded, d)
      return encoding_tree
    elif len(tree) == 0:
      return Tree(np.array(tree.node),[])
    else:
      print 'ERROR: node with ' + str(len(tree)) + ' children encountered; should have ' + str(n/m) + '.'
      #tree.draw()
      return False
  else:
    return Tree(np.array(tree),[])

# the recursive ('unfolding') decoding function -->
def decode(W_d, b_d, encoded):
  return np.tanh(np.dot(W_d, encoded.flatten()) + b_d).flatten()
  
def unfolded_decoding(W_d, b_d, tree, encoded):
  (n, m) = W_d.shape
  
  # store all a_e results in tree structure
  decoding_tree = Tree(encoded, [])
  try:
    decoding_tree.span = tree.span
  except:
    pass

  # if the given node (root) has children, decode the node's encoding, split it,
  # and use this as the children's encoding (output) to recurse back, until terminal
  # nodes are reached
  if type(tree) == nltk.tree.Tree and len(tree) > 0:
    decoded = decode(W_d, b_d, encoded)
    for i,child in enumerate(tree):

      # NOTE: the number of branchings n is NOT assumed, but that it is uniform and that
      # len(input layer) = n*len(encoding) IS assumed
      full_decoded = unfolded_decoding(W_d, b_d, child, decoded[i*m:m+(i*m)])
      decoding_tree.append(full_decoded)
    return decoding_tree
  else:
    decoding_tree = Tree(encoded, [])
    try:
      decoding_tree.span = tree.span
    except:
      pass
    return decoding_tree

# given tree, get concatenated terminal nodes
def get_concat_terminals(tree):
  if type(tree) == nltk.tree.Tree:
    if len(tree) > 0:
      out = np.array([])
      for child in tree:
        out = np.concatenate((out, get_concat_terminals(child)))
      return out
    else:
      return tree.node
  else:
    return np.array([tree])

# the up-then-down reconstruction- shortcut fn
def reconstruction(W_e, W_d, b_e, b_d, tree):
  encoding_tree = encoding(W_e, b_e, tree)
  return get_concat_terminals(unfolded_decoding(W_d, b_d, tree, encoding_tree.node))

# sum all node.gradW_d and .gradb_d of an a_d_tree
def sum_all_grad_d(tree):
  if len(tree) > 0:
    grads = [sum_all_grad_d(child) for child in tree]
    gradW_d = tree.gradW_d + np.sum([g[0] for g in grads],0)
    gradb_d = tree.gradb_d + np.sum([g[1] for g in grads],0)
    return gradW_d, gradb_d
  else:
    return tree.gradW_d, tree.gradb_d

# given a tree with an encoding root node, decoded all the way down (unfolded)
# to root terminal nodes, return the tree anotated with all node deltas,
# adding each delta contribution to gradient
def backprop_d(W_d, a_d_tree, a_e_tree):
  if len(a_d_tree) > 0:
    tree_out=Tree(a_d_tree.node, [backprop_d(W_d, child, a_e_tree[i]) for i,child in enumerate(a_d_tree)])
    delta_p = np.concatenate([child.delta for child in tree_out])

    # add contribution to gradient using current node a_d & prev. node delta
    tree_out.gradW_d = delta_p.reshape((len(delta_p.flatten()),1))*a_d_tree.node
    tree_out.gradb_d = delta_p

    # calculate this node's delta, return annotated tree
    tree_out.delta = np.dot(W_d.transpose(), delta_p)*(1 - tree_out.node**2)
    return tree_out
  else:
    tree_out = Tree(a_d_tree.node, [])
    tree_out.delta = (a_d_tree.node-get_concat_terminals(a_e_tree))*(1 - a_d_tree.node**2)
    tree_out.gradW_d = np.zeros(W_d.shape)
    tree_out.gradb_d = np.zeros(W_d.shape[0])
    return tree_out

# given a tree with an encoding root node, with all recursively encoded nodes down to
# the original terminal nodes, add each delta contribution to gradient
def backprop_e(W_e, a_e_tree, delta_p_d):
  gradW_e = np.zeros(W_e.shape)
  gradb_e = np.zeros(W_e.shape[0])
  parents = [a_e_tree]
  p_deltas = [delta_p_d]
  while len(parents) > 0:
    new_parents = []
    new_p_deltas = []
    for i,parent in enumerate(parents):
      delta_p = p_deltas[i]
      child_concat = np.concatenate([child.node for child in parent])

      # add contribution to gradient using current node a_e & prev. node delta
      gradW_e += delta_p.reshape((len(delta_p.flatten()),1))*child_concat
      gradb_e += delta_p

      # add new parents, parent deltas
      new_parents += [child for child in parent if len(child) > 0]
      new_delta_p_concat = np.dot(W_e.transpose(), delta_p)*(1 - child_concat**2)
      bf = len(parent)
      n = len(parent[0].node)
      new_p_deltas += [new_delta_p_concat[n*i:n*(i+1)] for i in range(bf)]
    parents = new_parents
    p_deltas = new_p_deltas
  return gradW_e, gradb_e

# backprop recursive function
def backprop_subtrees(W_e, W_d, b_e, b_d, tree, concat_terms):

  # STEP 1: feedforward pass up to root node of tree, and then back down fully unfolded
  encoding_tree = encoding(W_e, b_e, tree)
  decoding_tree = unfolded_decoding(W_d, b_d, tree, encoding_tree.node)

  # STEPS 2-4: calculate delta for output layer, backpropogate all the way back to input
  # layer, adding to gradient at each step
  
  # recursize algorithm for decoding side; grad contributions stored in tree out
  a_d_tree_deltas = backprop_d(W_d, decoding_tree, encoding_tree)
  gradW_d, gradb_d = sum_all_grad_d(a_d_tree_deltas)

  # non recursive: output is the grad contribution
  gradW_e, gradb_e = backprop_e(W_e, encoding_tree, a_d_tree_deltas.delta)

  # recurse to child node computations
  for child in tree:
    if type(child) == nltk.tree.Tree and len(child) > 0:
      gradW_e_i, gradW_d_i, gradb_e_i, gradb_d_i = backprop_subtrees(W_e, W_d, b_e, b_d, child, concat_terms)
      gradW_e += gradW_e_i
      gradW_d += gradW_d_i
      gradb_e += gradb_e_i
      gradb_d += gradb_d_i
  return gradW_e, gradW_d, gradb_e, gradb_d

# save params to np file
def save_params(params, n):
  param_file = open('urae_params'+str(n), 'wb')
  np.save(param_file, params)
  param_file.close()
  print 'Params saved as urae_params'+str(n)
  return True

# backpropogation through structure (BPTS) --> USING PICLOUD FOR PARRALEL PROCESSING
def Jgrad_picloud(params, encoding_size, decoded_size, training, wd, num_cores):
  W_e, W_d, b_e, b_d = unroll_params(params, encoding_size, decoded_size)
  
  gradW_e = np.zeros(W_e.shape)
  gradW_d = np.zeros(W_d.shape)
  gradb_e = np.zeros(b_e.shape)
  gradb_d = np.zeros(b_d.shape)
  
  # split the training set into batches, send out to picloud cores for backprop
  #offset = num_cores - len(training)%num_cores
  #for index in range(offset):
  # training.
  split = len(training)/num_cores

  final_training = []

  for i in range(num_cores):
    final_training.append(training[i*split:(i+1)*split])

  offset = len(training)%num_cores

  if offset > 0:
    final_training.append(training[len(training)-offset:])

  jids = cloud.map(Jgrad_picloud_sub, [params]*num_cores, [encoding_size]*num_cores, [decoded_size]*num_cores, final_training, _type='c2')

  # call for results
  results = cloud.result(jids)
  for result in results:
    gradW_e += result[0]
    gradW_d += result[1]
    gradb_e += result[2]
    gradb_d += result[3]

  # add weight decay factor and normalization coefficient
  a = 1.0/len(training)
  grad_J_W_e = a*gradW_e + wd*W_e
  grad_J_W_d = a*gradW_d + wd*W_d
  grad_J_b_e = a*gradb_e
  grad_J_b_d = a*gradb_d

  # roll up and return as 1-d array
  return np.concatenate((grad_J_W_e.flatten(), grad_J_W_d.flatten(), grad_J_b_e.flatten(), grad_J_b_d.flatten()))

# backpropogation through structure (BPTS)
def Jgrad_picloud_sub(params, encoding_size, decoded_size, training):
  W_e, W_d, b_e, b_d = unroll_params(params, encoding_size, decoded_size)
  gradW_e = np.zeros(W_e.shape)
  gradW_d = np.zeros(W_d.shape)
  gradb_e = np.zeros(b_e.shape)
  gradb_d = np.zeros(b_d.shape)

  for tree in training:
    concat_terms = get_concat_terminals(tree)
    gradW_e_i, gradW_d_i, gradb_e_i, gradb_d_i = backprop_subtrees(W_e, W_d, b_e, b_d, tree, concat_terms)
    gradW_e += gradW_e_i
    gradW_d += gradW_d_i
    gradb_e += gradb_e_i
    gradb_d += gradb_d_i

  return gradW_e, gradW_d, gradb_e, gradb_d


# backpropogation through structure (BPTS)
def Jgrad(params, encoding_size, decoded_size, training, wd):
  W_e, W_d, b_e, b_d = unroll_params(params, encoding_size, decoded_size)
  
  # for each of the training examples...
  gradW_e = np.zeros(W_e.shape)
  gradW_d = np.zeros(W_d.shape)
  gradb_e = np.zeros(b_e.shape)
  gradb_d = np.zeros(b_d.shape)
  for tree in training:
    concat_terms = get_concat_terminals(tree)

    # BPTS modified to take into account reconstruction errors also,
    # as in Socher et. al. EMNLP2011 but + unfolding all the way back
    gradW_e_i, gradW_d_i, gradb_e_i, gradb_d_i = backprop_subtrees(W_e, W_d, b_e, b_d, tree, concat_terms)
    gradW_e += gradW_e_i
    gradW_d += gradW_d_i
    gradb_e += gradb_e_i
    gradb_d += gradb_d_i

  # add weight decay factor and normalization coefficient
  a = 1.0/len(training)
  grad_J_W_e = a*gradW_e + wd*W_e
  grad_J_W_d = a*gradW_d + wd*W_d
  grad_J_b_e = a*gradb_e
  grad_J_b_d = a*gradb_d

  # roll up and return as 1-d array
  return np.concatenate((grad_J_W_e.flatten(), grad_J_W_d.flatten(), grad_J_b_e.flatten(), grad_J_b_d.flatten()))


# The cost function: computed as SUM OF unfolded reconstruction error OF ALL non-term. nodes
# return the sum of reconstruction error of root node + all immediate child nodes
def J_sub(W_e, W_d, b_e, b_d, tree):
  J_sum = 0.5*np.sum((reconstruction(W_e,W_d,b_e,b_d,tree) - get_concat_terminals(tree))**2)
  for child in tree:
    if len(child) > 0:
      J_sum += J_sub(W_e, W_d, b_e, b_d, child)
  return J_sum

# return the average cost of all training trees, with a wieght decay factor
def J(params, encoding_size, decoded_size, training, wd, num_cores=0):
  W_e, W_d, b_e, b_d = unroll_params(params, encoding_size, decoded_size)
  J_sum = 0.0
  for tree in training:
    J_sum += J_sub(W_e, W_d, b_e, b_d, tree)
  return (1.0/len(training))*J_sum + (wd/2.0)*np.sum(params[:(2*encoding_size*decoded_size)]**2)

# recursive fn for training tree data initialization
def init_tree(tree, bf, cn = 0):
  
  # if it is a tree, might not be non terminal...
  if type(tree) == nltk.tree.Tree:

    # check for correct branching factor
    if len(tree) == bf:

      # check for errors (False return) and recurse up to discard entire tree
      children = []
      for i, t in enumerate(tree):
        initialized_tree = init_tree(t, bf, cn*bf + i)
        if initialized_tree is not False:
          children.append(initialized_tree)
        else:
          return False
      out = Tree(tree.node, children)
      out.span = (out[0].span[0], out[-1].span[1])
    
    # if tree & len = 0, is a terminal node (if not NoneType node)
    elif len(tree) == 0:
      out = Tree(tree.node, [])
      try:
        out.span = (cn*len(out.node), (cn+1)*len(out.node)-1)
      except TypeError:
        print 'non-terminal ERROR: NoneType node encountered, tree discarded from training set.'
        return False
    
    # if not a terminal node or of correct branching factor, discard
    else:
      print 'non-terminal ERROR: wrong branching factor, tree discarded from training set.'
      return False

  # if it is not a tree and not None, assume it is a terminal node array, wrap in tree
  elif tree is not None:
    out = Tree(tree, [])
    out.span = ((cn-1)*len(out.node), cn*len(out.node)-1)
  
  # if None though, discard
  else:
    print 'non-terminal ERROR: terminal node ' + str(cn) + ' is NoneType, tree discarded from training set.'
    return False
  out.cn = cn
  return out

# add extra meta-data to the input nltk.tree Tree training data
def initialize_trees(training, bf):
  out = []
  for tree in training:

    # recursive init_tree algorithm can't catch single-node trees, so discard here
    if len(tree) == bf:
      initialized_tree = init_tree(tree, bf)
    else:
      initialized_tree = False
      print 'non-terminal ERROR: childless tree encountered, discarded from training set.'

    if initialized_tree is not False:
      out.append(initialized_tree)
  return out

# return the parameters as trained on the provided training set
def train_params(training_trees, wd, epsilon, picloud=True, num_cores=0, bf=2, numerical=False):
  training = initialize_trees(training_trees, bf)
  
  # get the param dimensions
  t = training[0]
  while len(t) > 0:
    t = t[0]
  encoding_size = len(t.node)
  decoded_size = bf*encoding_size
  
  # train the model with backprop + L-BFGS-B
  x0 = np.random.normal(0.0,epsilon**2,2*encoding_size*decoded_size+encoding_size+decoded_size)
  print "training model..." 
  
  if picloud:
    x, f, d = l_bfgs(J, x0, Jgrad_picloud, [encoding_size, decoded_size, training, wd, num_cores])
  elif numerical:
    x, f, d = l_bfgs(J, x0, None, [encoding_size, decoded_size, training, wd], True)
  else:
    x, f, d = l_bfgs(J, x0, Jgrad, [encoding_size, decoded_size, training, wd])
  print "minimum found, J = " + str(f)
  #save_params(x, '_final')
  return x
 






# simple test stuff
def test_tree(x):
  return Tree("a", [ Tree("aa", [Tree(x[0:4], []), Tree(x[4:8], [])]), Tree("ab", [Tree(x[8:12], []), Tree(x[12:16], [])]) ])

#test_set = [test_tree([1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0])]

test_set = [test_tree([1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0]), test_tree([1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1])]

answer_x = np.concatenate( (np.array([[10,0,0,0,0,0,0,0],[0,0,10,0,0,0,0,0],[0,0,0,0,10,0,0,0],[0,0,0,0,0,0,10,0]]).flatten(), np.array([[10,0,0,0],[10,0,0,0],[0,10,0,0],[0,10,0,0],[0,0,10,0],[0,0,10,0],[0,0,0,10],[0,0,0,10]]).flatten(), np.array([0,0,0,0]).flatten(), np.array([0,0,0,0,0,0,0,0]).flatten()) )

