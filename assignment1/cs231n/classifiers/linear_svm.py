import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        
        # Compute gradients (one inner and one outer sum)
        # Wonderfully compact and hard to read
        dW[:,y[i]] -= X[i,:] # this is really a sum over j != y_i
        dW[:,j] += X[i,:] # sums each contribution of the x_i's

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # compute scores
  scores = np.dot(X, W)
  
  # compute correct_class_score
  # correct_class_score = scores[np.arange(X.shape[0]), y].reshape((X.shape[0], -1))
  correct_class_score = []
  for i in xrange(X.shape[0]):
    correct_class_score.append(scores[i, y[i]])
  correct_class_score = np.array(correct_class_score).reshape((X.shape[0], -1))

  new_scores = scores - correct_class_score + 1.0
  new_scores[np.arange(X.shape[0]), y] = 0
  new_scores = np.maximum(new_scores, 0)
    
  loss = np.sum(new_scores) / X.shape[0] + 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  # make a binary matrix
  binary_scores = np.array(new_scores != 0, dtype = np.float32)
  
  # sum them column wise. This gets us the # of times we want to change for dWj
  binary_score_col = np.sum(binary_scores, axis = 1)
  binary_scores[np.arange(X.shape[0]), y] = -binary_score_col
  
  dW = X.T.dot(binary_scores) / X.shape[0] + reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
