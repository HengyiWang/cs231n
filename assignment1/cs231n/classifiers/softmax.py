from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
        scores = X[i].dot(W)  # (1,C)
        scores -= np.max(scores)
        scoresProb = np.exp(scores) / np.sum(np.exp(scores))  # (1,C)

        for j in range(num_classes):

            if(j == y[i]):
                loss_partial = -np.log(scoresProb[y[i]])
                dW[:, j] += ((scoresProb[j]-1) * X[i])
            else:
                dW[:, j] += scoresProb[j] * X[i]


        loss += loss_partial

    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2*reg*W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    scores = X.dot(W)  # (N,C)
    scores_max=np.max(scores, axis=1)
    scores=scores-scores_max.reshape(scores_max.size, 1)
    correct_class_score = scores[np.arange(num_train), y]
    correct_class_score=correct_class_score.reshape(y.size, 1)
    row_sum = np.sum(np.exp(scores), axis=1)  # (N,1)
    scores_prob = np.exp(correct_class_score) / row_sum.reshape(row_sum.size, 1)  # (N,C)
    loss_partial = -np.log(scores_prob)
    loss=np.sum(loss_partial)/num_train+reg * np.sum(W * W)

    y_trueClass = np.zeros_like(scores)  # (N,C)
    y_trueClass[range(num_train), y] = 1.0
    dW = X.T.dot(np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True) - y_trueClass)
    dW /= num_train
    dW += 2*reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
