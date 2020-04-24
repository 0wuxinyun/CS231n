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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N=X.shape[0]
    D=X.shape[1]
    C=dW.shape[1]
    for i in range(N):
      x=X[i,:] # dim: 1,D
      score = np.dot(x,W) 
      score -= np.max(score)
      score=np.exp(score)
      denominator = np.sum(score)
      probability = score/denominator
      loss+=-np.log(probability[y[i]])
      dP=probability
      dP[y[i]]-=1
      dW += np.reshape(x.T,(D,1))*np.reshape(dP,(1,C))

    # Need normalize them
    dW/=N+reg*W
    loss/=N+0.5*reg*np.sum(W*W)

   
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N=X.shape[0]
    score=np.dot(X,W)
    score-=np.reshape(np.max(score,axis=1),(N,1))
    E_scores= np.exp(score)
    Denominator = np.reshape(np.sum(E_scores,axis=1),(N,1))
    E_pro=E_scores/Denominator
    loss =np.mean(-np.log(E_pro[np.arange(N),y]))+0.5*reg*np.sum(W*W)

    # Derivative of loss function wpt weights 
    '''
    loss rpt probability : 
          true class : probability-1
          other class: probability
    loss rpt weights :
          true class : x(probability-1)
          other class: xprobability
    '''
    dP=E_pro
    dP[np.arange(N),y]=E_pro[np.arange(N),y]-1
    dW=np.dot(X.T,dP)/N+reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
