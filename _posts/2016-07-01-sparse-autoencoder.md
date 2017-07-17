---
layout: post
title:  "Sparse Autoencoder in theano"
excerpt: "I'll discuss about how to code up a Sparse(input) Autoencoder in theano"
comments: true
mathjax: true
---

Last month I was reading about Autoencoders for Collaborative filtering. Using Autoencoders for Collaborative filtering is a fairly recent idea and it proven to be very effective beating all state-of-art SVD based methods. Actually the idea is pretty simple, instead of using linear function for Matrix Factorization (dot product of the two latent factor matrices, use some complex non-linear function of the two matrices so as to capture more complicated dependencies. Autoencoders are a way to do so. 

Std. Matrix Factorization based CF minimizes the following objective:


$$
\begin{align}
\min_{W, H} \Vert X - WH \Vert _O ^2  
\end{align}
$$

Note that in the objective function only known/observed user-rating pairs (set O) are considered. This is a challenge in CF methods. X is usually very sparse, only few user-rating entries are observed. 

As we can observe the function which is used to approximate X is a linear function in latent factor matrices W and H. Now to capture more complicated depdencies some non-linear function of W and H needs to be used. This is where Autoencoder comes into picture. An Autoencoder solves the following objective function :

$$
\begin{align}
\min_{W, H} \Vert X - f(W(g(Vr + mu) + b)) \Vert _O ^2  
\end{align}
$$

W and V are latent factors, b and mu are biases and r is the user vector (rth row from user-item matrix). Now f and g are non-linear functions. This will capture some complicated non-linear dependencies between latent factors and input. Note that this objective is also defined only on the known values in the user-item matrix. This makes the implementation hard since we are dealing with a sparse input matrix. 

To implement this in theano, there are two options:

1. Use sparse tensor data type in theano: Theano provides a sparse tensor datatype to deal with sparse inputs, but theano cannot port sparse operations to GPU, so this method will not be scalable.

2. Sparse operations: We can pass full data matrix to theano but handle sparse objective part during computation. 

I used the second part to code this up in theano to create a scalable system. I will be using one other important functionality in theano, the scan op. This is very useful in case when vectorization is not possible or iteration over data is required. Using a for loop for computation is a very bad idea in theano. Theano will not be able to optimize the computation graph in such case. Scan is used in such cases.

Let's get start with the implementation. We will use scipy's sparse matrix to store input data. 


```python
# Input : T  - input sparse matrix
# collecting all non-zero indices (known values)
nz_ind = T.nonzero()
# Splitting data into 80-20 training-testing
NZ = np.vstack((nz_ind[0], nz_ind[1])).T
np.random.shuffle(NZ)
train_ind = NZ[:0.8*len(NZ)]
test_ind = NZ[0.8*len(NZ)+1:]
t = T.tolil()
t[self.test_ind[:, 0], self.test_ind[:,1]] = 0
n = self.T.shape[0]
r = self.T.shape[1]

# Theano model
# Defining shared variables (model paramters)
w = np.random.uniform(low=- np.sqrt(6 / float(self.n + self.k )),
                              high= np.sqrt(6 / float(self.n + self.k)),
                              size=(self.n, self.k)).astype(np.float32)
v = np.random.rand(self.k, self.r).astype(np.float32)
MU = np.zeros((self.k)).astype(np.float32)
B = np.zeros((self.n)).astype(np.float32)
W = theano.shared(w, name='W', borrow=True)
V = theano.shared(v, name='V', borrow=True)
mu = theano.shared(MU, name='mu', borrow=True)
b = theano.shared(B, name='b', borrow=True)
                      
param = [W, V, mu, b]
# Theano matrix which will accept input from training function
rating = T.matrix()

def step(rat, W, V, mu, b):
            # Function applied at each step of scan
            # find all non-zero indices from index (observed values)
            res = T.zeros_like(rat)
            rat_nz = T.neq(rat, 0).nonzero()[0]
            tar = rat[rat_nz]
	    # non-linear g
            hidden_activation = T.tanh(T.dot(V[:, rat_nz], rat[rat_nz]) #\
                                       + mu)
	    # Non-linear function f applied	
            output_activation = T.nnet.sigmoid(T.dot(W[rat_nz, :], \
                                             hidden_activation) \
                                       + b[rat_nz])
            res = T.set_subtensor(res[rat_nz] ,output_activation)
            return res


# Thaeno function which will call the step function for each row of the input matrix.          
scan_res, scan_updates = theano.scan(fn=step, outputs_info=None, \
                                             sequences=[rating],
                                             non_sequences=[W, V, \
                                                            mu, b])

# Defining the loss function 
self.loss = T.sum((scan_res - rating) ** 2) + \
            0.1 * T.sum(W ** 2) + 0.1 * T.sum(V ** 2)
        
updates_sgd = sgd(self.loss, self.param, learning_rate=lr)
ae_batch = theano.function([rating], self.loss, updates=updates_sgd)

```

This is the model part written in theano. Scan might look typical at first look, but essentially it's doing iteration over rows of ratings (sequences part) and updating paramters using only known values. 

Goal of using scan is that during computation in step those indices of the paramters are used whose values are known in the input matrix. (V[:, rat_nz] and W[rat_nz,:]). So the hope is that theano will recognize this and during update it will only update these rows of W and V only.

Training part for this model is pretty simple. We have to pass mini-batches (subset of rows to this model and it will update the paramters accordingly. 

```python
T = T.tocsr()
nonzero_indices = T.nonzero()

n_users = len(np.unique(nonzero_indices[0]))
indices = np.unique(nonzero_indices[0])
for epoch in xrange(self.epochs):
    l = []
    for ind, i in enumerate(xrange(0, n_users, batch_size)):
	ratings = T[indices[i:(i + batch_size)], :].toarray().astype(np.float32)
	loss = self.AE.ae_batch(ratings)
	l.append(loss)
    m = np.mean(np.array(l))
    print("mean Loss for epoch %d  batch %d is %f"%(epoch, ind, m))
```
This is how you can implement a sparse autoencoder (or perhaps any sparse input based method in theano if you want to use GPU power).


