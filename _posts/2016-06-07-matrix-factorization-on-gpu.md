---
layout: post
title:  "Matrix factorization on GPU"
excerpt: "I'll discuss briefly about popular mathematical model used in IR and CV community, matrix factorization"
comments: true
mathjax: true
---

In this post I will try to describe Matrix factorization methods from Machine Learning on which I am working on currently with it's implementation in theano.

Matrix factorization is a popular mathematical/machine learning model used in Image processing and Text mining field. The idea is to find latent factors from the data using some mathematical formulation. Formally the problem is,  For a given data matrix X of dimension n * n the goal of matrix factorization is to recover two matrices W and H of dimension n * r, r*n respectively such that reconstruction error between W * H and X is minimum. Mathematically it can be stated as following:

$$
\begin{align}
\min_{W, H} \Vert X - WH \Vert _F ^2  
\end{align}
$$

Goal is to find W and H which minimizes the given objective function. This is non-convex optimization objective. I have to admit I could not figure out the non-convex part of the objective in the first place. At first it appeared convex, because standard textbook knowledge tells us  norms are convex function. But then I read it's proof somewhere. To proof convexity of a function, we have to proof that the hessian of the function is PSD (Positive Semi Definite). I won't go into it's proof. There is a [stackoverflow](http://math.stackexchange.com/questions/393447/why-does-the-non-negative-matrix-factorization-problem-non-convex) question for the same. You can refer that for proof. 

Now that we have realised the problem is non-convex we cannot use of-the-shelf convex optimization packages which are nice and fast. We have to resort to Gradient Descent mathods which are popularly used in such cases. Before starting with a solution I'll write the full objective function :

$$
\begin{align}
\min_{W, H} \Vert X - WH \Vert _F ^2 + \lambda_1 \Vert X \Vert _2 ^2 + \lambda_2 \Vert H \Vert _2 ^2 
\end{align}
$$

Now to solve this objective using gradient descent method we have to calcualte gradient of this function w.r.t each of the parameters by hand and feed it to a program. In this case the function is relatively simpler, but it can be pretty complex too. Workaround to this problem is to use automatic differentiation methods. In python theano uses automatic differentiation to compute gradient of the function analytically. 

Now this problem can also serve as good starting point for learning theano as well because the objective function is quite simple in this case, and it will also teach you how to get use theano for solving common/novel machine learning tasks, because in the end almost all machine learning algorithms can be posed as an optimization problem.

Now in all machine learning algorithms there are essentially two types of variables : model parameters and input variables. In theano model parameters are defined using shared variables and input variables are defined as theano tensors. In our problem X is input variable and W and H are model paramters. Let's define them in theano:

```python
X = theano.tensor.matrix()
W = np.random.rand(n, r)
H = np.random.rand(r, n)
# The above definition creates two numpy arrays, we need to convert them to shared variables
W = theano.shared(W)
H = theano.shared(H)
```

We are initializing W and H randomly because we have no prior knowledge about them. Now lets compute the gradients

```python
cost = (theano.tensor.sum(X - theano.tensor.dot(W, H) ** 2)) + lambda1 * (theano.tensor.sum(W ** 2)) + lambda2 * (theano.tensor.sum(H ** 2))
grad = theano.tensor.grad(cost, wrt=[W,H]) # this is where the magic happens
# We can also define update equations in theano which can be applied to update the parameters (gradient descent update rule)
updates = [(param, param - beta * (param_grad / T.sqrt(T.sum(param_grad **2)))) for param, param_grad in zip(params, grad)] # Param can be defined as param = [W, H]
```

Update part might appear different due to normalization involved in it. The reason for normalizing is that gradients of L2 loss functions explodes after few epochs. So to get around that we can normalize the gradient. It will not affect the direction of parameter update because we are converting gradient vector into a unit vector whose direction is still the same, only the magnitude has changed. So paramters will be updated will less magnitude. This can be compensated by increasing the learning rate.

These equations define a computational graph. Computational graph means graph representation of the mathematical equations with nodes are variables and edges are mathematical operators. Theano constructs an optimized computational graph and performs backpropagation on this graph to compute gradients. Now the term backpropagation in this context can be confusiong, because this term is more popularly used in the context of Neural network. But backpropagation is more generic. Basically calculation of derivatives in any computational graph is backprop. After optimizing the computational graph theano generates an equivalent C code from it with GPU support. This is the reason "compiling" phase in theano is pretty slow. 
 
But until now all these computations are abstract, there is no interface to plug real data into it. Theano's 'function' are used for this purpose. It gives a gateway to supply input data to this computation graph and perform optimization updates. The code for same is :

```python
mf = theano.function(inputs=[X], outputs=cost, updates=updates)
```

Now to perform optimization this function can be called. 

```python
for i in xrange(n_iter):
    mf(X)
```

After some iterations the cost will converge (stabalize) and the parameters will settle to their respective minimum values. But these values are not gareented to be global minimum, because of non-convex nature of the problem. One heuristic which is commonly used in such cases is to run the routine multiple times with different random initial values and check which value gives the minimum cost and take that value as final parameters. 

There is an interesting extension to this problem popularly known as Sparse Coding in Machine Learning literature. The idea in Sparse Coding is to learn a sparse latent matrix W and H. This is essentially done to prevent W and H to be almost equal to X and to learn interesting features under the sparsity constraints. But one natural question arises, why would enforcing sparsity lead to model learning 'intersting' features. The idea is that because W and H is sparse network is forces to learn only important latent factors from the training data, it cannot learn everything because it can encode less information. So this leads to interesting feature extraction algorithm from data. Mathematically the objective doesn't requires much change. We have to only add one extra constraint - L1 norms of matrices in the objective. 

This concludes the discussion on matrix factorization using theano. Feel free to give your comments on it. 



