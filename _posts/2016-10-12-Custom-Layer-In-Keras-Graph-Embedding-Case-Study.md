---
layout: post
title:  "Writing a custom model in Keras with a demonstration on Graph Embedding problem"
excerpt: "How can you write a custom model in keras with an concrete example for Graph Embedding problem"
comments: true
mathjax: true
---

I started of writing Neural Nets in theano, which is a great library for writing Machine Learning algorithms. Just to give some intro.,  it is basically a math. experession compiler which takes any arbitary mathematical expression as input and produces a low level (optimized) C/C++ code which can run on GPU (if you have one). And since Deep Nets. are function approximation machines, in the end they boil down to a complicated mathematical expression written to model a real world process and on top of which you can run your favourite optimization algorithm to fit the data. 

It's a great tool as it heavylifts the messy GPU stuff, but still it has all the nuisances of writing your model from scratch everytime. This is not desirable since if you are using some standard Neural Net (say CNN, RNN, Autoencoder), you don't want to write it from scratch every damn time. One thing which immediately comes to your mind when you think for a solution to it is - Abstraction. Since some Neural Nets. are standard, someone should write an API for them, which you can call and use 'as is'.

It took some time for developers to come up with a wrapper for theano, and in around 2014 or so one ML engineer from Google François Chollet developed a great project - 'Keras'. It is basically the same thing which I described above, i.e. it abstracts low level details from theano, and lets you define your Neural net with very few lines of code (with std. Neural net building blocks like- Convolution, RNN, LSTM etc. defined as APIs) and in a much 'cleaner' way. It's a great library and a great effort by 'François Chollet' and anyone interested in Deep Learning should check it out. 

Although keras has lot of functionalities and the author has tried to include as many Neural Net APIs in the library itself, it may happen that you have some model which is a mix of standard NN models, plus some of your custom stuff. It turns out that Keras provides a way to define your model in Keras's environment itself. You don't need to deal with the backend in this case also. I think this is an incredibly useful feature of Keras. I faced a similar situtation when I had to code up a custom Graph Embedding model. I was little familiar with theano, and I thought it would look cool if I say that I coded up a new model from scratch. Turns out it was a very bad idea. Coding up from scratch in painfully slow and a buggy process in theano. You will be looking at error messages with super cryptic meaning, you will loose big bunch of your hair in finding out where the error occured half of the time. It tool me 4 times more time than I anticipated. I thought it's high time I learn how to write a custom layer in Keras. 

It turns out there are very few articles which explain in detail (with example) on writing a custom layer in keras. In this blog post, I will describe how to write a custom layer with all the details you need to know when you are writing your own layer. 

First I will give a brief description of the model for which I will be writing a custom layer. So the model deals with embedding nodes of a Graph in a low dimensional space such that network structure is preserved. Or to put it in another way - The model finds the feature vector of every node in the graph automatically, such that these feature vectors or representations of nodes which are closer in graph are closer in their space and nodes which are far in the graph are also far in the continuous space defined by the feature vector.

Given a graph G = (V, E) with nodes (n1, n2, ...., nM), mathematically the model can be defined as :

$$ 
\begin{align}
h_* = u \odot v \\
h_+ = \Vert u - v \Vert \\
h = tanh(W_* h_* + W_+ h_+ + b)\\
y = softmax(U h + c)\\
\end{align}
$$

where u and v are d-dimensional vector representations of nodes u and v. W's are n_h * d dimensional linear transformation matrices for feature h_* and h_+ (n_h is the hidden layer dimension which maps a d dimensional vector to a h dimensional vector). h_* and h_+ in one way encodes the angle and distance between the two feature vectors (representation) u and v [1]. U is a 2 * n_h dimensional matrix which transforms n_h hidden layer's feature vector into a scalar which is passed through softmax for probability distribution over labels 0 and 1. y is a distribution over labels in graph 0 and 1, where label 0 indicates that there is no edge between u and v and label 1 indicates otherwise. Training of the model is performed using 1st order methods (Gradient Descent based methods) on log-likelihood of y. During training u and v are also treated as parameters (which are initialized randonly as other paramters of the model) and are trained which optimizing for loss of the model. The values of u and v after training converges is used as embedding for the corresponding nodes. These embeddings can be used for any donwnstream task in graph mining, like link prediction, community detection, outlier detection etc. 

Now to code this model in keras we need to define a seperate layer for this model. Lets start defining this layer.

```python
# First we will import the abstract class 'Layer' which every custom layer's class should implement
from keras.engine import Layer
# Now since our model has 'trainable' parameters, we need to import module which deals with initialization of them
from keras import initialization
# import the keras backend module which deals with backend in a rather abstract manner
# Import other std. modules also which will deal with optimization, adding Fully Conntected layers for classification etc.
from keras.models import Model
from keras.layers import Lambda, Flatten, Dense
from keras.layers.core import Reshape, Permute, Dropout
from keras.layers.embedding import Embedding
from keras.models import Sequential
```

All initializations are done, now lets dig into the layer definition

```python
# Define a class which will wrap all the model details
# It should inherit the abstract parent class 'Layer' which is the parent class for all layers in Keras
class NodeEmbeddingLayer(Layer):
	# Constructor for this class with an argument for hidden layer's dimensionality which will be passed by user
	def __init__(self, hidden_dim, **kwargs):
		self.hidden_dim = hidden_dim
		# Define an inititialization method which can be called upon for for initializing model parameters
		self.init = initializations.get('glorot_uniform')
		# Call parent class's constructor which handle some details we don't need to worry about
		super(NodeEmbeddingLayer, self).__init__(**kwargs)
```

This is the constructor part with details annotated as comments. Now let's define the funciton which handles parameter declaration and initialization part

```python
	# Function that deals with parameter related stuff, takes one argument, input_shape is usually (batch_size, input_length, input_dimension) but due to some errors I faced d	   # -uring defining model equations I am using input_shape as (batch_size, input_dimension, input_sequence_length)
	def build(self, input_shape):
		# Get the embedding dimension, which is the second entry in the tuple input_shape
		embed_dim = input_shape[1]
		# Initialize the linear transformation paramter for 'distance' feature | u - v|
		# We will use the initialization function 'self.init' defined in the constructor of this class
		# We will define it of dimension (d, n_h) (reason explained below)
		self..W_p = self.init((self.embed_dim, self.hidden_dim))
		# angle feature's transformation matrix
		self.W_p = self.init((embed_dim, self.hidden_dim))
		# the bias parameter (dimension = n_h)
		self.b = K.zeros((self.hidden_dim,))

```

Now our parameters are defined, let's define the function which contains the 'logic', the model equations. Your main logic lies in this module for every custom layer 

```python
	# This function is called after to execute the forward prop. of your model and also all gradients are computed w.r.t the equations defined here
	# Arguments are, input x, which is output of the layer previous to this layer in the overall model. mask is something we will ignore at this moment
	def call(self, x, mask=None):
		# Now as explained before, our inout is of dimension (batch_size, Embedding_dimension, input_sequence_length
		# Now since this model predicts existence of label between a pair of node, input sequence length is just 2
		# We present input to the overall model as pairs of pairs with corresponding labels (node1_index, node2_index, label)
		# We need to convert node_index to a continuous vector (embedding), Embedding layer in Keras handles that. Will come to that part later
		# Lets define the angle feature (u * v) 
		X = x[:, :, 0] * x[:, :, 1]
		Y = K.abs(x[:, :, 0] - x[:, :, 1]
		# At this point X and Y contains the angle and distance feature vector for whole batch (remember we pass input in batches)
		# Now lets return the linear combination of these features weighted by paramters defined above
		z = K.tanh(K.dot(X, self.W_p) + K.dot(Y, self.W_m) + self.b)
		# I was facing some issues when I was transposing X. Ideally equations should be K.dot(W_p, X.T)
		return z

	# If we are changing input dimensions we need to tell keras that we are doing so and what the output dimension is
	def get_output_shape_for(self, input_shape):
		return (input_shape[0], self.hidden_dim)

```

Now our mail logic is done, we need to stack fully connected network on top of it. Good news, we can do this using pre-defined FCN in keras (the dense layer). This is where using Keras will pay off. 

```python

def model_graph_embedding(embed_dim=10):
	model = Sequential()
	model.add(Embedding(max_node_index, embed_dim, input_length=2))
	# Now the output will have dimension (batch_dim, input_length, input_embedding). We will permute last two dimensions as described above
	model.add(Permute(2, 1))
	model.add(NodeEmbedLayer(32))
	# We will add dropout to increase generalization power of the model
	model.add(Droput(0.25))
	# FCN with 1st hidden layer of dim 10
	model.add(Dense(10))
	model.add(Dropout(0.5))
	# Adding final layer which will be passed through sigmoid for training using log-likelihood
	model.add(Dense(1, activation='sigmoid'))
	return model

```

Now the model is defined. You can call the model function and compile the network with your fav. optimizer and pass input batches sampled from the graph. I will publish the full code on github soon. 

Hope this post helps people seeking to learn custom layer writing tricks on Keras. 

Please do share your views (and this post also :) ). Cheers !!


References:

[1] Author2vec: Learning Author Representations by Combining Content and Link Information, J. Ganesh et. al., In WWW 2016






	


