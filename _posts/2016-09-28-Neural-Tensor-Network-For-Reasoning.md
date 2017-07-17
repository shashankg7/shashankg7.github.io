---
layout: post
title:  "Neural Tensor Networks for Reasoning"
excerpt: "I'll discuss briefly about Neural Networks with Reasoning power"
comments: true
mathjax: true
---

To build a true AI agent one of the key feature required is it's ability to reason. Reasoning is one of the key factors that makes us "intelligent" in some sense. So it's natural to expect an AI agent to be "reasonable". That being said, is not easy to implement it in practice. There are lot of questions surrounding it. One of the key questions is how do you quantify reasoning ability? We know for humans we have some standard metrics to define it, like IQ, but how do we quantify it for a Machine?

Turns out we can use similar setup for a machine also. We judge IQ of a person by asking him to solve some series of "hard" questions and depending on his responses we assign some score to him. May not be true precisely but loosely resembles what we do to judge a person's IQ. We can do something similar for a machine also. We can judge it's reasoning power by asking it a series of questions and recording it's response, and based on it's response assign a score indicating how "reasonble" or "intelligent" it is.

But turns out it is not that simple. Human beings can reason because they have lot of "World Knowledge". They have lot of knowledge about external world by virtue of their experiences from interacting with it, reading about it, interacting with other human beings and so on. Machines don't have that priveledge. Although they do have access to a vast source of knowledge, "The Web", the problem is that they don't know how to read it, process it and extract knowledge from it. The main reason for it is because most of the information on the web is in unstructured form. What that means is that every source contains information in a different schema. Computers need the data to be presented to them in a fixed schema or format. So to learn about the world they need the "World Knowledge" to be presented to them in a structured format. 

It is hard to encompass whole "world knowledge" and "common sense" into one structured DB. But there is lot of efforts in this direction. There are few such structured databased which encodes this knowledge in structured format. These are called Knowledge Based (KBs). One popular KB is Freebase. It encodes knowledge in form of triplets with the following schema 
        
                       (entity1, relation, entity2)		

where entities are real world objects and relation is what connects them. For example one such entry could be
        
                       (Humans, part_of, living_beings)

And so on. To build an agent with reasoning power, it can use these KBs and learn these relations and develop some kind of "common sense". Now this is not a trivial task since we don't know how we (humans) develop a common sense. There is no mathematical model for it. So how do we simulate this on a computer?

We need to turn to Machine Learning for solving this task. Given large enough KB we let a Machine Learning model figure out some kind of encoding to store these knowledge in such a way such that it can be queried upon and it should be able to answer it intelligently depending upon how effectively it can learn interesting patterns from it. 

Since this is a more complicated problem, we need more complicated models for it. Few years back a paper from Stanford's AI lab attempted this problem using more sophisticated Neural Network. The reason for choice of Neural network is that due to it's design it is able to learn interesting non-linear interactions in the data. The more complex the network, it can discover more interesting patterns (well not technically correct, goes against Occum's Razor, but works for "more complicated" problems). 

Neural Tensor Networks operate over a poweful mathematical object called tensor as it's parameter. Mathematically Tensors are simple extension of Matrices in higher dimension. For example a point cloud in 3D space can be represented by a Matrix, while to represent a point cloud in 4D you need an extra dimension added to matrix. This is called a 4D tensor. Although, behind the scenes, tensors are used all the time in Neural Networks, for example input to a CNN is a tensor in most of Deep learning Frameworks (batch * Channel1*Channel2 * Channel3). But they are part of internal implementation of NNs, Parameters are still vectors or matrices. Due to efficiency issues they are converted into tensors and operated upon. 

But in a Neural Tensor Network paramter of the network is a tensor. I'll shed some light on the details of the network in the rest of the post. 

Inputs for these Neural Network are entities and their corresponding relations. To present an input to any Machine Learning model we need to extract features for it from the data. One thing that sets apart Neural Networks from other Machine Learning models is that they can "learn" this feature representation during the training process. This is what is making them so attractive these days, i.e. their ability of "Representation Learning". Since we are using Neural netowrks as our model, it will "learn" feature representation of entities during it's training, so we don't need to worry about it. 

Formally NTN is defined as, given two entities e1 and e2 "embedded" (features extracted) in a d-dimensional space and a relation R connecting them, we define a score function as follows:

$$
\begin{align}
g(e_1, R, e_2) = u_R ^ T f(e_1 ^ 2 W_R^{[1:k]} e_2 + V_R \begin{bmatrix} e_1 \\ e_2 \end{bmatrix} + b_R)
\end{align}
$$

where e1 and e2 are embedding vectors of the entities and Vr is the weight matrix and br for the rth relation respectively. Now the 'W' paramter here is very interesting. 
It is a tensor of dimensions (d*d*k), where d is the embedding dimensions of the entities and k is the "slice paramter", which I'll describe.

W encodes a k-way interection between two entities. This is called bilinear tensor product. Given a square matrix M of dimension d * d, x.T * M * y encodes one way interaction between vectors x and y. This tensor can be seen as 'k' different d * d matrices stacked up, so each slice of this matrix (in numpy format W[:, :, i] will encode one way interection netween two entities. 'f' is a non-linear function and it's dot product with the vector u will give a scalar score.

Now to train this machine ranking loss is used which ranks "true" triplet higher than "false" triplet. "True" triplet is sampled from training data and "false" triplet is randomly sampled by corrupting the second entity. Mathematically it is defined as :

$$
\begin{align}
\sum_{i=1}^{N} [max(0, 1 - g(T_+) + g(T_-))]
\end{align}
$$

SGD or L-BFGS can be used to optimize this function. Now comes the inference part.

This is pretty straight-forward, we only need to do forward propagation on the network. For a concrete example, consider a simple query : "What is the capital of Rajasthan?". From Freebase we will have a dictionary of entities and relations, we can parse this sentence to find occurence of entities and relations from the dictionary. In this text Rajasthan will be an entity and capital will be a relationship (?, capital, Rajasthan). Now we can formulate inference over this query as problem of finding the missing entity given relation and 2nd entity. This can be easily done by forward propagating all entities from the dictionary along with Capital and Rajasthan and taking the one with best score as response. If the machine is trained properly, it should output Jaipur as it's response. 

That's all about some background on Neural Tensor Network in this post. It will be interesting problem to code up in theano, since it involved tensors as it's parameter. Although it would be fairly simple to code it up in theano since it does not involved lot of operations, the only tricky part is to code up the bilinear tensor product. Scan op can be used here but it is not very efficient time-wise, we can repeat e1 and e2 k times and perform standard matrix-tensor product, but it will be inefficient memory-wise, since it will put burden on GPU. I'll figure something out and post the solutions in coming weeks :). 

PS : Since I am writing it in VIM (too lazy to setup a good editor with language support) with no grammatical checker and spell checker, you can spot lot of typos. Please try to ignore them :)

Reference:

[1] Socher et al. Reasoning With Neural Tensor Networks For Knowledge Base Completion Advances in Neural Information Processing Systems 26
