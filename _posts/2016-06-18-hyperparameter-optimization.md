---
layout: post
title:  "Automatic Hyperparameter tuning"
excerpt: "I'll discuss about Automatic Hyperparameter tuning for Machine Learning models (with example)"
comments: true
mathjax: true
---

One problem that annoys all Machine Learning practitionar/researcher is Hyperparameter tuning. It is the most boring and non-productive task in any ML application. It essentially means that you have to sit all day, turn different knobs (Hyperparameters) and see which setting gives the best result. Though you can write a program to that for you but still you need to tune ranges manually until you are sure you are using optimal values. 

The most commonly used stratergy for Hyperparameter tuning is Grid-search. If you have N number of hyperparameters in your model and each Hyperparameter can take M values then Grid-Search essentially means trying out all M^N possible values and see which one performs the best. The problem with this approach is that the number of settings grows exponentially with number of hyper-parameters. One smart alternative to this stratergy is called - Random Search. In Random Search, if you have N variables and each variable can take N values then in each search iteration one value is sampled uniformly from each variable and the final setting is used. The intuition behind this scheme is that such stratergy can result in "broader" search as compared to grid-search. To understand it further, image a 2D surface and the two hyper-parameters are x and y. So search will essentially go like :

	for x in X:
		for y in Y:
			evaluate_objective(x, y)

This is pseudo-code for grid-search in 2D. We can observe that for a fixed value of X - x we are using all of Y, locus of search for fixed x is vertical line X = x. Now this is too constraint. Now in random-search, for each iteration of search we are sampling x and y uniformly and independently. In this case sampling independently is important, so in each iteration we get points which are not constraint in a particular region, but can span much larger space of paramters. So in theory random search is better then grid-search. 

But still this is a form of brute force search only. There were no definite scientific study of Hyperparameter search uptil now. 

But recently there has been some formal work in automating this process. There was one paper from MILA lab Montreal which lead to one great open-source project [SPEARMINT](https://github.com/HIPS/Spearmint). This is based on paper by Hugo Lorochelle and his group at MILA titled - [Practical Bayesian Optimization of Machine Learning Algorithms](https://arxiv.org/pdf/1206.2944.pdf). While this is a great effort, I found the software pretty hard to use. I found an alternative to it which was relatively easy to use - [Hyperopt](https://github.com/hyperopt/hyperopt). 

I will discuss about it with application to MNIST image classification task using keras. It uses some fancy method called Tree of Parzen estimators. This is a form of Bayesian optimization. I don't know the details of it, you can refer the papers [Hyperopt-scipy](http://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf) and [hyperopt-nips](http://jmlr.org/proceedings/papers/v28/bergstra13.pdf) 

Hyperopt lets you define a search space for each variable through a option of multiple distributions. You can choose a distribution according to the behaviour of the Hyperparameter (random variable). For example you can define regularization paramter using lognormal distribution which samples values from a relatively small range (0.001 - 0.01) which is typical for regularization parameters. Dropout can be defined as uniform random variable with range (0-1) and so on and so forth. It then wraps your model into one function and samples from these distributions  intelligently and searches the parameter space smartly in an efficient manner. Let's see all this in action. 

I will use a simple MLP for MNIST digit classification in keras for demonstrating how to use it with keras for building Deep Learning models. 

```python
# KERAS IMPORTS ASSUMED
from hyperopt import fmin, tpe, hp, STATUS_OK
# tpe is the search method, fmin is optimization method which will wrap the model

# Defining the model in a function which accepts the Hyperparameter 

def model(x):
	model = Sequential()
	model.add(Dense(512, input_shape=(784,)))
	model.add(Activation(x[2]))
	model.add(Dense(1000))
	model.add(Dropout(x[0]))
	model.add(Activation('relu'))
	model.add(Dropout(x[1]))
	model.add(Dense(10))
	model.add(Activation('softmax'))
	
	model.summary()

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(X_train, y_train, batch_size=32, nb_epoch=5)
	# Now defining model loss (cross-entropy loss, which will be used as objective by hyperopt

	score, loss = model.evaluate(X_test, y_test)
	# This function needs to return a dictinary of loss and some status flags to hyperopt
	return {'loss':loss, 'status': STATUS_OK}


# Note that we are ooptimizing the test loss so it will give best parameter setting which is generalizable 
# Now the model is defined, let's define the search space for each of the paramter. Search space is a tuple with each entry being the hyperparamter
space = (
	hp.uniform('x', 0, 1),
	hp.uniform('y', 0, 1),
	hp.choice('z', ['tanh', 'relu'])
	)

# choice is a bernoulli random variable for selecting between relu and tanh activation
# Now lets define the hyperopt function which will search for best paramter setting among all 

best = fmin(model, space=space, algo=tpe.suggest, max_evals=10)
# best will have the best setting of parameters which optimizes test loss
# This setting can be used as final setting
print best
```

This example is just to give you an idea on how to use hyperopt for automatic paramter tuning. The application will be much clearer when search is made on larger hyperparamter space. (10-100 hyperparameters). 



	
	
