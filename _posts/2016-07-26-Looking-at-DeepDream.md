---
layout: post
title:  "Understanding DeepDream"
excerpt: "In this post I'll write about my understanding of DeepDream"
comments: true
mathjax: true
---

I was planning to write about something on Computer Vision since I started writing my blog. But I was not sure about the topic, although writing about CNN seems to a obvious choice for the 1st post, but there are millions (literally millions) of post of web about CNN so I thought it would be redundant to write about them. 

Now since past few days I am reading a lot about this new app, "Prisma". It basically converts your picture into a painting style image. When I saw some of its image, they looked a lot like some of the recently popular "neural-style" images. 

![Image alt]({{ site.baseurl }}/assets/neural-style.png)

Till now I have only heard about them. This app increased my curiosity and I started reading about how these so-called "DeepDream" and "Neural-style" methods work. So I decided to write about it as my first post on DL4CV. So in this post I'll describe about my understanding on "DeepDream". I'll write about Neural-style (with code) in a coming post.

This project of "Deepdream" started when people started wondering on how these CNN's actually work. What is the intuition on using these hierarchical Neural Network units. People knew that using hierarchical neural net. units gives an increasing abstract representation of the input but people wanted to actually "see" what these intermediate layers actually "learn". Since the input in this case is an image, it is fairly reasonable to ask these questions. We can actually visualize the responses of neurons in each intermediate layers (by treating them as images).

DeepDream at its crux is this only, i.e. visualizing the activations of intermediate layers, which will facilitate in better understanding of the working mechanism of CNNs. This is DeepDream explained at an higher lavel of abstraction. Let's discuss it in more technical details.

To visualize what a particular filter in a particular layer is learning, a very simple trick is used. Our goal is to generate an input image which gives the maximum feature activation at the layer we are interested in. This is a fairly simple optimization problem. Given a layer's feature activation as the objective function, we want to find an input image which will result in maximum avg. activation at that layer. That input will correspond to the visualization of activation which is learned at that layer.

If you are using plain numpy, then it can be done by setting the gradient vector at that layer be equal to 1.0 and then backpropagate to the image and do a gradient ascent. If you are working with some "auto-diff" framework, it is straight-forward. Some images corresponding to features learned at various intermediate layers are:

![Image alt]({{ site.baseurl }}/assets/filter-responses.png)

