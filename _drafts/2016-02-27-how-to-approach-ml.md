---
layout: post
type: post
title: "How to Assess Machine Learning from a Business and Product Perspective"
latex: false
draft: true
date: 2016-02-27
excerpt: ""
---

The last few years have seen an explosion of interest in machine learning technology and potential applications. As a non-expert you've probably either had to assess ML technology for your product and business or been considering if you ought to be. The jargon around machine learning technology is vast, confusing, and unfortunately increasingly being hijacked by overeager sales teams. This post is _not_ a primer on ML technology; this post won't pretend to give you an explanation of deep learning or any specific technology, because these concepts change frequently and are largely irrelevant to evaluation. Instead, this post will be how to asses the technology and it's impact on your product or business.

## What does Machine Learning _do_?

Ultimately, machine learning is about performing a _task_. The most common kinds of tasks, not an exhaustive list, are:

* [classification](https://en.wikipedia.org/wiki/Statistical_classification):  Predicting a label for a given item. For instance, guessing if a tweet about a given subject has a positive or negative sentiment. This is the kind of task most familiar to people. In order for a classification task to make sense, the valid labels for an item need to be well-defined (most people would agree). For instance, guessing if a picture is of a dog or cat is a relatively objective task.
* [regression](https://en.wikipedia.org/wiki/Regression_analysis): Predicting a real value (or multiple real values) for a given item. For instance, predicting the selling price of a home from inputs such as pictures of the home, past selling prices, listing info, etc.
* [ranking](https://www.cs.utah.edu/~piyush/teaching/ranking_tutorial.pdf): Given a set of items, rank the items according to a task-specific criterion. For instance, search ranking is about taking all the possible matches to a given query and ranking by the most user relevant result (as gauged by user interaction on results). This also includes specific tasks like recommendations (for instance Amazon similar products or Netflix movie suggestions). In most applications, ranking tasks don't have an inherently well-defined sense of "correct" answers, but is judged purely on user interactions (e.g., a system is better than another if more users click on it's suggestions relative to another).

There are plenty of other interesting problems (e.g., clustering) as well as refinements of the above categories that have relevant practical applications; for instance, a lot of financial modeling involves time-series prediction which I'm taking to be a more sophisticated sub-case of regression.

Note that none of these task definitions inherently have anything to do with machine learning. I can write a piece of code that classifies emails as spam that doesn't leverage any data or models learned from that data. Such _deterministic algorithms_, so named because their guesses are independent, can often perform as well, or in some maddening cases better, than sophisticated ML algorithms.


<!-- Footnotes and Links -->
