---
layout: post
title: Clojure Unsupervised Part-Of-Speech Tagger Explained
categories:
- clojure
- computer science
- nlp
tags: []
status: publish
latex: true
type: post
published: true
excerpt: Analaysis of the Clojure code sample that imprlements unsupervised part-of-speech tagger. 
---
<a href="http://aria42.com/blog/?p=33">Last week</a>, I posted a <a href="http://gist.github.com/578348">300 line clojure script</a> which implements some <a href="http://www.cs.berkeley.edu/~aria42/pubs/typetagging.pdf">recent work</a> I've published in <a href="http://en.wikipedia.org/wiki/Part-of-speech_tagging">unsupervised part-of-speech tagging</a>. In this post, I'm going to describe more fully how the model works and also how the implementation works. This post is going to assume that you have some basic background in probability and that you know some clojure. The post is massive, so feel free to skip sections if you feel like something is too remedial; I've  put superfluous details in footnotes or marked paragraphs.

## What exactly is unsupervised part-of-speech tagging?


Unsupervised <a href="http://en.wikipedia.org/wiki/Part-of-speech_tagging">part-of-speech (POS) tagging</a> is the task of taking the raw text of a language and inducing syntactic categories of words. For instance, in English, the words "dog","basketball", and "compiler", aren't semantically related, but all are common nouns. You probably know the basic syntactic categories of words: nouns, verbs, adjectives, adverbs, etc. However, natural language processing (NLP) applications typically require more fine grained distinctions. For instance, the difference between a singular, plural, or proper nouns. In English, the most commonly used annotated POS data has 45 different tags.

What we're interested in here is <a href="http://en.wikipedia.org/wiki/Unsupervised_learning">unsupervised learning</a>, meaning that at no point does the model get told about the kinds of syntactic categories we want nor does it get examples of annotated sentences or examples (there is no <emph>supervised</emph> data); you just get raw data. There are several advantages to using unsupervised learning, not least of which being there are languages that don't have POS annotated data.

A subtle consequence of being unsupervised is that we aren't going to directly learn that the word "dog" is a singular common noun. Instead, we learn there are some fixed number of tag states and all the things we call a singular common noun may map to tag state 42, for instance. Basically, the tags in the model don't come with the names we recognize, we have to map them to meaningful names (if that's what you're application requires). Essentially, what you get out of this is a clustering over words which corresponds to meaningful syntactic distinctions.

### How does the model work?
The model is a variation on the standard Hidden Markov Model (HMM), which I'll briefly recap. The HMM unsupervised POS tagging story works as follows: We assume a fixed number of possible tag states, $$K$$ as well as a fixed vocabulary of size  $$V$$. The Markov model part of HMM refers to the fact that the probability distribution of tag states for a single sentence is generated under a first-order Markov assumption. I.e., the probability
$$P(t_1,\ldots,t_m)$$ for a sentence of length $$m$$ is given by,

$$
P(t_1,\ldots,t_m) = \prod_{i=1}^m P(t_{i+1} | t_i)
$$

This encodes the intuition that typically some kind of noun or adjectives usually follows a determiner (e.g. "the","a","this").[ref]For each tag $$t$$, there are transition parameters $$\psi_t$$ over successor tags, drawn from a Dirichlet distribution over $$K$$ elements and hyper-parameter $$\alpha$$. These parameterize the $$P(t_{i+1} | t_i)$$ distributions.[/ref]

<div>
Once the tags of the sentence have been generated, for each position, a word is drawn conditioned on the underlying tag. Specifically, for $i=1,\ldots,n$ a word $w_i$ is drawn conditioned on the corresponding tag  $t_i$, $P(w_i | t_i)$. This <emph>emission</emph> distribution is parametrized according to parameters $\theta_t$ for each tag $t$ over the $V$ vocabulary elements. So for instance, for tag state 42, which we suppose corresponds to a singular noun, there is a distribution over all possible singular nouns, that might look like this:<sup id="fnref:42"><a href="#fn:42" class="footnote">1</a></sup>
</div>

$$
 \theta_{42} =  \left\{ dog: 0.03, basketball: 0.02, compiler: 0.01, \ldots \right\}
$$

If we let, $$\mathbf{w}$$ denote a corpus, consisting of a bunch of sentences, then the HMM puts mass on the corpus as well as corresponding
tag sequences $$\mathbf{t}$$ as follows,

$$
P(\mathbf{w},\mathbf{t}) = \prod_{(w,t) \in (\mathbf{w},\mathbf{t})} P(w,t) = \prod_{(w,t) \in (\mathbf{w},\mathbf{t})} \left( \prod_{i=0}^m P(w_{i+1} | t_{i+1}) P(t_{i+1} | t_i) \right)
$$

<h3>What's wrong with the HMM?</h3>
There's a lot wrong with the HMM approach to learning POS structure. One of the most important is that the model doesn't encode the constraint that a given word typically should be associated with a few number of tags. The model is perfectly happy to learn that a given word can be generated any number of tags. A lot of doing unsupervised machine learning is understanding how to alter models to reflect the constraints and preferences that the structure we are interested in has.

Another more subtle issue is that there is a significant skew to the number of words which belong to each part-of-speech category. For instance, there are very few determiners in most languages, but they appear very frequently at the token level. There is no way to encode this constraint that some tags are infrequent or frequent at the <emph>type-level</emph> (have very few (or many) unique word types that can use a given tag category). So the model has a prior $$P(T)$$ over tag assignments to words.[ref]This distribution is parametrized from a symmetric Dirichelt with hyper-parameter $$\beta$$ over $$K$$ possible tags.[/ref]

<h3>What's the approach in the paper?</h3>
The approach in the paper is actually very simple: For each word type $$W_i$$, assign it a single legal tag state $$T_i$$. So for the word type "dog", the model chooses a single legal tag (amongst $$t=1,\ldots,K$$); essentially, decisions are made for each word once at the  type level, rather than at the token-level for each individual instantiation of the word. Once this has been done for the entire vocabulary, these type tag assignments constrain the HMM $$\theta_t$$ parameters so only words assigned to tag $$t$$ can have non-zero probability. Essentially, we strictly enforce the constrain that a given word be given a single tag throughout a corpus.

When the model makes this decision it can use a type-level prior on how likely it is that a word is a determiner. Determiners, or articles, in general are very frequent at the token level (they occur a lot in sentences), but there are very few unique words which are determiners. Another thing we can do is have features on a word type in a <a href="http://en.wikipedia.org/wiki/Naive_Bayes_classifier">naive-bayes</a> fashion. We assume that each word is a bag of feature-type and feature-value pairs which are generated from the tag assigned to the word. The features you might have on a word type are what is the suffix of the word? Is it capitalized? You can configure these features very easily.

Let's summarize the model. Assume that the vocabulary of a language consists of $$n$$ word types. The probability of a type-level tag assignment is given by:

$$
P(\mathbf{T},\mathbf{W}) = \prod_{i=1}^n P(T_i) P(W_i | T_i) = \prod_{i=1}^n P(T_i) \left( \prod_{(f,v) \in W_i} P(v | f, T_i) \right)
$$

where, $$(f,v)$$ is a feature-type and feature-value pair in the word type (e.g., <code>(:hasPunctuation, false)</code>.  So each tag $$t$$ has a distribution over the values for each feature type. For instance, the common noun, tag 42 in our examples so far, is somewhat likely to have punctuation in the word (as in "roly-poly"). It's distribution over the <code>:hasPunctuation</code> feature-type might look like:[^3]

$$
\left\{ false: 0.95, true: 0.05 \right\}
$$



Once the tag assignments have been generated, everything proceeds identically to the standard token-level HMM except with the constraint that emission distributions have been constrained  so that a tag can only emit a word if that word has been assigned to the tag.

<h3>How do you learn?</h3>
The fairly simple change to the model made in the last section not only yields better performance, but also makes learning much simpler and efficient. Learning and inference will be done using <a href="http://en.wikipedia.org/wiki/Gibbs_sampling">Gibbs Sampling</a>. I can't go over Gibbs Sampling fully, but I'll summarize the idea in the context of this work.  The random variable we don't know in this model are the type-level assignments $$\mathbf{T} = T_1,\ldots, T_n$$. In the context of Bayesian models, we are interested in the posterior $$P(\mathbf{T} | \mathbf{W}, \mathbf{w})$$, where $$\mathbf{W}$$ and $$\mathbf{w}$$ denote the word types in the vocabulary and  the tokens of the corpus respectively; essentially, they're both observed data.[ref]Note that the token-level tags $$\mathbf{t}$$ are determined by type-assignments $$\mathbf{T}$$, since each word can only have one tag which can generate it.[/ref] We can obtain samples from this posterior by repeatedly sampling each of the $$T_i$$ variables with the  other assignments, denoted $$\mathbf{T}_{-i}$$, fixed. We sample $$T_i$$  according to the posterior $$P(T_i | \mathbf{T}_{-i}, \mathbf{W}, \mathbf{w})$$, which basically reprsents the following probability: If I assume all my other tag assignments are correct, what is the distribution for the tag assignment to the $$i$$th word. It's relatively straightforward to show that if we continually update the sampling state $$\mathbf{T}$$ one-tag-at-a-time in this way, at some point, the sampling state $$\mathbf{T}$$ is drawn from the desired posterior $$P(\mathbf{T} | \mathbf{W}, \mathbf{w})$$.[ref]In practice, for any real problem, one doesn't know when Gibbs Sampling, or MCMC in general, has "burned in".[/ref] So essentially, learning boils down to looping over tagging assignments and sampling values while all other decisions are fixed.

 In the original HMM, when using Gibbs Sampling, the state consists of all token-level assignments of words to tags. So the number of variables you need to sample is proportional to the number of words in the corpus, which can be massive. In this model, we only need to sample a variable for each word type, which is substantially smaller, and importantly grows very slowly relative to the amount of data you want to learn on.

Okay, so learning with this model boils down to how to compute the local posterior:

$$
\begin{array}{cl}
        P(T_i = t| \mathbf{T}_{-i}, \mathbf{W}, \mathbf{w})
             \propto& P(T_i = t | \mathbf{T}_{-i}) P(W_i | T_i = t,\mathbf{T}_{-i},\mathbf{W}_{-i}) \\
             & P(\mathbf{w} | T_i = t, \mathbf{T}_{-i})
\end{array}
$$

<div>
Let me break down each of these terms. The $P(T_i = t | \mathbf{T}_{-i})$ is straight-forward to compute; if we count all the other tag assignments, the probability of assigning $T_i$ to $t$ is given by, $ \frac{n_{t} + \alpha}{n-1 + \alpha} $ where $n_t$ is the number of tags in $\mathbf{T}_{-i}$ which are currently assigned to $t$. The $\alpha$ term is the smoothing concentration parameter.[^params]
</div>

A similar reasoning is used to compute,
$$! P(W_i | T_i = t,\mathbf{T}_{-i}) = \prod_{(f,v) \in W_i} P(v | f, T_i = t, \mathbf{T}_{-i}, \mathbf{W}_{-i}) $$
which decomposes a product over the various features on the word type. Each individual feature probability can be computed by using counts of how often a feature value is seen for other words assigned to the same tag.

The last term requires a little thinking. For the purpose of Gibbs Sampling, any probability term which doesn't involve the thing we're sampling, we can safely drop. At the token-level, the assignment of the $$i$$th word type to $$t$$ only affects the local contexts in which the $$i$$th word type appears. Let's use $$w$$ to denote the $$i$$th word type. Each usage of $$w$$ in the corpora are associated with a previous (before) word and a following (after) word.[ref]We pad each sentence with start and stop symbols to ensure this.[/ref] Let's use $$(b,w,a)$$ to represent the before word, the word itself, and the after word; so $$(b,w,a)$$ represents a trigram in the corpus. Let $$T(b)$$ and $$T(a)$$ denote the tag assignments to words $$b$$ and $$a$$ (this is given to us by $$\mathbf{T}$$). The only probability terms associated with this usage which not constant with respect to the $$T_i = t$$ assignment are:
$$! P(w | T_i = t, \mathbf{T}_{-i}, \mathbf{w}_{-i}) P(t | T(b), \mathbf{T}) P(T(a) | t, \mathbf{T}) $$
These terms are the probability of the word itself with the considered tag, the probability of transitioning to tag $$t$$ from the tag assigned to the previous word, and transitioning to the tag assigned to the successor word. The only terms which are relevant to the assignment come from all the context usages of the $$i$$th word type.
Specifically, if $$C_i$$  represents the multi-set of such context usages, we have $$P(\mathbf{w} | T_i=t, \mathbf{T}_{-i})$$ is proportional to a product of the terms
in each $$(b,w,a)$$ usage.  These probabilities can be computed by storing corpus level counts. Specifically for each word, we need counts of the <code>(before, after)</code> words as well as the counts for all individual words.

<h2>Finally, walking through the implementation!</h2>
Okay, so after a lot of prep work, we're ready to dissect the code. I'm going to go linearly through the code and explain how each piece work. For reference, the full
script can be found <a href="http://gist.github.com/578348">here</a>.

<h3>It's all about counters</h3>
So one of the basic data abstractions you need for probabilistic computing is a counter.[ref]A lot of the names for these abstractions come from <a href="http://www.cs.berkeley.edu/~klein/">Dan Klein</a>, my PhD advisor, but I'm pretty sure modulo the name, the abstractions are pretty universal from my survey of machine learning libraries.[/ref] Essentially, a counter is a map of items to their counts, that needs, for computing probabilities, to support a fast way to get the sum of all counts. Here's the code snippet that declares the appropriate data structure as well as the important methods. The proper way to do this is to make Counter a protocol (which I've done in my NLP clojure library <a href="http://github.com/aria42/mochi/blob/master/src/mochi/counter.clj">here</a>):

{% gist 587011 %}

The two functions here are the only two we need for a counter: <code>inc-count</code> increments a count and returns a new counter, and <code>get-count</code> returns the current count. Since in Gibbs Sampling, none of our counts should be negative, we add an important <code>:post</code> check on <code>get-count</code> which will likely catch bugs.

<h3>Dirichlet Distributions</h3>

Once we have the <code>counter</code> abstraction, it's very straightforward to build a probability distribution; all the distributions here are over a finite number of possible events. This kind of distribution is called a <a href="http://en.wikipedia.org/wiki/Multinomial_distribution">multinomial</a>. Here, we use a <code>DiricheltMultinomial</code> which represent the a multinomial drawn from the symmetric <a href="http://en.wikipedia.org/wiki/Dirichlet_distribution">Dirichlet distribution</a>, which essentially means that all outcomes are given "fake" counts to smooth the probabilities (i.e., ensure no probability becomes zero or too small). The kinds of things we want to do with a distribution, simply include asking for the log-probability[ref]To guard against numerical underflow, we work primarily with log-probabilities.[/ref] and making a weighted observation which changes the probabilities the distribution produces. Here's the code. I'll give more explanation and examples after:


[gist id="587021"]

<b>Paragraph can be safely skipped</b>: The probabilities we need from the <code>DirichletMultinomial</code> are actually the "predictive" probabilities obtained from integrating out the Dirichelt parameters. Specifically, suppose  we have a distribution with $$n$$ possible event outcomes and  assume the multinomial over these $$n$$ events are drawn $$\theta \sim Dirichlet(n, \alpha)$$. Without observing any data, all $$n$$ outcomes are equally likely. Now, suppose we have observed data $$\mathbf{X}$$ and that $$n_i$$ is the number of times, we have observed the $$i$$th outcome in $$\mathbf{X}$$. Then, we want the probability of a new event $$e^*$$ given the observed data,

$$
 P(e^* = i | \mathbf{X}) = \int_\theta P(\theta | \mathbf{X}) P(e^* = i | \theta) d \theta = \frac{n_i + \alpha}{\left(\sum_{i'} n_{i'}\right) + n * \alpha}
 $$

Given, a counter over events, we can efficiently compute a given probability. Each probability depends on knowing: the count of the event (<code>get-count</code>), the sum over all counts for all events (<code>total</code> from the <code>counter</code>), as well as the number of unique keys that this distribution could emit (<code>num-keys</code>). The reason we don't just look at the number of keys in the counter is because we're interested in the number of <emph>possible</emph> values; at any given time, we may not have counts for all possible events.

Making an observation to a distribution, in this context, just requires increment the count of the event so that subsequent calls to <code>log-prob</code> reflect this observation.

<h3>What's in a word?</h3>
Okay, now that we have some standard code out of the way, we need to do some POS-specific code. I'm going ot use a record <code>WordInfo</code> which represents all the information we need about a word in order to efficiently support Gibbs Sampling inference. This information includes: a string of the word itself, its total count in the corpus, a map of the feature-type and feature-value pairs, and a counter over the pairs of the context words which occur before and after word (specifically it will be a counter over <code>[before-word after-word]</code> pairs). Here's the code:

{% gist 587038 %}

The <code>get-feats</code> function simply returns a map of the feature-type (a keyword here) and its value. You can easily edit this function to have other features and the rest of the code will just work.  

Now that we have this data-structure, we need to build this data structure to represent the statistics from a large corpus. Okay, suppose that I want to update the word-info for a given word after having observed a usage. The only info we need from the usage is the before and after word:

{% gist 587045 %}

Two things need to change: (1) We need to increment the total usage of the word (the <code>:count</code> field in <code>WordInfo</code>). (2) We need to increment the count of the before-and-after pair <code>[before after]</code> in the counter for the context usages. Here's what I love about clojure: If you design your abstractions and functions correctly, they work seamlessly with the language. If you don't know the <code>-></code> threading macro: learn it, live it, love it. I think in conjunction with the <code>update-in</code> function, it allows for very succinct functions to update several piece of a complex data structure.

Okay, so let me show you the rest of the pieces which build of the word info data structures from a corpus (a seq of sentences):

{% gist 587049 %}

What we want to do in <code>tally-sent</code> is update the word-info records for a given sentence. For this function, we have a map from a word string to its corresponding <code>WordInfo</code> record. The <code>(partition 3 1 sent)</code> produces a sequence of <code>(before-word word after-word)</code> trigrams which are all we need to update against. For each <code>word</code> in this triple, we ensure we have a <code>WordInfo</code> record (is there a <code>assoc-if-absent</code> function in core or contrib). And then we use our <code>tally-usage</code> function to update against the before and after word. Finally, we perform this update over all the sentences of a corpus in <code>build-vocab</code>.

<h3>Gibbs Sampling State</h3>
Let's talk about how we represent the state of the Gibbs Sampler. Okay state is a dirty word in Clojure, and luckily the usage of state here is from Statistics and it represents an immutable value: for a given point in Gibbs Sampling, what are all the relevant assignments and the derived corpus counts from this assignment. Here's the code:

{% gist 587057 %}

I think the comments are sufficient here. The one thing that I should explain is that given the corpus and the <code>type-assigns</code>, all the other fields are determined and could theoretically be computing on the fly as needed. For efficiency however, it's better to update those counts incrementally.

<h3>Updating Gibbs Sampling State After an assignment/unassignment</h3>

Now there are a lot of functions we need to write to support what happens when you add an assignment of a tag to a given word type or remove the assignment. These operations are the same, except when you make an assignment you are adding positive counts, and when you are unassigning, you remove counts. All these functions tend to take a <code>weight</code> to allow code reuse for these operations. Okay, so let's take the case of updating the emission distribution associated with the tag which has been assigned/unassigned to a word-info. Two things need to change: we need to change the number of possible values the distribution can produce. If we are assigning the tag to the word, there is another possible outcome for the emission distribution; similarly we need to decrement if we are removing the assignment. Also, we need to observe the word the number of times it occurs in the corpus.

{% gist 587062 %}

To be clear, <code>tag-emission-distr</code> is obtained from <code>(get-in state [:emission-distrs , tag])</code> where <code>state</code> is an instance of <code>State</code>.

There are analogous functions for updating the counts for the feature distributions and for the transitions. I'll briefly go over updating the transitions since it's bit trickier. When we assign a word to a tag, we need to loop over the <code>[before-word after-word]</code> counts in the <code>WordInfo</code> and, depending on the current tagging assignment, change these counts. Here's the code:

{% gist 587068 %}

<h3>Gibbs Sampling Step</h3>
Okay, so let's take a top-down perspective for looking at how we make a simple Gibbs Sampling step. We first take our current state, unassign the current assignment to a word, and then sample a new value from the distribution $$ P(T_i = t| \mathbf{T}_{-i}, \mathbf{W}, \mathbf{w})$$:

{% gist 587070 %}

I didn't show you the <code>assign</code> and </code>unassign</code> functions. All they do is update the Gibbs Sampling state data structures to reflect the change in assignment for a given word as discussed above. They both are nice pure functions and return new states.

You also haven't seen <code>score-assign</code> and </code>sample-from-scores</code>, which I'll discuss now. <code>score-assign</code> will return something proportional to the log-probability of $$ P(T_i = t| \mathbf{T}_{-i}, \mathbf{W}, \mathbf{w})$$. </code>sample-from-scores</code> will take these scores from the possible assignments and sample one.

Here's <code>score-assign</code>:
{% gist 587075 %}

The <code>(log-prob (:tag-prior state)  tag)</code> corresponds to $$P(T_i = t | \mathbf{T}_{-i})$$. The following <code>sum</code> form corresponds to the log of $$\prod_{(f,v) \in W_i} P(v | f, T_i)$$, the probability of the bundle of features associated with a given word type conditioned on the tag. The last top-level form (headed by <code>let</code>) has all the token-level terms:  $$P(w | \mathbf{T}, \mathbf{w}_{-i})^{n_i} \prod_{(b,a) \in C_i}  P(t | T(b), \mathbf{T}) P(T(a) | t, \mathbf{T})$$. That <code>let</code> statement needs to suppose that the tag assignment has already happened to correctly compute the probability of the word under the tag. The inner <code>sum</code> term for each <code>[[before-word after-word] count]</code> entry adds the log-probabilities for all these usages (I also lump in the word log-probability itself, although this could be in a separate term weighted with the total occurrence of the word).

Note that the time it takes to score a given assignment is proportional to the number of unique contexts in which a word appears.

Once we have this function, we need to sample proportionally to these log-probabilities. Here is some very standard machine learning code that would normally be in a standard library:

{% gist 587090 %}

<h3>All the rest...</h3>
From here, I think the rest of the code is straightforward. An iteration of the code consists of sampling each word's assignment. There is a lot of code towards the end for initializing state. The complexity here is due to the fact that I need to initialize all maps with distributions with the correct number of possible keys. I hope this code make sense.

[^params]: I don't have the room to discuss this here, but this probability represents the "predictive" distribution obtained by integrating out the distribution parameters.
[^42]: Each $$\theta_t$$ is drawn from a symmetric Dirichlet prior over $V$ elements and with concentration parameter $$\alpha$$ (shared with transition parameters for simplicity).
[^3]: For each tag and feature-type, the distribution is parametrized by a symmetric Dirichlet over all possible feature-values and hyper-parameter $$\beta$$.
