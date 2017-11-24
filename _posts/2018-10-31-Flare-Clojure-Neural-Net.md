---
layout: post
title: "Flare: Clojure Dynamic Neural Net Library"
draft: false
latex: true
draft: true
date: 2017-10-31
excerpt: ""
---

**tl;dr**: *I wrote [Flare](https://github.com/aria42/flare), a Dynamic Neural Net library in Clojure. It currently can do some non-trivial things and it's pretty fast: about 2.5x-3x faster than [PyTorch](http://pytorch.org/) for a CPU-based [a simple Bi-LSTM classifier](https://github.com/aria42/flare/blob/master/src/flare/examples/bilstm_tag.clj) (although PyTorch does **many** more things and is vastly more stable).*

It'd been a few years since I wrote a large piece of software, and so one of my goals this year was to just that. Checking out the machine learning software landscape, one of the surprising things that has happened was that Python has become the defacto language for ML due to due strong auto-grad neural net libraries like [TensorFlow](https://www.tensorflow.org/), PyTorch, [DyNet](https://github.com/clab/dynet), and several others. While I really like PyTorch's clean interface and generally prefer it amongst popular neural net libraries, I just don't feel anywhere near as productive in Python as I do with Clojure. So I looked at some JVM options, but none quite felt right or at the same level of simplicity as PyTorch. 

Since this was completely for fun, I made the really unpragmatic decision to write something from scratch. I think the result shares a lot with PyTorch, but feels like it was made for a functional programming language. 

## Enter Flare

Like other dynamic neural net libraries, you define a computation with tree, where each leaf is either a constant or model parameter. Each internal node represents some operation (matrix multipication, sigmoid transform, etc.). Here's a simple example in [Flare](http://github.com/aria42/flare):

{% highlight clojure %}
(ns example
  (:require [flare.core :as flare]
            [flare.node :as node]
            [flare.computation-graph :as cg]))

(flare/init!)
;; Z = X + Y example
(def X (node/const [2 3]))
(def Y (node/const [3 4]))
(def Z (cg/+ X Y))
(:value Z)
;; returns [5.0 7.0] tensor 
{% endhighlight %}

Notice that the computation happens *eagerly*, you don't need to call `forward` on any node, the operation happens as soon great the graph node.[^TF_EAGER] You can disable eager mode if you prefer lazier computation. The vast majority of the math in Flare happens natively. While Flare is 100% Clojure, the tensor implementations typically happen on native hardward. Flare Flare has a pluggable backed, but defaults to [Neandrthal](https://github.com/uncomplicate/neanderthal) which is incredibly fast and calls into native hardward (Intel MKL on the CPU and various GPU platforms) whenever possible.

While the above example is slightly verbose compared to PyTorch, for longer pieces of code you get more expressiveness. Like PyTorch, one of the core abstractions is a *module*, which builds larger parts of a network. In Flare, a module closes over other modules or parameters, and knows how to generate graphs from input(s). Here's what my [LSTM cell](https://en.wikipedia.org/wiki/Long_short-term_memory) implementation looks like (see [here](https://github.com/aria42/flare/blob/40e4fa0e27a2ddd5664e752640927d23f2e6d766/src/flare/rnn.clj#L17) for code in repo): 

{% highlight clojure %}
(defn lstm-cell [model input-dim hidden-dim]
  (let [cat-dim (+ input-dim hidden-dim)
        ;; stack (input, output, forget, gate) params
        ;; W_[i, o, f, g] [x, h_{t-1}] + b_[i, o, f, g]
        activations (module/affine model (* 4 hidden-dim) [cat-dim])
        zero  (flare/zeros [hidden-dim])
        init-output (node/const "h0" zero)
        init-state (node/const "c0"  zero)]
    (reify RNNCell
      (add-input! [this input last-output last-state]
        (flare/validate-shape! [input-dim] (:shape input))
        (flare/validate-shape! [hidden-dim] (:shape last-state))
        (let [x (cg/concat 0 input last-output)
              acts (module/graph activations x)
              ;; split (i,o,f) and state
              [iof, state] (cg/split acts 0 (* 3 hidden-dim))
              ;; split iof into (input, forget, output)
              [input-probs forget-probs output-probs]
                (cg/split (cg/sigmoid iof) 0 hidden-dim (* 2 hidden-dim))
              state (cg/tanh state)
              ;; combine hadamard of forget past, keep present
              state (cg/+
                     (cg/hadamard forget-probs last-state)
                     (cg/hadamard input-probs state))
              output (cg/hadamard output-probs (cg/tanh state))]
          [output state]))
        (cell-model [this] m)
        (output-dim [this] hidden-dim)
        (input-dim [this] input-dim)
        (init-pair [this] [init-output init-state]))))
{% endhighlight %}

Here's an example of building a simple bidirectional LSTM sentiment classifier for a given sentence using a module:


{% highlight clojure %}
(defn lstm-sent-classifier [model word-emb lstm-size num-classes]
  (node/let-scope
      ;; let-scope so the parameters get smart nesting of names
      [emb-size (embeddings/embedding-size word-emb)
       num-dirs 2
       input-size (* num-dirs emb-size)
       hidden-size (* num-dirs lstm-size)
       lstm (rnn/lstm-cell model input-size hidden-size)
       hidden->logits (module/affine model num-classes [hidden-size])]
    (reify
      module/PModule
      (graph [this sent]
        ;; build logits
        (let [inputs (embeddings/sent-nodes word-emb sent)
              [outputs _] (rnn/build-seq lstm inputs (= num-dirs 2))
              train? (:train? (meta this))
              hidden (last outputs)
              hidden (if train? (cg/dropout 0.5 hidden) hidden)]
          (module/graph hidden->logits hidden))))))
{% endhighlight %}


## Performance

The big surprise with Flare has been that performance is relatively strong compared to PyTorch, about 2x-3x faster for the models I've built in Flare. While I've optimized the obvious things in Flare, there is still a lot more low hanging fruit. I suspect some of the performance wins relative to PyTorch are coming from graph construction. While PyTorch and Flare both fallback to Intel native MKL on CPU, graph construction happens in the host language (Python or Clojure) and this is where PyTorch and Flare can differ performance-wise. 

As an example, I compared a simple Bi-LSTM sentence classifier task in Flare and PyTorch using the [Stanford Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) and [GlVe Vectors](https://nlp.stanford.edu/projects/glove/). Here is the [Flare train/eval script](https://github.com/aria42/flare/blob/0532d095b52c513d32fce2b90d25dd8b33722b86/src/flare/examples/bilstm_tag.clj) and here's the [PyTorch one](https://gist.github.com/aria42/2ff21b8c567d12d979a64f3a37fd029d). Note that, each iterations runs forward/backwards on training data, but also evaluates on train/test sets, so there's a 3:1 mix of forward versus backward computations. On my recent MPB laptop, running both of these scripts yields the following average iteration time (over 10 iterations, 5 runs) and varied the LSTM hidden state size:

| Library | LSTM Size | Iter Secs |
| :---    |     :---: |      ---: |
|---------|-----------|-----------|
| Flare |        25 |      38.3 |
| PyTorch |        25 |     120.5 |
|         |           |           |


## Up Next

While I'm not 100% sure the world needs another neural net library, I'm interested in building this out more as long as it's fun and I'm hitting interesting challenges. Here's a list of things I'd like to get to

* GPU Support: Neanderthal supports many platforms here, so this should be straightforward
* Auto-Batching: I like the auto-batching idea, described in this [paper](https://arxiv.org/abs/1705.07860). I have the start of a auto-batching computation, which shows some sign of speeding up training substantially. I'm not quite happy with the design, and want to take some time to think through if there's a cleaner way to add this.
* CNN-1D models: This is an obvious one. My first models built in Flare where all LSTM variants and I haven't had a need yet.

<!-- Footnotes and Links -->

[^global-min]:This assumes there is a unique global minimizer for $f$. In practice, in practice unless $f$ is convex, the parameters used are whatever pops out the other side of an iterative algorithm.
[^dl4j]:I took a look at [DeepLearning4J](https://deeplearning4j.org/), and while it's clearly fully-featured, it doesn't feel as expressive as say PyTorch. 
[^TF_EAGER]: TensorFlow recently developed [an eager computation mode](https://research.googleblog.com/2017/10/eager-execution-imperative-define-by.html) 
