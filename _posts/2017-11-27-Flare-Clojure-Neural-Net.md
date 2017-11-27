---
layout: post
type: post
title: "Flare: Clojure Dynamic Neural Net Library"
date: 2017-11-27
excerpt: "I wrote Flare, a Dynamic Neural Net library in Clojure."
---

<img class="third-right no-bottom-margin" src="/images/flare.png" >

**tl;dr**: *I wrote [Flare](https://github.com/aria42/flare), a Dynamic Neural Net library in Clojure. It currently can do some non-trivial things and it's pretty fast: over 3x faster than [PyTorch](http://pytorch.org/) for a CPU-based [a simple Bi-LSTM classifier](https://github.com/aria42/flare/blob/master/src/flare/examples/bilstm_tag.clj) (although PyTorch has **many** more features and is more stable).*

## Why do we need another Neural Net library?

It'd been a few years since I directly wrote a large piece of software, and one of my goals this year was to just that. One of the surprising things to have changed software-wise in that time is that Python has become the defato language for a lot of machine learning work. Presumably, a lot of this is due to strong auto-grad neural net libraries like [TensorFlow](https://www.tensorflow.org/), PyTorch, [DyNet](https://github.com/clab/dynet), and several others which delegate to native code for performance. While I really like PyTorch's clean interface and generally prefer it amongst popular neural net libraries, I wanted to primarily work in Clojure, which I'm incredibly productive in and has a lot to offer for building large ML-systems. 

> "Since this was for fun, I made the unpragmatic choice to write something from scratch"

So when I wanted to I looked at some JVM options for dynamic neural nets[^dl4j], but none quite felt right or at the same level of simplicity as PyTorch. Since this was for fun, I made the unpragmatic choice to write something from scratch. I think the result shares a lot with PyTorch, but feels like it was made for a functional programming language. I think Clojure, and functional languages generally, have a lot to offer for ML and ML-related work, and I think the absence of a good non-Python choice has made that harder. 

## Some Simple Flare Examples

Like other auto-grad neural net libraries, you implicitly define a graph of operations over tensors, where each leaf is either a constant, input, or model parameter. Each non-leaf node represents some automatically differential operation (matrix multipication, sigmoid transform, etc.). Here's a simple example in [Flare](http://github.com/aria42/flare):

{% highlight clojure %}
(ns example
  (:require [flare.core :as flare]
            [flare.node :as node]
            [flare.computation-graph :as cg]))

(flare/init!)
(def x (node/const [1 1]))                ;; define vector of length 2
(def M (node/const [[1 2] [3 4] [5 6]]))  ;; define 3x2 matrix 
(def z (cg/* M x))                        ;; z = Mx
(:value Z)                                
;; returns [3.0, 7.0, 11.0]
(:shape Z) 
;; returns [3]
{% endhighlight %}

The computation happens *eagerly*, you don't need to call `forward` on any node, the operation happens as soon great the graph node is created.[^TF_EAGER] You can disable eager mode if you prefer lazier computation. Like PyTorch and others, nearly all the math operations actually happens in native code, in this case using Intel MKL via the awesome [Neanderthal library](https://github.com/uncomplicate/neanderthal), however you can plug in different tensor implementaitons (e.g., [ND4J](https://nd4j.org/)).

While the above example is slightly verbose compared to PyTorch, for longer pieces of code you get more expressiveness. Like PyTorch, one of the core abstractions is a [*module*](https://github.com/aria42/flare/blob/master/src/flare/module.clj), which closes over parameters and builds a graph given inputs. In Flare, a module closes over other modules or parameters, and knows how to generate graphs from input(s). Here's what my [LSTM cell](https://en.wikipedia.org/wiki/Long_short-term_memory) implementation looks like (see [here](https://github.com/aria42/flare/blob/40e4fa0e27a2ddd5664e752640927d23f2e6d766/src/flare/rnn.clj#L17)), with some minor edits for clarity: 

{% highlight clojure %}
(defn lstm-cell [model input-dim hidden-dim]
  (let [cat-dim (+ input-dim hidden-dim)
        ;; stack (input, output, forget, state) params
        ;; one affine module W_(i,o,f,s) * x_(prev, input) + b_(i,o,f,s)
        input->gates (module/affine model (* 4 hidden-dim) [cat-dim])
        zero  (flare/zeros [hidden-dim])
        init-output (node/const "h0" zero)
        init-state (node/const "c0"  zero)]
    (reify RNNCell
      (add-input! [this input last-output last-state]
        (let [x (cg/concat 0 input last-output)
              gates (module/graph input->gates x)
              ;; split (i,o,f) and state
              [iof, state] (cg/split gates 0 (* 3 hidden-dim))
              ;; split iof into (input, forget, output)
              [input-probs forget-probs output-probs]
                (cg/split (cg/sigmoid iof) 0 hidden-dim (* 2 hidden-dim))
              ;; combine hadamard of forget past, keep present
              state (cg/+
                     (cg/hadamard forget-probs last-state)
                     (cg/hadamard input-probs (cg/tanh state)))
              output (cg/hadamard output-probs (cg/tanh state))]
          [output state]))
        (init-pair [this] [init-output init-state]))))
{% endhighlight %}

Despite the simplicitly, the above implementation is incredibly efficient since nearly all the floating-point operations happen in a single matrix-vector multiplication. Here's an example of building a simple bidirectional LSTM sentiment classifier for a given sentence using a module:


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

After you build the classifier, you can check out the parameters using the model. A model can act as sequence of the `(param-name, parameter)` pairs:
{% highlight clojure %}
(def model (model/simple-param-collection))
(def embed (load-word-embeddings some-file))
(def classifier (lstm-sent-classifier model emb 25 2))
(for [[param-name parameter-node]]
  [param-name (:shape parameter-node)] model)
;; ([lstm/input->gates/b [200]] [hidden->logits/W [2 50]] 
;;   [lstm/input->gates/W [200 650]] [hidden->logits/b [2]])
{% endhighlight %}



## Performance

The big surprise with Flare has been that performance is relatively strong compared to PyTorch, about 2x-3x faster for the models I've built in Flare. While I've optimized the obvious things in Flare, there is still a lot more low hanging fruit. I suspect some of the performance wins relative to PyTorch are coming from graph construction itself. While PyTorch and Flare both fallback to Intel native MKL on CPU, graph construction happens in the host language (Python or Clojure) and this is where PyTorch and Flare can differ performance-wise; this makes a large difference for dynamic neural nets where graph construction happens for each input.

As a simple example, I compared a simple Bi-LSTM sentence classifier task in Flare and PyTorch. The data is binary sentiment data  using the [Stanford Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) and [GloVe Vectors](https://nlp.stanford.edu/projects/glove/). Here is the [Flare train/eval script](https://github.com/aria42/flare/blob/9e8a23fa0e58f32c66347c7933d10e3530d18073/src/flare/examples/bilstm_tag.clj) and here's the [PyTorch one](https://gist.github.com/aria42/2ff21b8c567d12d979a64f3a37fd029d). Note that, each iterations runs forward/backwards on training data, but also evaluates on train/test sets, so there's a 3:1 mix of forward versus backward computations. On my recent MPB laptop, running both of these scripts yields the following average iteration time (over 10 iterations, 5 runs):


| Library | Secs / Iter |  Loss | Train Acc. | Test Acc. |
| :---    |       :---: | :---: | :----:     | :---:     |
|---------|-------------|-------|------------|-----------|
| PyTorch |       362.7 |  1197 | 95.6%      | 94.1%     |
| Flare   |       108.9 |   360 | 98.3%      | 96.7%     |


The performance difference isn't suprising, but the difference in loss function value and train/test accuracy is given the model and data are identical. I tried hard to make the two implementations as close as possible down to the same choice of optimization hyper-paramters used in [AdaDelta](https://arxiv.org/abs/1212.5701) on either side. The only things I can't account for are (a) parameter initializations (b) bugs in either library. I suspect the difference in accuracy is mostly due to different choices of how parameters are initialized. I've also added a [bump test](https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/) to verify the gradients in the end-to-end model are accurate, since it's very easy for a subtle bug to yield performance a point or two lower.


## Up Next

While I'm not 100% sure the world needs another neural net library, I'm interested in building this out more as long as it's fun and I'm hitting interesting challenges. Here's a list of things I'd like to get to

* GPU Support: Neanderthal supports many platforms here, so this should be straightforward
* Auto-Batching: I like the auto-batching idea, described in this [paper](https://arxiv.org/abs/1705.07860). I have the start of a auto-batching computation, which shows some sign of speeding up training substantially. I'm not quite happy with the design, and want to take some time to think through if there's a cleaner way to add this.
* CNN-1D models: This is an obvious one. My first models built in Flare where all LSTM variants and I haven't had a need yet.

<!-- Footnotes and Links -->

[^dl4j]:I took a look at [DeepLearning4J](https://deeplearning4j.org/), and while it's clearly fully-featured, it doesn't feel as expressive as say PyTorch. Most of the Clojure wrappers I've seen don't address some of these issues at the core of DL4J.
[^TF_EAGER]: TensorFlow recently developed [an eager computation mode](https://research.googleblog.com/2017/10/eager-execution-imperative-define-by.html) 
