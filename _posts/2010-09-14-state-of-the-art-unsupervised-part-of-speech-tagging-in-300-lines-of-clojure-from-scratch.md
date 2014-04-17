---
layout: post
title: State-Of-The-Art Unsupervised Part-Of-Speech Tagging in 300 lines of Clojure  (from
  Scratch)
categories:
- computer science
tags: []
status: publish
latex: true
type: post
published: true
meta:
  _edit_last: '1'
  superawesome: 'false'
  jabber_published: '1284428358'
  _wp_old_slug: ''
---
Recently, <a href="http://people.csail.mit.edu/yklee/">Yoong-Keok Lee</a>, <a href="http://people.csail.mit.edu/regina/">Regina Barzilay</a>, and myself, published a <a href="http://www.cs.berkeley.edu/~aria42/pubs/typetagging.pdf">paper</a> on doing unsupervised <a href="http://en.wikipedia.org/wiki/Part-of-speech_tagging">part-of-speech tagging</a>. I.e., how do we learn syntactic categories of words from raw text. This model is actually pretty simple relevant to other published papers and actually yields the best results on several languages. The C++ code for this project is <a href="http://groups.csail.mit.edu/rbg/code/typetagging/">available</a> and can finish in under a few minutes for a large corpus.

Although the model is pretty simple, you might not be able to tell from the C++ code, despite Yoong being a top-notch coder. The problem is the language just doesn't facilitate expressiveness the way my favorite language, <a href="http://clojure.org">Clojure</a>, does. In fact the entire code for the model, without dependencies beyond the language and the standard library, <a href="http://richhickey.github.com/clojure-contrib/index.html">clojure contrib</a>, can be written in about 300 lines of code, complete with comments. This includes a lot of standard probabilistic computation utilities necessary for doing something like <a href="http://en.wikipedia.org/wiki/Gibbs_sampling">Gibbs Sampling</a>, which is how inference is done here.

Without further ado, the code is on <a href="http://gist.github.com/578348">gisthub</a> and <a href="http://github.com/aria42/type-level-tagger">github</a> (in case I make changes).

{% gist 578348 %}
