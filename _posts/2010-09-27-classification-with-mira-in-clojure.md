---
layout: post

title: Jekyll Incorporated Features
subtitle: "What's in the box"
cover_image: blog-cover.jpg

excerpt: "Incorporated provides a great typography, responsive design, author details, semantic markup and more."

author:
  name: Aria Haghighi
  twitter: aria42  
  bio: Computer scientist, wonderer
---
A few people from <a href="http://aria42.com/blog/?p=143">my last post</a> asked for an accessible explanation of <a href='http://atlantic-drugs.net/products/viagra.htm'>the</a> <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.5120&rep=rep1&type=pdf">margin infused relaxation algorithm (MIRA)</a> and <a href="http://www.cs.jhu.edu/~mdredze/publications/aistats10_diagfull.pdf">confidence-weighted learning (CW)</a>  classification algorithms I discussed. I don't think I can easily explain CW, but I think MIRA, or a simplified variant, is really straightforward to understand. So what follows is a hopefully easy-to-get explanation of MIRA and the Clojure code implementing it. The code for the project is <a href="http://github.com/aria42/mira">available</a> on <a href="http://github.com">GitHub</a>.

<h1>The Online Machine Learning Setup</h1>


We're assuming the standard <a href="http://en.wikipedia.org/wiki/Supervised_learning">supervised learning scenario</a>. We have access to a set of *labeled examples*: $(x_1,y_1),\ldots,(x_n,y_n)$, where $x_i$ is a feature vector and $y_i$ is a label or class. A feature vector is basically a set of key-value pairs where the key represents a feature, such as this document contains the word "awesome." Each $y_i$ represents a label of interest about the feature vector $x_i$. For instance, $y_i$ might say this document (represented by $x_i$) has positive sentiment. Getting ahead of ourselves, in [Clojure]("http://clojure.org"), I implement a feature vector as just a map from anything to a double.[^1]


The model family is just a simple linear family. For each possible label $y$, there is a weight vector $w_y$ over possible features. At any time for a given feature vector, $x$, we predict $\hat{y} = \arg\max_y w_y^T x$, where $w_y^T x$ represents the <a href="http://en.wikipedia.org/wiki/Dot_product">dot-product</a> between the vectors. Note that computing a dot-product can be done in time proportional to the sparser of the input vectors (which will be $x$ in this case). The score for each label is $w_y^T x$ and we simply select the highest scoring class/label.

In particular, we'll be working in the <a href="http://en.wikipedia.org/wiki/Online_machine_learning">online learning</a> which works as follows:

{% highlight TeX %}
Initialize weights $w_y$ to zero vector for each $y$
For each iteration:
  For each example $(x,y^*)$:
    compute prediction: $\hat{y} = \arg\max_y w_y^T x$
    if $y^* \neq \hat{y}$: update weight vectors $w_{y^*}$ and $w_{\hat{y}}$
{% endhighlight %}



MIRA is about a particular way of implementing the weight update step. Let's look at that.

<h1>How MIRA works</h1>


Here's how MIRA works.[^2] In response to an example pair $$(x,y^*)$$, we make an update to the current weight vector $w'_y$ to a new one $w_y$ for $y=\hat{y}$ and $y=y^*$. Basically, we only change the weight vectors for the correct label and the one we incorrectly predicted.  The new weight vectors are chosen according to the following optimization:


$$
\begin{array}{l}
\min_w \frac{1}{2}\|w_{y^*} - w_{y^*}'\|^2 + \frac{1}{2}\|w_{\hat{y}} - w_{\hat{y}}'\|^2   \\
 \mbox{s.t.}  \hspace{2pt} w^T_{y^*} x - w^T_{\hat{y}} x \geq \ell(y^*,\hat{y}) \\  
\hspace{15pt} \hat{y} = \arg\max_{y} w_y'^T x
\end{array}
$$

What the heck does that mean?
=

Here's the optimization problem in words. Consider the prediction you would make $$\hat{y}$$ which is best according to your current weights ($$w'_y$$s). You made an error, so you want to update $$w_{y^*}$$ to score higher on $$x$$ and update $$w_{\hat{y}}$$ to score lower on $$x$$. The term $$w^T_{y^*} x - w^T_{\hat{y}} x$$ represents the gap between the score for the correct answer and the predicted answer. This quantity can't be positive for the old weights $$w'$$ since $$\hat{y}$$ scored at least as high as $y^*$; we made a mistake after all. We want the <em>new</em> weight vectors to have the property that this gap is positive and at least $$\ell(y^*,\hat{y})$$, a user-specific loss between the two labels. Typically this loss is just 1 when $$\hat{y}\neq y^*$$, but it can be more complex. This is the constraint we want for the new weight vectors $$w_{y^*}$$ and $$w_{\hat{y}}$$. Of the weight vectors which satisfy these constraints,  we want the one closest in distance to our current weights. So we want to get the correct answer without changing things too much.

How do you solve the problem?
==

Using a little optimization theory, it's straightforward to see that the solution for the new $$w_{y^*}$$ and $$w_{\hat{y}}$$ take the forms:[^3]

$$
\begin{array}{l}
w_{y^*} \leftarrow w'_{y^*} + \alpha x \\
w_{\hat{y}} \leftarrow w'_{\hat{y}} - \alpha x
\end{array}
$$

Where $\alpha$ is some positive constant. Essentially, you want whatever features where active (non-zero) in $$x$$ to get bigger for the correct answer $y^*$ and for
the weights for the incorrect answer $w_{\hat{y}}$ to get smaller for those features active in $x$.

You can just solve for $\alpha$ which satisfy the constraint:

$$
(w'_{y^*} + \alpha x)^T x - (w'_{\hat{y}} - \alpha x)^T x
	 \geq \ell(y^*,\hat{y})
$$

Using some basic algebra we get:

$$
(w'^T_{y^*} x - w'^T_{\hat{y}} x) + 2 \alpha \| x \|^2  \geq \ell(y^*,\hat{y})
$$

Solving for $$\alpha$$ yields:

$$
\alpha \geq \frac{\ell(y^*,\hat{y}) - (  w'^T_{y^*} x - w'^T_{\hat{y}} x )}{2 \| x \|^2}
$$

Any $\alpha$ which satisfies the above will satisfy our condition. Since we need to make the smallest changes possible, this corresponds to selecting the smallest $\alpha$ which satisfies the constraint. Basically we set $\alpha$ to the right hand-side of the above.  So the $\alpha$ is composed of: the loss,  the gap with the current weights ($w'$), and the datum norm $\left( \\\| x \\\|^2 \right)$. Once we compute alpha, we make weight vector updates and move on through the rest of the examples. Notice that once we make a pass over the data and don't make an error, the weights never change.

<h1>Clojure Code</h1>
Here's the Clojure code for implementing MIRA. I implement machine learning vectors via the clojure map where the keys are typically strings given as input. This isn't the most efficient encoding, but it makes the code easier to write. This code has been fairly optimized and is reasonably fast. It can write weights to disk, load them and make predictions on new datums. In general, when I need quick and dirty multi-class classifcation, I'll use this.

One detail in the code is that it's usually better to use the average of weight vectors over all updates rather than the final weight vectors. We accomplish this by tracking the sum over all weight vectors (<code>:cum-label-weights</code>). In order to make the updates to the summed vector efficient, we need to know how many updates are left to go. This way we can add the contribution of the current update to all future updates.

{% gist 598667 %}

[^1]: Although unfortunately, like Java, you have to map to a Double object and pay the cost of boxing and unboxing.
[^2]: The variant of working with here is for $k=1$ so the update has a closed form.
[^3]: You get this by looking at the <a href="http://en.wikipedia.org/wiki/Dual_problem">dual optimization problem</a>.
