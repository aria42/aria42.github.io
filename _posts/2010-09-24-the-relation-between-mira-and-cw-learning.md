---
layout: post
title: The Relation between MIRA and CW Learning
categories:
- computer science
- machine learning
- nlp
tags: 3]
latex: true
status: publish
type: post
published: true
---
**Note:** This post won't make sense unless you're steeped in recent <a href="http://en.wikipedia.org/wiki/Machine_learning">machine learning</a>. There's a good chance that if you are, you already know this.

During a machine learning reading group with <a href="http://people.csail.mit.edu/mcollins/">Mike Collins</a>, <a href="http://www.stanford.edu/~jrfinkel/">Jenny Finkel</a>,  Alexander Rush and myself were reading a paper about <a href="http://www.cs.jhu.edu/~mdredze/publications/aistats10_diagfull.pdf">confidence-weighted (CW) learning</a>. At a very high-level, CW is a online learning approach: you make updates to parameters after observing a labeled examples $(x,y^*)$. Online methods have enjoyed a lot of popularity recently: the <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.61.5120&rep=rep1&type=pdf">margin infused relaxation algorithm (MIRA)</a>[^1] and <a href="http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.161.9629&rep=rep1&type=pdf">Primal Estimated sub-GrAdient SOlver for SVM (PEGASOS)</a>[^2]. These techniques are much simpler and faster to converge than their batch counterparts; the performance gap is or has become negligible (although there might be folks who disagree).

One thing we wanted to understand better is how this approach is different from MIRA. One obvious difference which the authors push is that they're capturing variance of individual features as well as between features which yields stronger performance. Those are all valid points. But if we strip the feature covariance out of the picture how does the update optimization problem differ? The answer, I think, is that they're essentially equivalent modulo one subtle difference which is probably important. This is probably obvious to machine learning gurus, but it took me a few minutes to work out. I'm sure this observation is even spelled out in one of the CW papers.  

<b>MIRA:</b> Here's the variant of MIRA I'm working with. You have a current weight vector $\mu'$ and you want to update to a new weight vector $\mu$ based on a new example pair $(x,y^*)$ :

$$
\begin{array}{l}
 \min_\mu \|\mu - \mu'\|^2  \hspace{2pt} \mbox{s.t.}  \hspace{2pt} \mu^T \Delta f \geq \gamma \\  
 \hat{y} = \arg\max_{y} \mu^T f(x,y)   \\
 \Delta f = f(x,y^*) - f(x,\hat{y})
\end{array}
$$



Here $\gamma > 0$ is typically a fixed constant and the above update is done only when an error is made $(\hat{y} \neq y^*)$.

<b>CW:</b>  In contrast, CW doesn't have a single weight vector, it has distribution over weight vectors $w \sim \mathcal{N}(\mu,\Sigma)$. In normal CW, you get the covariance matrix $\Sigma$ as parameters. Here, I'm considering a variant where the covariance matrix is fixed to be the identity.[^3] The only parameters I'm considering here are the mean weight vector $\mu$. The update optimization for CW in this context is given by:

$$
\begin{array}{l}
\min_\mu  KL(\mathcal{N}(\mu',I) | \mathcal{N}(\mu,I))\cr
\mbox{s.t.} \hspace{2pt}  P(w^T \Delta f \geq 0) \geq \eta \cr
 \hspace{15pt} w \sim \mathcal{N}(\mu, I)
\end{array}
$$


The $$KL(\cdot)$$
is the <a href="http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">Kullback-Liebler Divergence</a>. If you take a look at the expression for the KL diverence between two gaussians, <a href="http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence">here</a>, it's pretty straightforward to see that if the covariance matrices are the identity,  the KL divergence is within a constant of $\| \mu - \mu' \|^2$.


Now for the constraint. The first thing to notice is that

$$
w^{T} \Delta f \sim \mathcal{N}(\mu^{T} \Delta f, \| \Delta f \|^{2})
$$

So if $Z$ is a zero-mean unit-variance gaussian, we want

$$ P(\| \Delta f \| Z + \mu^{T} \Delta f \geq 0) = P\left(Z \geq -\frac{ \mu^{T} \Delta f}{\| \Delta f \|}\right) $$

If $\Phi$ is the cumulative distribution function for the unit-normal, we want:
$$ 1 - \Phi\left(-\frac{ \mu^{T} \Delta f}{\| \Delta f \|}\right) \geq \eta $$

This implies,

$$
	   \Phi\left(-\frac{ \mu^{T} \Delta f}{\| \Delta f \|}\right) \leq 1 - \eta \Rightarrow
	    erf\left(\frac{ \mu^{T} \Delta f}{\sqrt{2} \| \Delta f \|}\right) \leq 1 - 2\eta
$$

where $erf(\cdot)$ is the <a  href="http://en.wikipedia.org/wiki/Error_function">error function</a>.

Here's the subtlety: If we assume that our feature vectors $(x,y)$ are normalized and that for any two $y,y'$ that $f(x,y)$ and $f(x,y')$ don't overlap in non-zero features (which is common in NLP since weight vectors are partitioned for different $y$s) then $\| \Delta f \|$ is a constant independent of the particular update. In which case, ensuring $erf (c \mu^{T} \Delta f) \leq 1 - 2 \eta$ (assuming $\eta > 0.5$) just amounts to making sure $\mu^{T} \Delta f$ exceeds some constant independent of the particular update, which is equivalent to selecting that choice of $\gamma$ in MIRA.  So the two optimizations are essentially the same.

However, if feature vectors are not normalized, then the two aren't equivalent. Essentially, the larger the feature vector norm the larger the "gap" term $\mu^T \Delta f$ needs to be. If you have exclusively binary features, which many NLP applications do, this means the more features active in a datum, the larger "gap" ($\mu^T \Delta f$) we require. This makes a lot of sense. We can get this in MIRA pretty straightforwardly:

$$\begin{array}{l}
\min_\mu \|\mu - \mu'\|^2  \\
\hspace{2pt} \mbox{s.t.} \hspace{2pt} \mu^T \Delta f > \gamma \| \Delta f \| \\
\end{array} $$

then it's always equivalent modulo constant choices. I don't actually know if the $ \\\| \Delta f \\\|$ scaling improves accuracy, but I wouldn't be surprised if it did.

[^1]: This algorithm is actually called the Passive Aggressive algorithm, but I've always known it as MIRA.
[^2]: Yes I know, it's a tortured <a href="http://en.wikipedia.org/wiki/Backronym">backronym</a>
[^3]: In practice, you wouldn't use this variant, here I'm just trying to get a handle on the objective
