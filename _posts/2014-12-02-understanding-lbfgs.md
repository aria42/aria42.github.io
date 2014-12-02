---
layout: post
title: "Numerical Optimization: Understanding L-BFGS"
latex: true
date: 2014-12-02
excerpt: "Numerical optimization is at the core of much of machine learning. In this post, we derive the L-BFGS algorithm, commonly used in batch machine learning applications."
---

<div style="display:none">
$$
\newcommand{\hessian}{\mathbf{H}}
\newcommand{\grad}{\mathbf{g}}
\newcommand{\invhessian}{\mathbf{H}^{-1}}
\newcommand{\qquad}{\hspace{1em}}
$$
</div>

Numerical optimization is at the core of much of machine learning. Once you've defined your model and have a dataset ready, estimating the parameters of your model typically boils down to minimizing some [multivariate function] $f(x)$, where the input $x$ is in some high-dimensional space and corresponds to model parameters. In other words, if you solve:

$$
  x^* = \arg\min_x f(x)
$$

then $x^*$ is the 'best' choice for model parameters according to how you've set your objective.[^global-min]

In this post, I'll focus on the motivation for the [L-BFGS] algorithm for unconstrained function minimization, which is very popular for ML problems where 'batch'  optimization makes sense. For larger problems, online methods based around [stochastic gradient descent] have gained popularity, since they require fewer passes over data to converge. In a later post, I might cover some of these techniques, including my personal favorite [AdaDelta].

__Note__: Throughout the post, I'll assume you remember multivariable calculus. So if you don't recall what a [gradient] or [Hessian] is, you'll want to bone up first.

!["Illustration of iterative function descent"](/images/steepest-descent.png "Illustration of iterative function descent")

# Newton's Method

Most numerical optimization procedures are iterative algorithms which consider a sequence of 'guesses' $x_n$ which ultimately converge to $x^*$ the true global minimizer of $f$. Suppose, we have an estimate $x_n$ and we want our next estimate $x_{n+1}$ to have the property that $$f(x_{n+1}) < f(x_n)$$.

Newton's method is centered around a quadratic approximation of $f$ for points near $x_n$.
Assuming that $f$ is twice-differentiable, we can use a quadratic approximation of $f$ for points 'near' a fixed point $x$ using a [Taylor expansion]:

$$
\begin{align}
f(x + \Delta x)
&\approx f(x) + \Delta x^T \nabla f(x)  + \frac{1}{2} \Delta x^T \left( \nabla^2 f(x) \right)  \Delta x
\end{align}
$$

where $\nabla f(x)$ and $\nabla^2 f(x)$ are the gradient and Hessian of $f$ at the point $x_n$. This approximation holds in the limit as $\|\| \Delta x \|\| \rightarrow 0$. This is a generalization of the single-dimensional Taylor polynomial expansion you might remember from Calculus.

In order to simplify much of the notation, we're going to think of our iterative algorithm of producing a sequence of such quadratic approximations $h_n$. Without loss of generality, we can write $$x_{n+1} = x_n + \Delta x$$ and re-write the above equation,

$$
\begin{align}
h_n(\Delta x) &= f(x_n) + \Delta x^T \grad_n + \frac{1}{2} \Delta x^T \hessian_n  \Delta x
\end{align}
$$

where $\grad_n$ and $\hessian_n$ represent the gradient and Hessian of $f$ at $x_n$.

We want to choose $\Delta x$ to minimize this local quadratic approximation of $f$ at $x_n$. Differentiating with respect to $\Delta x$ above yields:

$$
\begin{align}
\frac{\partial h_n(\Delta x)}{\partial \Delta x} = \grad_n + \hessian_n \Delta x
\end{align}
$$

Recall that any $\Delta x$ which yields $\frac{\partial h_n(\Delta x)}{\partial \Delta x} = 0$ is a local extrema of $h_n(\cdot)$. If we assume that $\hessian_n$ is [postive semi-definite] (psd) then we know this $\Delta x$ is also a global minimum for $h_n(\cdot)$. Solving for $\Delta x$:[^why-global]

$$
\Delta x = - \invhessian_n \grad_n
$$

This suggests $\invhessian_n \grad_n$ as a good direction to move $x_n$ towards. In practice, we set $$x_{n+1} = x_n - \alpha (\invhessian_n \grad_n)$$ for a value of $\alpha$ where $f(x_{n+1})$ is  'sufficiently' smaller than $f(x_n)$.


## Iterative Algorithm

The above suggests an iterative algorithm:

$$
\begin{align}
 & \mathbf{NewtonRhapson}(f,x_0): \\
 & \qquad \mbox{For $n=0,1,\ldots$ (until converged)}: \\
 & \qquad \qquad \mbox{Compute $\grad_n$ and $\invhessian_n$ for $x_n$} \\
 & \qquad \qquad d = \invhessian_n \grad_n \\
 & \qquad \qquad \alpha = \min_{\alpha \geq 0} f(x_{n} - \alpha d) \\
 & \qquad \qquad x_{n+1} \leftarrow x_{n} - \alpha d
\end{align}
$$

The computation of the $\alpha$ step-size can use any number of [line search] algorithms. The simplest of these is [backtracking line search], where you simply try smaller and smaller values of $\alpha$ until the function value is 'small enough'.

In terms of software engineering, we can treat $\mathbf{NewtonRhapson}$ as a blackbox for any twice-differentiable function which satisfies the Java interface:

{% highlight java %}
public interface TwiceDifferentiableFunction {
  // compute f(x)
  public double valueAt(double[] x);

  // compute grad f(x)
  public double[] gradientAt(double[] x);

  // compute inverse hessian H^-1
  public double[][] inverseHessian(double[] x);
}
{% endhighlight %}

With quite a bit of tedious math, you can prove that for a [convex function], the above procedure will converge to a unique global minimizer $x^*$, regardless of the choice of $x_0$. For non-convex functions that arise in ML (almost all latent variable models or deep nets), the procedure still works but is only guranteed to converge to a local minimum. In practice, for non-convex optimization, users need to pay more attention to initialization and other algorithm details.

## Huge Hessians

The central issue with $\mathbf{NewtonRhapson}$ is that we need to be able to compute the inverse Hessian matrix.[^implicit-multiply] Note that for ML applications, the dimensionality of the input to $f$ typically corresponds to model parameters. It's not unusual to have hundreds of millions of parameters or in some vision applications even [billions of parameters]. For these reasons, computing the hessian or its inverse is often impractical. For many functions, the hessian may not even be analytically computable, let along representable.

Because of these reasons, $\mathbf{NewtonRhapson}$ is rarely used in practice to optimize functions corresponding to large problems. Luckily, the above algorithm can still work even if $\invhessian_n$ doesn't correspond to the exact inverse hessian at $x_n$, but is instead a good approximation.

# Quasi-Newton

Suppose that instead of requiring $\invhessian_n$ be the inverse hessian at $x_n$, we think of it as an approximation of this information. We can generalize $\mathbf{NewtonRhapson}$ to take a $\mbox{QuasiUpdate}$ policy which is responsible for producing a sequence of $\invhessian_n$.  

$$
\begin{align}
& \mathbf{QuasiNewton}(f,x_0, \invhessian_0, \mbox{QuasiUpdate}): \\
& \qquad \mbox{For $n=0,1,\ldots$ (until converged)}: \\
& \qquad \qquad \mbox{// Compute search direction and step-size } \\
& \qquad \qquad d = \invhessian_n \grad_n \\
& \qquad \qquad \alpha \leftarrow \min_{\alpha \geq 0} f(x_{n} - \alpha d) \\
& \qquad \qquad x_{n+1} \leftarrow x_{n} - \alpha d \\
& \qquad \qquad \mbox{// Store the input and gradient deltas } \\
& \qquad \qquad \grad_{n+1} \leftarrow \nabla f(x_{n+1}) \\
& \qquad \qquad s_{n+1} \leftarrow x_{n+1} - x_n \\
& \qquad \qquad y_{n+1} \leftarrow \grad_{n+1} - \grad_n \\
& \qquad \qquad \mbox{// Update inverse hessian } \\
& \qquad \qquad \invhessian_{n+1} \leftarrow \mbox{QuasiUpdate}(\invhessian_{n},s_{n+1}, y_{n+1})
\end{align}
$$

We've assumed that $\mbox{QuasiUpdate}$ only requires the former inverse hessian estimate as well tas the input and gradient differences ($s_n$ and $y_n$ respectively). Note that if $\mbox{QuasiUpdate}$ just returns $\nabla^2 f(x_{n+1})$, we recover exact $\mbox{NewtonRhapson}$.

In terms of software, we can blackbox optimize an arbitrary differentiable function (with no need to be able to compute a second derivative) using $\mathbf{QuasiNewton}$ assuming we get a quasi-newton approximation update policy. In Java this might look like this,

{% highlight java %}
public interface DifferentiableFunction {
  // compute f(x)
  public double valueAt(double[] x);

  // compute grad f(x)
  public double[] gradientAt(double[] x);  
}

public interface QuasiNewtonApproximation {
  // update the H^{-1} estimate (using x_{n+1}-x_n and grad_{n+1}-grad_n)
  public void update(double[] deltaX, double[] deltaGrad);

  // H^{-1} (direction) using the current H^{-1} estimate
  public double[] inverseHessianMultiply(double[] direction);
}
{% endhighlight %}

Note that the only use we have of the hessian is via it's product with the gradient direction. This will become useful for the L-BFGS algorithm described below, since we don't need to represent the Hessian approximation in memory. If you want to see these abstractions in action, here's a link to a [Java 8](https://github.com/aria42/java8-optimize/tree/master/src/optimize) and  [golang](https://github.com/aria42/taskar/blob/master/optimize/newton.go) implementation I've written.

## Behave like a Hessian

What form should $\mbox{QuasiUpdate}$ take? Well, if we have $\mbox{QuasiUpdate}$ always return the identity matrix (ignoring its inputs), then this corresponds to simple [gradient descent], since the search direction is always $\nabla f_n$. While this actually yields a valid procedure which will converge to $x^*$ for convex $f$, intuitively this choice of $\mbox{QuasiUpdate}$ isn't attempting to capture second-order information about $f$.

Let's think about our choice of $$\hessian_{n}$$ as an approximation for $f$ near $x_{n}$:

$$
h_{n}(d) = f(x_{n}) + d^T \grad_{n} + \frac{1}{2} d^T \hessian_{n} d
$$

### Secant Condition

A good property for $$h_{n}(d)$$ is that its gradient agrees with $f$ at $x_n$ and $x_{n-1}$. In other words, we'd like to ensure:

$$
\begin{align}
\nabla h_{n}(x_{n}) &= \grad_{n} \\
\nabla h_{n}(x_{n-1}) &= \grad_{n-1}\\
\end{align}
$$

Using both of the equations above:

$$
\nabla h_{n}(x_{n}) - \nabla h_{n}(x_{n-1}) = \grad_{n} - \grad_{n-1}
$$

Using the gradient of $h_{n+1}(\cdot)$ and canceling terms we get

$$
\hessian_{n}(x_{n} - x_{n-1}) = (\grad_{n} - \grad_{n-1}) \\
$$

This yields the so-called "secant conditions" which ensures that $\hessian_{n+1}$ behaves like the Hessian at least for the diference $$(x_{n} - x_{n-1})$$. Assuming $$\hessian_{n}$$ is invertible (which is true if it is psd), then multiplying both sides by $$\invhessian_{n}$$ yields

$$
\invhessian_{n} \mathbf{y}_{n}   = \mathbf{s}_{n}
$$

where $$\mathbf{y}_{n+1}$$ is the difference in gradients and $$\mathbf{s}_{n+1}$$ is the difference in inputs.

### Symmetric

Recall that the a hessian represents the matrix of 2nd order partial derivatives: $\hessian^{(i,j)} = \partial f / \partial x_i \partial x_j$. The hessian is symmetric since the order of differentiation doesn't matter.

### The BFGS Update

Intuitively, we want $\hessian_n$ to satisfy the two conditions above:

- Secant condition holds for $\mathbf{s}_n$ and $\mathbf{y}_n$
- $\hessian_n$ is symmetric

Given the two conditions above, we'd like to take the most conservative change relative to $\hessian_{n-1}$. This is reminiscent of the [MIRA update], where we have conditions on any good solution but all other things equal, want the 'smallest' change.

$$
\begin{aligned}
\min_{\invhessian} & \hspace{0.5em} \| \invhessian - \invhessian_{n-1} \|^2 \\
\mbox{s.t. } & \hspace{0.5em} \invhessian \mathbf{y}_{n}   = \mathbf{s}_{n} \\
            & \hspace{0.5em} \invhessian \mbox{ is symmetric }
\end{aligned}
$$

The norm used here $$\| \cdot \|$$ is the [weighted frobenius norm].[^weighted-norm] The solution to this optimization problem is given by

$$
\invhessian_{n+1} = (I - \rho_n y_n s_n^T) \invhessian_n (I - \rho_n s_n y_n^T) + \rho_n s_n s_n^T
$$

where $\rho_n = (y_n^T s_n)^{-1}$. Proving this is relatively involved and mostly symbol crunching. I don't know of any intuitive way to derive this unfortunately.

<img src="/images/bfgs.png">

This update is known as the Broyden–Fletcher–Goldfarb–Shanno (BFGS) update, named after the original authors. Some things worth noting about this update:

* $\invhessian_{n+1}$ is positive semi-definite (psd) when $\invhessian_n$ is. Assuming our initial guess of  $\hessian_0$ is psd, it follows by induction each inverse Hessian estimate is as well. Since we can choose any $\invhessian_0$ we want, including the $I$ matrix, this is easy to ensure.

* The above also specifies a recurrence relationship between $$\invhessian_{n+1}$$ and $$\invhessian_{n}$$. We only need the history of $$s_n$$ and $$y_n$$ to re-construct $$\invhessian_n$$.

The last point is significant since it will yield a procedural algorithm for computing $\invhessian_n d$, for a direction $d$, without ever forming the $\invhessian_n$ matrix. Repeatedly applying the recurrence above we have

$$
\begin{align}
& \mathbf{BFGSMultiply}(\invhessian_0, \{s_k\}, \{y_k\}, d): \\
& \qquad r \leftarrow d \\
& \qquad \mbox{// Compute right product} \\
& \qquad \mbox{for $i=n,\ldots,1$}: \\
& \qquad \qquad \alpha_i \leftarrow \rho_{i} s^T_i r \\
& \qquad \qquad r \leftarrow r - \alpha_i y_i \\
& \qquad \mbox{// Compute center} \\
& \qquad r \leftarrow \invhessian_0 r \\
& \qquad \mbox{// Compute left product} \\
& \qquad \mbox{for $i=1,\ldots,n$}: \\
& \qquad \qquad \beta \leftarrow \rho_{i} y^T_i r \\
& \qquad \qquad r \leftarrow r + (\alpha_{n-i+1}-\beta)s_i \\
& \qquad \mbox{return $r$}
\end{align}
$$

Since the only use for $\invhessian_n$ is via the product $\invhessian_n \grad_n$, we only need the above procedure to use the BFGS approximation in $\mbox{QuasiNewton}$.


### L-BFGS: BFGS on a memory budget

The BFGS quasi-newton approximation has the benefit of not requiring us to be able to analytically compute the Hessian of a function. However, we still must maintain a history of the $s_n$ and $y_n$ vectors for each iteration. Since one of the core-concerns of the $\mathbf{NewtonRhapson}$ algorithm were the memory requirements associated with maintaining an Hessian, the BFGS Quasi-Newton algorithm doesn't address that since our memory use can grow without bound.

The L-BFGS algorithm, named for _limited_ BFGS, simply truncates the $$\mathbf{BFGSMultiply}$$ update to use the last $m$ input differences and gradient differences. This means, we only need to store $$s_n, s_{n-1},\ldots, s_{n-m-1}$$ and $$y_n, y_{n-1},\ldots, y_{n-m-1}$$ to compute the update. The center product can still use any symmetric psd matrix $$\invhessian_0$$, which can also depend on any $$\{s_k\}$$ or $$\{ y_k \}$$.

# L-BFGS variants

There are lots of variants of L-BFGS which get used in practice. For non-differentiable functions, there is an [othant-wise varient] which is suitable for training $L_1$ regularized loss.

One of the main reasons to _not_ use L-BFGS is in very large data-settings where an online approach can converge faster. There are in fact [online variants] of L-BFGS, but to my knowledge, none have consistently out-performed SGD variants (including [AdaGrad] or AdaDelta) for sufficiently large data sets.


<!-- Footnotes and Links -->

[^global-min]:This assumes there is a unique global minimizer for $f$. In practice, in practice unless $f$ is convex, the parameters used are whatever pops out the other side of an iterative algorithm.
[^why-global]: We know $- \invhessian \nabla f$ is a local extrema since the gradient is zero, since the Hessian has positive curvature, we know it's in fact a local minima. If $f$ is convex, we know the Hessian is always positive semi-definite and we know there is a single unique global minimum.
[^implicit-multiply]: As we'll see, we really on require being able to multiply by $\invhessian d$ for a direction $d$.
[^weighted-norm]: I've intentionally left the weighting matrix $W$ used to weight the norm since you get the same solution under many choices. In particular for any positive-definite $W$ such that $W s_n = y_n$, we get the same solution.
[AdaGrad]: http://www.magicbroom.info/Papers/DuchiHaSi10.pdf
[othant-wise varient]: http://research.microsoft.com/en-us/um/people/jfgao/paper/icml07scalable.pdf
[MIRA update]: http://aria42.com/blog/2010/09/classification-with-mira-in-clojure/
[postive semi-definite]: http://en.wikipedia.org/wiki/Positive-definite_matrix
[line search]:http://en.wikipedia.org/wiki/Line_search
[backtracking line search]:http://en.wikipedia.org/wiki/Backtracking_line_search
[Taylor expansion]: http://en.wikipedia.org/wiki/Taylor_series
[L-BFGS]: http://en.wikipedia.org/wiki/Limited-memory_BFGS
[online variants]: http://jmlr.org/proceedings/papers/v2/schraudolph07a/schraudolph07a.pdf
[multivariate function]:http://en.wikipedia.org/wiki/Multivariable_calculus
[stochastic gradient descent]:http://en.wikipedia.org/wiki/Stochastic_gradient_descent
[AdaDelta]:http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
[gradient]:http://en.wikipedia.org/wiki/Gradient
[Hessian]:http://en.wikipedia.org/wiki/Hessian_matrix
[Newton's method]:http://en.wikipedia.org/wiki/Newton%27s_method
[billions of parameters]: http://static.googleusercontent.com/media/research.google.com/en/us/archive/large_deep_networks_nips2012.pdf
[gradient descent]: http://en.wikipedia.org/wiki/Gradient_descent
[convex function]: http://en.wikipedia.org/wiki/Convex_function
[weighted frobenius norm]: http://mathworld.wolfram.com/FrobeniusNorm.html
[rank]: http://en.wikipedia.org/wiki/Rank_%28linear_algebra%29
