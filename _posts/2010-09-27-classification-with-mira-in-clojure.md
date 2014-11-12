---
layout: post

title: Classification with Mira In Clojure
# subtitle: "Mira explained"
excerpt: "A brief introduction to passive-agressive algorithm (sometimes erroneously called Mira)..."
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
\begin{aligned}
\min_w & \frac{1}{2}\|w_{y^*} - w_{y^*}'\|^2 + \frac{1}{2}\|w_{\hat{y}} - w_{\hat{y}}'\|^2   \\
 \mbox{s.t.} &  w^T_{y^*} x - w^T_{\hat{y}} x \geq \ell(y^*,\hat{y}) \\  
 & \hat{y} = \arg\max_{y} w_y'^T x
\end{aligned}
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

{% highlight clojure %}
  (ns mira  
  {:doc "Implements margin-infused relaxation algorithm (MIRA)
         multi-class classifcation Fairly optimized."
   :author "Me <me@aria42.com>"}
  (:gen-class)
  (:use [clojure.string :only [join]]
        [clojure.java.io :only [reader]]))

(defn dot-product
  "dot-product between two maps (sum over matching values)
   Bottleneck: written to be efficient"
  [x y]  
  (loop [sum 0.0 y y]
    (let [f (first y)]
      (if-not f sum
        (let [k (first f)  v (second f)]
          (recur (+ sum (* (get x k 0.0) v))
                 (rest y)))))))

(defn sum [f xs]
  (reduce + (map f xs)))

(defn norm-sq
  "||x||^2 over values in map x"
  [x] (sum #(* % %) (map second x)))

(defn add-scaled
 "x <- x + scale * y
  Bottleneck: written to be efficient"
 [x scale y]
 (persistent!
  (reduce
    (fn [res elem]
      (let [k (first elem) v (second elem)]
         (assoc! res k (+ (get x k 0.0) (* scale v)))))
     (transient x)
     y)))

; Needed for averaged weight vector
(def +updates-left+ (atom nil))

; (cum)-label-weights: label -> (cum)-weights
(defrecord Mira [loss-fn label-weights cum-label-weights])

(defn new-mira
  [labels loss-fn]
  (let [empty-weights #(into {} (for [l labels] [l {}]))]
    (Mira. loss-fn (empty-weights) (empty-weights))))

(defn get-labels
  "return possible labels for task"
  [mira]  (keys (:label-weights mira)))  

(defn get-score-fn
  "return fn: label => model-score-of-label"
  [mira datum]
  (fn [label]
    (dot-product ((:label-weights mira) label) datum)))

(defn get-loss
  "get loss for predicting predict-label
   in place of gold-label"
  [mira gold-label predict-label]
  ((:loss-fn mira) gold-label predict-label))

(defn ppredict
   "When you have lots of classes,  useful to parallelize prediction"
  [mira datum]
  (let [score-fn (get-score-fn mira datum)
        label-parts (partition-all 5 (get-labels mira))
        part-fn (fn [label-part]
                  (reduce
                    (fn [res label]
                       (assoc res label (score-fn label)))
                    {} label-part))
        score-parts (pmap part-fn label-parts)
        scores (apply merge score-parts)]
    (first (apply max-key second scores))))

(defn predict
  "predict highest scoring class"
  [mira datum]
  (if (> (count (get-labels mira)) 5)
    (ppredict mira datum)
    (apply max-key (get-score-fn mira datum) (get-labels mira))))

(defn update-weights
  "returns new weights assuming error predict-label instead of gold-label.
   delta-vec is the direction and alpha the scaling constant"
  [label-weights delta-vec gold-label predict-label alpha]  
  (->  label-weights
       (update-in [gold-label]  add-scaled alpha delta-vec)
       (update-in [predict-label] add-scaled (- alpha) delta-vec)))

(defn update-mira
  "update mira for an example returning [new-mira error?]"
  [mira datum gold-label]
  (let [predict-label (predict mira datum)]
       (if (= predict-label gold-label)
            ; If we get it right do nothing
            [mira false]
            ; otherwise, update weights
            (let [score-fn (get-score-fn mira datum)
                  loss (get-loss mira gold-label predict-label)
                  gap (- (score-fn gold-label) (score-fn predict-label))
                  alpha  (/ (- loss  gap) (* 2 (norm-sq datum)))
                  avg-factor (* @+updates-left+ alpha)
                  new-mira (-> mira
                            ; Update Current Weights
                            (update-in [:label-weights]
                              update-weights datum gold-label
                                    predict-label alpha)
                            ; Update Average (cumulative) Weights  
                            (update-in [:cum-label-weights]
                              update-weights datum gold-label
                              predict-label avg-factor))]
              [new-mira true]))))

(defn train-iter
  "Training pass over data, returning [new-mira num-errors], where
   num-errors is the number of mistakes made on training pass"
  [mira labeled-data-fn]
   (reduce
     (fn [[cur-mira num-errors] [datum gold-label]]
       (let [[new-mira error?]
              (update-mira cur-mira datum gold-label)]
          (swap! +updates-left+ dec)
          [new-mira (if error? (inc num-errors) num-errors)]))
     [mira 0]
     (labeled-data-fn)))

(defn train
  "do num-iters iterations over labeled-data (yielded by labeled-data-fn)"
  [labeled-data-fn labels num-iters loss-fn]
  (loop [iter 0 mira (new-mira labels loss-fn)]
    (if (= iter num-iters)
        mira
        (let [[new-mira num-errors]  (train-iter mira labeled-data-fn)]
          (println
            (format "[MIRA] On iter %s made %s training mistakes"
                    iter num-errors))
          ; If we don't make mistakes, never will again  
          (if (zero? num-errors)
            new-mira (recur (inc iter) new-mira))))))

(defn feat-vec-from-line
  "format: feat1:val1 ... featn:valn. feat is a string and val a double"
  [#^String line]
  (for [#^String piece (.split line "\\s+")
        :let [split-index (.indexOf piece ":")
              feat (if (neg? split-index)
                      piece
                      (.substring piece 0 split-index))
              value (if (neg? split-index) 1
                      (-> piece (.substring (inc split-index))
                          Double/parseDouble))]]
    [feat value]))

(defn load-labeled-data
  "format: label feat1:val1 .... featn:valn"
  [path]
  (for [line (line-seq (reader path))
        :let [pieces (.split #^String line "\\s+")
              label (first pieces)
              feat-vec (feat-vec-from-line
                          (join " " (rest pieces)))]]
    [feat-vec label]))

(defn load-data
  "load data without label"
  [path] (map feat-vec-from-line (line-seq (reader path))))

(defn normalize-vec [x]
  (let [norm (Math/sqrt (norm-sq x))]
    (into {} (for [[k v] x] [k (/ v norm)]))))

(defn -main [& args]
  (case (first args)
    "train"
      (let [[data-path num-iters outfile] (rest args)
            labeled-data-fn #(load-labeled-data data-path)
            labels (into #{} (map second (labeled-data-fn)))
            num-iters (Integer/parseInt num-iters)]
        ; For Average Weight Calculation
        (compare-and-set! +updates-left+ nil
            (* num-iters (count (labeled-data-fn))))
        (let [mira (train labeled-data-fn labels num-iters  (constantly 1))
              avg-weights
                (into {}
                  (for [[label sum-weights] (:cum-label-weights mira)]
                    [label (normalize-vec sum-weights)]))]
          (println "[MIRA] Done Training. Writing weights to " outfile)
          (spit outfile avg-weights)))
    "predict"
      (let [[weight-file data-file] (rest args)
            weights (read-string (slurp weight-file))
            mira (Mira.  (constantly 1) weights weights)]
        (doseq [datum (load-data data-file)]
          (println (predict mira datum))))
    "test"
      (let [[weight-file data-file] (rest args)
            weights (read-string (slurp weight-file))
            mira (Mira. (constantly 1) weights weights)
            labeled-test (load-labeled-data data-file)
            gold-labels (map second labeled-test)
            predict-labels (map #(predict mira %) (map first labeled-test))
            num-errors (->> (map vector gold-labels predict-labels)
                            (sum (fn [[gold predict]] (if (not= gold predict) 1 0))))]
        (println "Error: " (double (/ num-errors (count gold-labels))))))
    (shutdown-agents))
{% endhighlight   %}

[^1]: Although unfortunately, like Java, you have to map to a Double object and pay the cost of boxing and unboxing.
[^2]: The variant of working with here is for $k=1$ so the update has a closed form.
[^3]: You get this by looking at the <a href="http://en.wikipedia.org/wiki/Dual_problem">dual optimization problem</a>.
