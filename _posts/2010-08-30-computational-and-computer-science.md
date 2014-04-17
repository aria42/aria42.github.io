---
layout: post
title: Computer and Computational  Science
categories:
- computer science
tags:
- computer science
- thoughts
status: publish
type: post
published: true
---
There's a divide I've noticed amongst people lumped into a "computer science" department. Compactly, I think there are computer scientists and computational scientists; the knowledge base of these groups is rapidly diverging and CS departments should do a better job catering to each's needs.

So what exactly is the difference? Well, it's definitely a fuzzy distinction, but essentially a computational scientist works with data and her primary job is extracting useful information from it. Typically, a computational scientist requires a significant amount of statistical knowledge as well as usually a lot of knowledge from a particular domain in order to make use of data.

Take myself for example: my specialty is <a href="http://en.wikipedia.org/wiki/Natural_language_processing">statistical natural language processing</a>. My research essentially involves inducing structured data
from unstructured language data and this requires <em>far more</em> knowledge about statistics and linguistics than it does expertise with computer architecture, databases, or systems.

A computer scientist, on the other hand, well, is a computer scientist. Her daily bread is understanding the science of how computers run: low-level operating and embedded systems, tuning a database, scaling a web server, etc. So for instance, a post <a href="http://al3x.net/2010/07/27/node.html">like this</a> is all about the computer science.

Now, most computational scientists have to know a little bit about computer science in order to implement what it is she wants to do with data. Increasingly though,  advances, made by computer scientists, have enabled data scientists to do their job at higher levels of abstractions without having to think much about what computer scientists think about. These improvements range from the fact that you can make performant systems in <a href="http://clojure.org">higher-level languages</a>  to frameworks like <a href="http://hadoop.apache.org/">Hadoop</a> that let a computational scientist focus on data and her domain.

There is plenty the areas share which justifies putting them in the same department: much of standard algorithms and computational theory I believe are still broadly relevant to both areas. Procedural thinking, for better or worse, is at the foundation of computer science as well as how we think about doing things with data.

Thinking about data and how to use it certainly isn't new; statisticians have been doing it for centuries. What is new is the availability of large data and a focus on what actionable decision should be made with it. Computational science has certainly enjoyed a lot of recent success and growth. The New York Times recently called the area the <a href="http://www.nytimes.com/2009/08/06/technology/06stats.html">new sexy job</a>. The number of areas which can make use of computational science is growing and will continue to do so for a long time. Computational science will, hopefully, still be a big part of CS departments for a long time to come.

Here's the issue though. I don't think the educational curriculum of CS departments has adjusted itself to this growing area. Machine learning isn't a standard part of the undergraduate curriculum; some instructors have converted their Artificial Intelligence courses into ML ones, but those aren't always required either. A statistics course isn't typically required; and no, bundling probability theory into the tail end of a  discrete math course doesn't count. I mean a course where a student does basic analytics on a larg-ish dataset, including things such as simple statistical tests, which are useful in a lot of <a href="http://www.niemanlab.org/2009/10/how-the-huffington-post-uses-real-time-testing-to-write-better-headlines/">surprising contexts</a>. Many universities require physics and  EE courses for the computer scientists, where is the equivalent statistics course for computational scientists?

A related problem with the standard CS (conflating computer and computational) curriculum is that it doesn't really convey the broad range of potential CS applications: social science, biology, law, finance, linguistics, astronomy, even <a href="http://aclweb.org/anthology-new/P/P10/P10-1015.pdf">comparative literature</a>. I think this is one of the most exciting things about doing CS and exploring these applications is important for budding young computational scientists. However, early  CS courses focus on the nuts &amp; bolts important to computer scientists: programming language details, data structures, low-level memory management, etc. I'm not sure it's a fair analogy, but it's as though your first year biology course focused on the structure and use of lab equipment; this week: bunsen burners. Clearly, you need a little computer science to do computational science, but I don't think it needs to be buried so deep in the curriculum.
