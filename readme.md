# Clustering techniques

To install dependencies used by this repo, run the following in your terminal

> `pip install requirements.txt`

---

## Intro

Clustering: is the assignment of a set of observations into subsets (called clusters) so that observations in the same cluster are similar in some sense. Clustering is a method of unsupervised learning, and a common technique for statistical data analysis used in many fields.

Clustering can be done in various ways:

1. K-means
2. Hierarchal
3. DBSCAN

---

### K-means Clustering

K-means is vastly used for clustering in many data science applications,
especially useful if you need to quickly discover insights from unlabeled data.
It is only guaranteed to converge to local optima

---

### Hierarchal Clustering

As the name suggests is an algorithm that builds hierarchy of clusters. This algorithm starts with all the data points assigned to a cluster of their own. Then two nearest clusters are merged into the same cluster. In the end, this algorithm terminates when there is only a single cluster left.

Agglomerative clustering, at each iteration, the algorithm must update the distance matrix to reflect the distance
of the newly formed cluster with the remaining clusters in the forest. It performs a hierarchical clustering using a bottom up approach

Hierarchical clustering does not require a pre-specified number of clusters. However, in some applications we want a partition of disjoint clusters just as in flat clustering.

---

### Difference between K-means & Hierarchal Clustering?

Hierarchical clustering can’t handle big data well but K Means clustering can. This is because the time complexity of K Means is linear i.e. O(n) while that of hierarchical clustering is quadratic i.e. O(n2).
In K Means clustering, since we start with random choice of clusters, the results produced by running the algorithm multiple times might differ. While results are reproducible in Hierarchical clustering.
K Means is found to work well when the shape of the clusters is hyper spherical (like circle in 2D, sphere in 3D).
K Means clustering requires prior knowledge of K i.e. no. of clusters you want to divide your data into. However with HCA , you can stop at whatever number of clusters you find appropriate in hierarchical clustering by interpreting the Dendogram.

---

### DBSCAN Clustering

DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise.
This technique is one of the most common clustering algorithms which works
based on density of object. The whole idea is that if a particular point
belongs to a cluster, it should be near to lots of other points in that
cluster.

It works based on two parameters: Epsilon and Minimum Points
Epsilon determine a specified radius that if includes enough number of points
within, we call it dense area
minimumSamples determine the minimum number of data points we want in
a neighborhood to define a cluster.

---
