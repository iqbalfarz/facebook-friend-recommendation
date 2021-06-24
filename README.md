# Social Network Graph Link Prediction-Facebook Challenge
It is a kaggle competition, launched by Facebook as a recruitment test.

## Table of Contents
* [Problem Statement](#problem-statement)
* [Data Overview](#data-overview)
* [Mapping the problem into supervised learning problem](#mapping-the-problem-into-supervised-learning-problem)
* [Business objectives and constraints](#business-objectives-and-constraints)
* [Performance metric for supervised learning](#performance-metric-for-supervised-learning)
* [EDA(Exploratory Data Analysis)](#exploratory-data-analysis)
* [Posing the problem as a classification problem](#posing-the-problem-as-a-classification-problem)
* [Dataset Split](#dataset-split)
* [Feature Engineering](#feature-engineering)
* [Machine Learning Modeling](#machine-learning-modeling)
* [Team](#team)
* [Credit](#credit)


## Problem Statement
Given a directed social graph, have to prdict missing liks to recommend users(Missing Link Prediction in Graph)

## Data Overview
Taken data from facebook's recruiting challenge on kaggle:[Click to download dataset](https://www.kaggle.com/c/FacebookRecruiting/data)

Data contains two columns, source and destination for each edge in the graph.
<pre>
- Data columns (total 2 columns):
- source_node            int64
- destination_node       int64
</pre>

## Mapping the problem into Supervised Learning problem
- Generated training samples of good and abd links from given directed graph and for each link got some features like no.of followers, is he followed back, page rank, katz score, adar index, some SVD features of adjacent matrix, some weight features etc. and trained ML model based on these features to predict link.
- **Some reference papers and video**:
    - __research paper__: [Link prediction](https://www.cs.cornell.edu/home/kleinber/link-pred.pdf)
    - __research paper__: [another publications, click to read](https://www3.nd.edu/~dial/publications/lichtenwalter2010new.pdf)
    - __Video__: [click to watch](https://www.youtube.com/watch?v=2M77Hgy17cg)


## Business Objectives and constraints:
- No low-latency requirement.
- Probability of prediction is useful to recommend highest probability links

## Performance metric for supervised learning
- Both precision and recall is important so __f1-score__ is good choice.
- Confusion Matrix

__Example of sub-graph__:

![image](https://user-images.githubusercontent.com/32350208/123230967-61ee1100-d4f5-11eb-9b67-d0752b23c439.png)
## Exploratory Data Analysis

- The number of unique persons : 1862220
- Number of followers for each person:
- ![image](https://user-images.githubusercontent.com/32350208/123231157-9366dc80-d4f5-11eb-9c35-1a7f5eb20e4d.png)
- Number of people each person is following:
- ![image](https://user-images.githubusercontent.com/32350208/123231389-c9a45c00-d4f5-11eb-90cc-5c8e410cac52.png)
- ![image](https://user-images.githubusercontent.com/32350208/123231432-d1640080-d4f5-11eb-94a5-d7b5918cc57b.png)
- Both followers and following:
- ![image](https://user-images.githubusercontent.com/32350208/123231572-f2c4ec80-d4f5-11eb-8807-a197cf3b904c.png)
- ![image](https://user-images.githubusercontent.com/32350208/123231589-f8223700-d4f5-11eb-8488-83c6407ed0d5.png)


## Posing the problem as a classification problem
- We have only positive dataset means we have dataset containing columns source and destination. 
- We are creating negative dataset also to get balanced dataset of negative and positive samples.
- Generated Bad links from graph which are not in graph and whose shortest path is greater than 2.


## Dataset split
- Removed edges from Graph and used as test data and after removing, used that graph for creating features for Train and Test data.
- Splitting distribution: Train:Test=>80:20

## Feature Engineering
I created many features like:
- __Jaccard Distance__: [Click to see!](http://www.statisticshowto.com/jaccard-index/)
- __Cosine Distance__: [Click to see!](https://en.wikipedia.org/wiki/Cosine_similarity)
- __Ranking Measures__: [Click to see!](https://networkx.github.io/documentation/networkx-1.10/reference/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html)
- __Page Ranking__: [Click here!](https://en.wikipedia.org/wiki/PageRank)

- #### Other Graph Features
- __Shortest Path__: Getting Shortest path between twoo nodes, if nodes have direct path i.e directly connected then we are removing that edge and calculating path.
- __Checking for same community__:
- __Adamic/Adar Index__:
Adamic/Adar measures is defined as inverted sum of degrees of common neighbours for given two vertices.
$$A(x,y)=\sum_{u \in N(x) \cap N(y)}\frac{1}{log(|N(u)|)}$$

- __Is person was following back__: 
- __Katz Centrality__:https://en.wikipedia.org/wiki/Katz_centrality

https://www.geeksforgeeks.org/katz-centrality-centrality-measure/
 Katz centrality computes the centrality for a node 
    based on the centrality of its neighbors. It is a 
    generalization of the eigenvector centrality. The
    Katz centrality for node `i` is
 
$$x_i = \alpha \sum_{j} A_{ij} x_j + \beta,$$
where `A` is the adjacency matrix of the graph G 
with eigenvalues $$\lambda$$.

The parameter $$\beta$$ controls the initial centrality and 

$$\alpha < \frac{1}{\lambda_{max}}.$$
- __HITS Score__: The HITS algorithm computes two numbers for a node. Authorities estimates the node value based on the incoming links. Hubs estimates the node value based on outgoing links.

[Click Here!]:(https://en.wikipedia.org/wiki/HITS_algorithm)

- __SVD features for both source and destination__: 

## Machine Learning Modeling

- __Random Forest Classifier__:
- Feature Importance:
![image](https://user-images.githubusercontent.com/32350208/123243478-af23b000-d500-11eb-8f37-2c4f55240a47.png)


## Team
<a href="https://github.com/iqbal786786"><img src="https://avatars.githubusercontent.com/u/32350208?v=4" width=300></a>
|-|
[Muhammad Iqbal Bazmi](https://github.com/iqbal786786) |)

## Credit
- [Applied AI Course](https://www.appliedaicourse.com): For teaching an in-depth Machine Learning and Deep Learning.
- [Machine Learning Mastery](https://www.machinelearningmastery.com): For best Machine Learning Blog.

