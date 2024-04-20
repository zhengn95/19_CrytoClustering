# 19_CrytoClustering
This repository utilizes Python and Unsupervised Machine Learning to predict if cryptocurrencies are affected by 24-hour or 7-day price changes.

*Introduction of Unsupervised ML:*  
Unsupervised Machine Learning refers to a subset of machine learning techniques where algorithms analyze unlabeled data to uncover patterns, structures, or relationships without explicit guidance. Unlike supervised learning, there are no predefined labels or outcomes for the algorithm to learn from. Instead, it autonomously identifies inherent structures within the data.

One prominent application of Unsupervised Machine Learning is clustering, where algorithms group similar data points together based on shared characteristics. The `K-means algorithm` is a popular choice for clustering tasks, as it partitions the data into a predetermined number of clusters, with each cluster represented by its centroid.

Determining the optimal number of clusters for K-means involves techniques like the `elbow method`, which aims to identify the point where the addition of more clusters provides diminishing returns in explaining the variance within the data.

In addition to clustering, another fundamental technique in Unsupervised ML is dimensionality reduction, with `Principal Component Analysis` (PCA) being a widely used method. PCA focuses on reducing the dimensionality of high-dimensional data while preserving variance. By identifying principal components capturing maximum variance, PCA enables a concise representation of the dataset. This streamlined data can enhance clustering algorithms like K-means, aiding in the discovery of meaningful patterns. 

In this crytoclustering repo, Python libraries `sci-kit learn` for ML, `pandas`, and `hvplot` are used to implement these algorithms, and build and deploying machine learning models. Please follow the links for documentation of the libraries:  
- [Sci-kit learn](https://scikit-learn.org/stable/user_guide.html)
- [Pandas](https://pandas.pydata.org/docs/user_guide/index.html)
- [hvPlot](https://hvplot.holoviz.org/user_guide/index.html)

Steps:
- Find the Best Value for k Using the Original Scaled DataFrame
- Cluster Cryptocurrencies with K-means Using the Original Scaled Data
- Optimize Clusters with Principal Component Analysis
- Find the Best Value for k Using the PCA Data
- Cluster Cryptocurrencies with K-means Using the PCA Data
What is the impact of using fewer features to cluster the data using K-Means?
