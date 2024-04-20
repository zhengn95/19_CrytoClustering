# 19_CrytoClustering
This repository utilizes Python and Unsupervised Machine Learning to predict if cryptocurrencies are affected by 24-hour or 7-day price changes.

### Introduction of Unsupervised ML:
Unsupervised Machine Learning refers to a subset of machine learning techniques where algorithms analyze unlabeled data to uncover patterns, structures, or relationships without explicit guidance. Unlike supervised learning, there are no predefined labels or outcomes for the algorithm to learn from. Instead, it autonomously identifies inherent structures within the data.

One prominent application of Unsupervised Machine Learning is clustering, where algorithms group similar data points based on shared characteristics. The `K-means algorithm` is a popular choice for clustering tasks, as it partitions the data into a predetermined number of clusters, with each cluster represented by its centroid.

Determining the optimal number of clusters for K-means involves techniques like the `elbow method`, which aims to identify the point where the addition of more clusters provides diminishing returns in explaining the variance within the data.

In addition to clustering, another fundamental technique in Unsupervised ML is dimensionality reduction, with `Principal Component Analysis` (PCA) being a widely used method. PCA focuses on reducing the dimensionality of high-dimensional data while preserving variance. By identifying principal components capturing maximum variance, PCA enables a concise representation of the dataset. This streamlined data can enhance clustering algorithms like K-means, aiding in the discovery of meaningful patterns. 

In this crytoclustering repo, Python libraries `sci-kit learn` for ML, `pandas`, and `hvplot` are used to implement these algorithms, and build and deploying machine learning models. Please follow the links for documentation of the libraries:  
- [Sci-kit learn](https://scikit-learn.org/stable/user_guide.html)
- [Pandas](https://pandas.pydata.org/docs/user_guide/index.html)
- [hvPlot](https://hvplot.holoviz.org/user_guide/index.html)

### Data Analysis: K-means with the Original Scaled Data
**Step 1:** Preparing the Data:  
- `StandardScaler()` module from scikit-learn to scale and standardize the `crypto_market_data.csv` data.
- Created a DataFrame with the scaled data and set the `"coin_id"` index from the original DataFrame as the index for the new DataFrame.
**Step 2:** Find the Best Value for k using the `elbow method` the scaled data:
- Create a list with the number of k values from 1 to 11.
- Create an empty list to store the inertia values.
- Create a for loop to compute the inertia with each possible value of k.
- Create a dictionary with the data to plot the elbow curve.
- Plot a line chart with all the inertia values computed with the different values of k to visually identify the optimal value for k.
Answer the following question in your notebook: What is the best value for k?
**Answer:** Based on the elbow curve, k=3 or k=4 appear to be good values to use.

**Step 3:** Cluster Cryptocurrencies with K-means with the scaled data
Use the following steps to cluster the cryptocurrencies for the best value for k on the original scaled data:
- Initialize the K-means model with the best value for k.
- Fit the K-means model using the original scaled DataFrame.
- Predict the clusters to group the cryptocurrencies using the original scaled DataFrame.
- Create a copy of the original data and add a new column with the predicted clusters.
- Create a scatter plot using hvPlot as follows:
- Set the x axis to `["price_change_percentage_24h"]` and y-axis to `["price_change_percentage_7d]`
- Add the "coin_id" column in the hover_cols parameter to identify the cryptocurrency represented by each data point.
  
### Data Anaylsis: K-means with reduced PCA data
**Step 4:** Optimize Clusters with Principal Component Analysis (PCA)
Using the original scaled DataFrame, perform a PCA and reduce the features to three principal components.
Retrieve the explained variance to determine how much information can be attributed to each principal component and then answer the following question in your notebook:
What is the total explained variance of the three principal components?
Answer: To calculate the total explained variance of the three principal components, you simply sum up the explained variance ratios. Total Explained Variance=0.37269822+0.32489961+0.18917649 Total Explained Variance=0.88677432

The total explained variance of the three principal components is approximately 0.8868 or 88.68% of the total variance in the data.
Create a new DataFrame with the PCA data and set the "coin_id" index from the original DataFrame as the index for the new DataFrame.  

**Step 5:** Find the Best Value for k Using the PCA Data
Use the elbow method on the PCA data to find the best value for k using the following steps:

Create a list with the number of k-values from 1 to 11.
Create an empty list to store the inertia values.
Create a for loop to compute the inertia with each possible value of k.
Create a dictionary with the data to plot the Elbow curve.
Plot a line chart with all the inertia values computed with the different values of k to visually identify the optimal value for k.
Answer the following question in your notebook:
What is the best value for k when using the PCA data?
**Answer:** Based on this Elbow Curve, it looks like k=4 is the correct one.
Does it differ from the best k value found using the original data?  
**Answer:** Based on this Elbow Curve for the PCA data, it us more obvious that `k=4` is more correct than `k=3`.

**Step 6:** Cluster Cryptocurrencies with K-means Using the PCA Data
Use the following steps to cluster the cryptocurrencies for the best value for k on the PCA data:
Initialize the K-means model with the best value for k.
Fit the K-means model using the PCA data.
Predict the clusters to group the cryptocurrencies using the PCA data.
Create a copy of the DataFrame with the PCA data and add a new column to store the predicted clusters.
Create a scatter plot using hvPlot as follows:
Set the x-axis as "price_change_percentage_24h" and the y-axis as "price_change_percentage_7d".
Color the graph points with the labels found using K-means.
Add the "coin_id" column in the hover_cols parameter to identify the cryptocurrency represented by each data point.

### Conclusion: Original Scaled Data vs. PCA Data
After visually analyzing the cluster analysis results, what is the impact of using fewer features to cluster the data using K-Means?  
Using fewer features for clustering data with K-Means can have several impacts:
- **Reduced Dimensionality:** It reduces the dimensionality of the data, which, in this case, makes it easier to visualize and interpret the crypto clusters.
- **Sensitivity to Noise:** The algorithm is less affected by irrelevant features, leading to more robust or obvious clusters.
- **Loss of Information:** May result in a loss of information, as some important characteristics of the data may not be captured in the reduced feature set.
- **Simpler Model:** Fewer features led to a simpler model, which makes it easier to interpret the crypto clusters, but it may also oversimplify the underlying structure of the data.
- **Bias:** Depending on which features are selected to represent the data, there could be bias introduced into the clustering process, potentially leading to biased cluster assignments.
- **Impact on Cluster Separation:** The choice of features can significantly impact the separation between clusters. Using fewer features may result in clusters that are less distinct or more overlapping.

In summary, while using fewer features can simplify the clustering process and potentially make the results more robust, it's essential to carefully consider the trade-offs between simplicity and the loss of information when deciding on the number and choice of features for clustering.
