# 19_CrytoClustering
In this repository Python and Unsupervised Machine Learning are used to predict if cryptocurrencies are affected by 24-hour or 7-day price changes. See `Crypto_Clustering.ipynb` for the code.

![Cryto_Image](https://github.com/zhengn95/19_CrytoClustering/blob/main/Images/Intro_Crypto.jpeg)

### Introduction: Unsupervised Machine Learning
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
- Load the `crypto_market_data.csv` into a DataFrame using pandas '.read_csv`.
- Scale and standardized the data using `StandardScaler()` module from scikit-learn.
- Create a DataFrame with the scaled data and set the `"coin_id"` index from the original DataFrame as the index for the new DataFrame.
  
**Step 2:** Finding the Best Value for k using the "Elbow Method"
- Create a list with the number of k values from 1 to 11 & an empty list to store the inertia values.
- Create a for loop to compute the inertia with each possible value of k & a dictionary with the data to plot the elbow curve.
- Plot the `elbow curve` as a line chart with all the inertia values computed with the different values of k to visually identify the optimal value for k.
  
   **Question:** *What is the best value for k?*
   **Answer:** *Based on the elbow curve, `k=3` or `k=4` appear to be good values.*

**Step 3:** Clustering Cryptocurrencies with K-means 
- Initialize the K-means model with the best value for k: `KMeans(n_clusters=4)`
- Fit the K-means model: `.fit(market_data_transformed)`
- Predict the clusters to group the cryptocurrencies: `.predict(market_data_transformed)`
- Create a copy of the original data and add a new column with the predicted clusters.
- Create a scatter plot using hvPlot and set the x axis to `["price_change_percentage_24h"]` and y-axis to `["price_change_percentage_7d]`
  
### Data Anaylsis: K-means with reduced PCA data
**Step 4:** Optimize Clusters with Principal Component Analysis (PCA)  
- Using the original scaled DataFrame, perform a PCA and reduce the features to three principal components.
- Retrieve the explained variance to determine how much information can be attributed to each principal component
  
  **Question:** What is the total explained variance of the three principal components?  
  **Answer:** The total explained variance of the three principal components is approximately 0.8868 or 88.68% of the total variance in the data.
    
  Total Explained Variance=0.37269822+0.32489961+0.18917649  
  Total Explained Variance=0.88677432

- Create a new DataFrame with the PCA data and set the `"coin_id"` index from the original DataFrame as the index for the new DataFrame.  

**Step 5:** Find the Best Value for k using the PCA Data with the "Elbow Method"
- Create a list with the number of k values from 1 to 11 & an empty list to store the inertia values.
- Create a for loop to compute the inertia with each possible value of k & a dictionary with the data to plot the elbow curve.
- Plot the `elbow curve` as a line chart with all the inertia values computed with the different values of k to visually identify the optimal value for k.

  **Question:** What is the best value for k when using the PCA data? Does it differ from the best k value found using the original data?
  **Answer:** Based on this Elbow Curve, it looks like k=4 is the correct one. Compared to the original scaled elbow curve, it us more obvious in the PCA elbow curve that `k=4` is more correct than `k=3` clusters.

![Elbow_Curve](https://github.com/zhengn95/19_CrytoClustering/blob/main/Images/Elbow_Curve.png)

**Step 6:** Cluster Cryptocurrencies with K-means Using the PCA Data
- Initialize the K-means model with the best value for k: `KMeans(n_clusters=4)`
- Fit the K-means model: `.fit(df_market_pca)`
- Predict the clusters to group the cryptocurrencies: `.predict(df_market_pca)`
- Create a copy of the original data and add a new column with the predicted clusters.
- Create a scatter plot using hvPlot and set the x axis to `PC1` and y-axis to `PC2`  
  
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

![Scatter_Plot](https://github.com/zhengn95/19_CrytoClustering/blob/main/Images/Scatter_Plot.png)

