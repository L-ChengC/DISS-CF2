# DISS-CF
Direct Item Session Similarity enhanced Collaborative Filtering

Anyone who want to use this plugin to import GCN-based Recommender System can do as the following:
1. download the file and put it to your project directory
2. Call the method and pass the parameters and the method would return an (#users+#items, #users+#items) adjcency matrix
3. Using the new adjacency matrix in the following programming

How LightGCN applies DISS-CF method is also provided and the code of LightGCN is from https://github.com/kuandeng/LightGCN
