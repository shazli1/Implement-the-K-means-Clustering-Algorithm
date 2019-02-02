# Implement-the-K-means-Clustering-Algorithm
Required to implement the K-means clustering algorithm. The algorithm is applied to data provided in the file “Data.txt” with K = 3. Each row in the file corresponds to one data point. One important aspect of K-means that changes the results significantly is the initialization.

# Initialization Strategy
A good strategy for initializing cluster centers is as follows:  
1- Pick one of the dataset points randomly as the center of the first cluster.  
2- For the next cluster, find the point with maximum distance to the center of the previous cluster.  
3- Choose this point as the center of the next cluster.  
4- Repeat steps 2 and 3 until you initialize the centers of all clusters.  

# Run Criteria
The K-means algorithm with the initialization method above is required to be run 100 times.  
The final output of the K-means clustering is the result that gives the minimum average distance between the points and the centers of their corresponding clusters.
