import numpy as np
import random
import matplotlib.pyplot as plt
import math
import statistics

def separate_data(seq):
    '''
    Function to separate 2D input list into two 1D lists (x & y)
    '''
    temp_x = []
    temp_y = []
    for i in range(len(seq)):
        temp_x.append(seq[i][0])
        temp_y.append(seq[i][1])
    return temp_x, temp_y

def calculate_rnk(data_points, mu1, mu2, mu3):
    '''
    This Function estimates "rnk" of each point and do the clustering based on distance between each point and Mu1, Mu2 & Mu3.
    The Function returns the 3 lists of computed cluster points.
    '''
    first_cluster_points = []
    second_cluster_points = []
    third_cluster_points = []
    for m in range(len(data_points)):      # Calculate the distance between each point & the 3 centers
        distance_to_first_center = (x[m] - mu1[0])**2 + (y[m] - mu1[1])**2
        distance_to_second_center = (x[m] - mu2[0])**2 + (y[m] - mu2[1])**2
        distance_to_third_center = (x[m] - mu3[0])**2 + (y[m] - mu3[1])**2
        total_distances = [distance_to_first_center, distance_to_second_center, distance_to_third_center]
        if total_distances.index(min(total_distances)) == 0:
            first_cluster_points.append(data_points[m])
        elif total_distances.index(min(total_distances)) == 1:
            second_cluster_points.append(data_points[m])
        else:
            third_cluster_points.append(data_points[m])

    return [first_cluster_points, second_cluster_points, third_cluster_points]

def compute_mu(cluster1, cluster2, cluster3):
    '''
    This Function computes "Mu-k" based on the points of each cluster (according to K-Mean Algorithm).
    '''
    # Compute First Center again based on cluster points:
    if len(cluster1) == 0:
        mean1 = 0
    else:
        x_acc = 0
        y_acc = 0
        for a in range(len(cluster1)):
            x_acc += cluster1[a][0]
            y_acc += cluster1[a][1]
        mean1 = [(1 / len(cluster1)) * x_acc, (1 / len(cluster1)) * y_acc]
    # Compute Second Center again based on cluster points:
    if len(cluster2) == 0:
        mean2 = 0
    else:
        x_acc1 = 0
        y_acc1 = 0
        for s in range(len(cluster2)):
            x_acc1 += cluster2[s][0]
            y_acc1 += cluster2[s][1]
        mean2 = [(1 / len(cluster2)) * x_acc1, (1 / len(cluster2)) * y_acc1]
    # Compute Third Center again based on cluster points:
    if len(cluster3) == 0:
        mean3 = 0
    else:
        x_acc2 = 0
        y_acc2 = 0
        for d in range(len(cluster3)):
            x_acc2 += cluster3[d][0]
            y_acc2 += cluster3[d][1]
        mean3 = [(1 / len(cluster3)) * x_acc2, (1 / len(cluster3)) * y_acc2]

    return mean1, mean2, mean3

def calculate_average_distance(final_center1, final_center2, final_center3, final_cluster1, final_cluster2, final_cluster3):
    '''
    This Function estimates average distance between the points and the centers of their corresponding clusters.
    '''
    # Start with cluster 1:
    temp_cluster1 = 0
    for q in range(len(final_cluster1)):
        temp_cluster1 += (np.array(final_cluster1[q]) - np.array(final_center1))**2
    total_cluster1 = (1 / len(final_cluster1)) * temp_cluster1
    # Cluster 2 calculation:
    temp_cluster2 = 0
    for w in range(len(final_cluster2)):
        temp_cluster2 += (np.array(final_cluster2[w]) - np.array(final_center2))**2
    total_cluster2 = (1 / len(final_cluster2)) * temp_cluster2
    # Cluster 3 calculation:
    temp_cluster3 = 0
    for e in range(len(final_cluster3)):
        temp_cluster3 += (np.array(final_cluster3[e]) - np.array(final_center3))**2
    total_cluster3 = (1 / len(final_cluster3)) * temp_cluster3

    total_average_distance = (1 / 3) * (total_cluster1 + total_cluster2 + total_cluster3)
    return total_average_distance

###################################################### Load Data & Start the 100 Iterations ###########################################################

data = np.loadtxt(r'C:\Users\chtv2985\Desktop\Assig 4\Data.txt')   # load data
data_list = data.tolist()
x, y = separate_data(data_list)     # separate data input points into x & y lists

# Apply K-Means algorithm for 100 times:
total_clustered_points = []
total_calculated_means = []
all_average_distances = []
iteration_number = 1
for n in range(100):
    # Select the First Center randomly:
    random_x = random.choice(list(enumerate(x)))  # randomly select data point from x and keep it's index, random_x[0] is the index and
                                                    #  random_x[1] is the element itself
    first_center = [random_x[1], y[random_x[0]]]  # select data point from y with the corresponding index
    # Compute Second Center according to max distance from First Center:
    second_center_list = []
    for j in range(len(data_list)):
        temp = np.linalg.norm(np.array(first_center) - np.array(data_list[j]))  # Calculate Euclidean distance between each point & 1st center
        second_center_list.append(temp)
    second_center = data_list[second_center_list.index(max(second_center_list))]  # Select Second Center with the max distance from 1st center
    # Compute Third Center according to max distance from Second Center (after excluding First Center from list):
    third_center_list = []
    data_list_temp = []
    for l in range(len(data_list)):  # Copy data points to another temp list
        data_list_temp.append(data_list[l])
    del data_list_temp[random_x[0]]  # Remove first center from data list before we compute the third center
    for k in range(len(data_list_temp)):
        temp2 = np.linalg.norm(np.array(second_center) - np.array(data_list_temp[k]))
        third_center_list.append(temp2)
    third_center = data_list_temp[third_center_list.index(max(third_center_list))]

    # Cluster data points based on computed centers:
    first_clustering_output = calculate_rnk(data_list, first_center, second_center, third_center)
    # Re-calculate the three centers based on clusters average (mean):
    first_center_modified, second_center_modified, third_center_modified = compute_mu(first_clustering_output[0], first_clustering_output[1],
                                                                                      first_clustering_output[2])
    # Now need to keep calculating centers till K-means algorithm converges:
    i = 1
    while first_center != first_center_modified or second_center != second_center_modified or third_center != third_center_modified:
        first_center = first_center_modified
        second_center = second_center_modified
        third_center = third_center_modified
        second_clustering_output = calculate_rnk(data_list, first_center_modified, second_center_modified, third_center_modified)
        first_center_modified, second_center_modified, third_center_modified = compute_mu(second_clustering_output[0], second_clustering_output[1],
                                                                                          second_clustering_output[2])
        print("(for Mu's Convergence) Internal Sub-iteration Number: " + str(i))
        i += 1

    print("Finish, K-Means Algorithm Converged!")

    # After algorithm convergence, we need to calculate the average of distances between points and their corresponding centers in each cluster
    calculated_distance = calculate_average_distance(first_center_modified, second_center_modified, third_center_modified,
                                                     second_clustering_output[0], second_clustering_output[1], second_clustering_output[2])

    print("Final Calculated Centers: ")
    print(first_center_modified, second_center_modified, third_center_modified)

    total_calculated_means.append([first_center_modified, second_center_modified, third_center_modified])
    total_clustered_points.append([second_clustering_output[0], second_clustering_output[1], second_clustering_output[2]])
    all_average_distances.append(calculated_distance)
    print("Iteration Number: " + str(iteration_number))
    iteration_number += 1
    print("=============================================================================================================")

######################################################## Check & Plot Obtained Clustered Data #######################################################

# Check the obtained centers in the 100 iterations:
# We will compare using the mean of the three centers of each operation (with round to the 3rd decimal digit)
check_difference = 0
for u in range(len(total_calculated_means)):
    if round(statistics.mean(total_calculated_means[0][0])+statistics.mean(total_calculated_means[0][1])+statistics.mean(total_calculated_means[0][2]), 3) != round(statistics.mean(total_calculated_means[u][0])+statistics.mean(total_calculated_means[u][1])+statistics.mean(total_calculated_means[u][2]), 3):
        check_difference += 1
        print("Different Centers Found at Index (iteration) Number: " + str(u))

if check_difference == 0:
    print("All Centers in the 100 Iterations have the same value!")

# As we obtained the same 3 centers in all of the 100 iterations
# Therefore, we shall use any of them to finally cluster data. Let's choose the firstly calculated centers.

# Separating final clustered data into x & y lists:
cluster1_points_x , cluster1_points_y = separate_data(total_clustered_points[0][0])
cluster2_points_x , cluster2_points_y = separate_data(total_clustered_points[0][1])
cluster3_points_x , cluster3_points_y = separate_data(total_clustered_points[0][2])

# Plot the 3 obtained clusters:
plt.plot(cluster1_points_x, cluster1_points_y, 'yX', label='Cluster 1')
plt.plot(cluster2_points_x, cluster2_points_y, 'rX', label='Cluster 2')
plt.plot(cluster3_points_x, cluster3_points_y, 'bX', label='Cluster 3')
plt.title('Clustered Data Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best', framealpha=1, borderpad=1)
plt.grid()
plt.show()