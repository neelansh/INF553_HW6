import random
import os
import sys
import json
import numpy as np
from sklearn.cluster import KMeans
import math
from collections import Counter, defaultdict
import itertools

seed = 1

def data_generator():
    data = np.loadtxt(file_path, delimiter=',')
    data = data.tolist()
    random.shuffle(data)
    data_batches = []
    for i in range(0, len(data), math.floor(0.2*len(data))):
        d = np.array(data[i:i+math.floor(0.2*len(data))])
        data_batches.append(d)
        
    return data_batches

class stats:
    def __init__(self, points):
        # create the stats from points
        self.point_indices = points[:, 0].tolist()
        self.actual_clusters = points[:, 1].tolist()
        self.n = len(points)
        self.sum = np.sum(points[:, 2:], axis=0)
        self.sumsq = np.sum(np.power(points[:, 2:], 2), axis=0)\
        
    def update_stats(self, point):
        self.point_indices.append(point[0])
        self.actual_clusters.append(point[1])
        self.n += 1
        self.sum += point[2:]
        self.sumsq += np.power(point[2:], 2)
        
        return 
    
    def merge_with(self, cluster):
        self.point_indices += cluster.point_indices
        self.actual_clusters += cluster.actual_clusters
        self.n += cluster.n
        self.sum += cluster.sum
        self.sumsq += cluster.sumsq
        
    def calculate_variance(self):
        return np.power((self.sumsq/self.n) - np.power(self.sum/self.n, 2), 1/2)
    
    def calculate_centroid(self):
        return self.sum / self.n
    
def mahalanobis_distance(cluster, point):
    return np.power(np.sum(np.power((point - cluster.calculate_centroid()) / cluster.calculate_variance(), 2)), 1/2)



def bfr(data, k, d, output_file_path):
    global seed
    rs = defaultdict(list)
    ds = defaultdict(list)
    cs = defaultdict(list)
    
    output_file = open(output_file_path, "wt")
    output_file.write("The intermediate results:\n")
    for i, batch in enumerate(data):
        if(i == 0):
            #  Run K-Means (e.g., from sklearn) with a large K (e.g., 5 times of the number of the input clusters) on the data in memory using the Euclidean distance as the similarity measurement.
            model = KMeans(n_clusters=k*5, random_state=seed)
            model = model.fit(batch[:, 2:])
            
            # In the K-Means result from Step 2, move all the clusters that contain only one point to RS
            index = defaultdict(list)
            for pos, centroid_id in enumerate(model.labels_):
                index[centroid_id].append(pos)
                
            rest_of_data = []
            
            for centroid_id, positions in index.items():
                if(len(positions) == 1):
                    rs[centroid_id] = np.take(batch, positions, axis=0)
                if(len(positions) > 1):
                    rest_of_data.append(np.take(batch, positions, axis=0))
            
            rest_of_data = np.concatenate(rest_of_data, axis=0)
            # Run K-Means again to cluster the rest of the data points with K = the number of input clusters.
            model = KMeans(n_clusters=k, random_state=seed)
            model = model.fit(rest_of_data[:, 2:])

            # Use the K-Means result from Step 4 to generate the DS clusters (i.e., discard their points and generate statistics).
            index = defaultdict(list)
            for pos, centroid_id in enumerate(model.labels_):
                index[centroid_id].append(pos)
            
            for centroid_id, positions in index.items():
                points = np.take(rest_of_data, positions, axis=0)
                ds[centroid_id] = stats(points)
                
            points = [x for x in rs.values()]
            points = np.concatenate(points, axis=0)
            
            model = KMeans(n_clusters=min(5*k, len(points)), random_state=seed)
            model = model.fit(points[:, 2:])
            
            index = defaultdict(list)
            for pos, centroid_id in enumerate(model.labels_):
                index[centroid_id].append(pos)
            
            rs = defaultdict(list)
            for centroid_id, positions in index.items():
                p = np.take(points, positions, axis=0)
                if(len(positions) == 1):
                    rs[centroid_id] = p
                if(len(positions) > 1):
                    cs[centroid_id] = stats(p)
            
        else:
            for point in batch:
                cluster_id, min_dist = min(list(map(lambda x: (x[0], mahalanobis_distance(x[1], point[2:])), ds.items())), key=lambda x: x[1])
                
                if(min_dist < 2*math.pow(d, 1/2)):
                    ds[cluster_id].update_stats(point)
                else:
                    if(len(cs) != 0):
                        cluster_id, min_dist = min(list(map(lambda x: (x[0], mahalanobis_distance(x[1], point[2:])), cs.items())), key=lambda x: x[1])
                        if(min_dist < 2*math.pow(d, 1/2)):
                            cs[cluster_id].update_stats(point)
                        else:
                            if(len(rs) == 0):
                                rs[0] = np.expand_dims(point, axis=0)
                            else:
                                rs[max(rs.keys())+1] = np.expand_dims(point, axis=0)
                    else:
                        if(len(rs) == 0):
                            rs[0] = np.expand_dims(point, axis=0)
                        else:
                            rs[max(rs.keys())+1] = np.expand_dims(point, axis=0)
                        
            points = [x for x in rs.values()]
            points = np.concatenate(points, axis=0)
            
            model = KMeans(n_clusters=min(5*k, len(points)), random_state=seed)
            model = model.fit(points[:, 2:])
            
            index = defaultdict(list)
            for pos, centroid_id in enumerate(model.labels_):
                index[centroid_id].append(pos)
            
            rs = defaultdict(list)
            for centroid_id, positions in index.items():
                p = np.take(points, positions, axis=0)
                if(len(positions) == 1):
                    rs[centroid_id] = p
                if(len(positions) > 1):
                    if(len(cs) == 0):
                        cs[0] = stats(p)
                    else:
                        cs[max(cs.keys())+1] = stats(p)
                    
            # merge cs clusters if distance < 2 root d
            to_be_merged = []
            for c1, c2 in itertools.combinations(cs.keys(), 2):
                dist = mahalanobis_distance(cs[c1], cs[c2].calculate_centroid())
                if(dist < 2*math.pow(dist, 1/2)):
                    to_be_merged.append((c1, c2))
                    
            for (c1, c2) in to_be_merged:
                if(c1 in cs and c2 in cs):
                    cs[c1].merge_with(cs[c2])
                    del cs[c2]
        # after each round output
        number_of_ds_points = sum([x.n for x in ds.values()])
        number_of_clusters_cs = len(cs)
        number_of_cs_points = sum([x.n for x in cs.values()])
        number_of_rs_points = sum([len(x) for x in rs.values()])
        if(i != len(data)-1):
            output_file.write("Round {}: {},{},{},{}\n".format(i+1, number_of_ds_points, number_of_clusters_cs, number_of_cs_points, number_of_rs_points))
        
        
    # after last round
    
    # merge cs with ds with distance less than 2 root d
    merged_cs = []
    for k, c in cs.items():
        point = c.calculate_centroid()
        cluster_id, min_dist = min(list(map(lambda x: (x[0], mahalanobis_distance(x[1], point)), ds.items())), key=lambda x: x[1])
        if(min_dist < 2*math.pow(d, 1/2)):
            ds[cluster_id].merge_with(c)
            merged_cs.append(k)

    for k in merged_cs:
        del cs[k]
        
    number_of_ds_points = sum([x.n for x in ds.values()])
    number_of_clusters_cs = len(cs)
    number_of_cs_points = sum([x.n for x in cs.values()])
    number_of_rs_points = sum([len(x) for x in rs.values()])
    output_file.write("Round {}: {},{},{},{}\n".format(len(data), number_of_ds_points, number_of_clusters_cs, number_of_cs_points, number_of_rs_points))
        
    gt = []
    pred = []
    original_index = []
    cluster_id = 0
    for i, x in ds.items():
        gt += x.actual_clusters
        pred += [cluster_id]*x.n
        original_index += x.point_indices
        cluster_id += 1


    for i, x in cs.items():
        gt += x.actual_clusters
        pred += [cluster_id]*x.n
        original_index += x.point_indices
        cluster_id += 1

    for i, x in rs.items():
        gt += x[:, 1].tolist()
        original_index += x[:, 0].tolist()
        pred += [-1]*len(x)
    
    gt = [int(x) for x in gt]
    pred = [int(x) for x in pred]
    original_index = [int(x) for x in original_index]
    final_output = sorted([(x,y) for x,y in zip(original_index, pred)])
    
    output_file.write("\n")
    output_file.write("The clustering results:\n")
    for x, y in final_output:
        output_file.write("{},{}\n".format(x, y))
    
    #from sklearn.metrics.cluster import v_measure_score
    #print(v_measure_score(gt, pred))
    
    output_file.close()
    return ds, cs, rs 
            
if __name__ == "__main__":
    file_path = sys.argv[1].strip()
    k = int(sys.argv[2].strip())
    output_path = sys.argv[3].strip()
    random.seed(seed)
    data_batches = data_generator()
    d = data_batches[0].shape[-1]-2
    ds, cs, rs = bfr(data_batches, k, d, output_path)