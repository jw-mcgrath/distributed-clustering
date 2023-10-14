import ray
import numpy as np
from sklearn.cluster import KMeans
ray.init()

@ray.remote
def assign(x, centroids):
    """
    x: [batch, dim] tensor
    centroids: [k, dim] tensor

    output is a [batch] tensor of assignments in [0, k)
    """
    # x: [batch, dim] 
    return np.argmin(np.linalg.norm(x[:, np.newaxis] - centroids, axis=2), axis=1)

@ray.remote
def get_sum(x, assignments, k):
    """
    x: [batch, dim] tensor
    assignments: [batch] tensor
    k: int
    returns: [k, dim] tensor
    """
    return np.stack([np.sum(x[assignments == i], axis=0) for i in range(k)])

@ray.remote
def get_counts(assignments, k):
    """
    assignments: [batch] tensor
    k: int
    returns: [k] tensor
    """
    return np.array([np.sum(assignments == i) for i in range(k)])

# Generate some random data
num_points = 100_000
dimensions = 28
data = np.random.randn(num_points, dimensions)

# Initialize centroids
k = 10
initial_centroids = np.random.randn(k, dimensions)
centroids = initial_centroids.copy()


# todo write a fixed point stopping condition on # of reassignments
max_iters = 100
for i in range(max_iters):
    print(f"Iteration {i}")
    chunks = np.split(data, 20, axis=0)
    # Asynchronously compute the assignments for each data point
    assignments_ids = [assign.remote(chunk, centroids) for chunk in chunks]
    assignments = ray.get(assignments_ids)
   
    
    # Update centroids
    new_centroids_parts = [ray.get(get_sum.remote(chunk, assignment, k)) for chunk, assignment in zip(chunks, assignments)]
    new_centroid_counts = ray.get([get_counts.remote(assignment, k) for assignment in assignments])
    new_centroids = np.sum(new_centroids_parts, axis=0) / np.sum(new_centroid_counts, axis=0)[:, np.newaxis]
    centroids = np.array(new_centroids)

print("testing against sklearn")
kmeans = KMeans(n_clusters=k, init=initial_centroids).fit(data)
print("Sklearn Centroids:", kmeans.cluster_centers_)
print("ray Centroids:", centroids)
print("distance", np.linalg.norm(kmeans.cluster_centers_ - centroids))

ray.shutdown()