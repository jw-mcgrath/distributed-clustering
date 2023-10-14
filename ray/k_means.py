import ray
import numpy as np

ray.init()

@ray.remote
def assign(x, centroids):
    # map each data point to its nearest centroid
    return np.argmin(np.linalg.norm(x - centroids, axis=1))

@ray.remote
def update(x, assignments, k):
    # update each centroid to be the mean of all data points assigned to it
    return [np.mean(x[assignments == i], axis=0) for i in range(k)]

# Generate some random data
num_points = 10000
dimensions = 2
data = np.random.randn(num_points, dimensions)

# Initialize centroids
k = 10
centroids = np.random.randn(k, dimensions)


# todo write a fixed point stopping condition on # of reassignments
max_iters = 300
for _ in range(max_iters):
    # Asynchronously compute the assignments for each data point
    assignments_ids = [assign.remote(point, centroids) for point in data]
    assignments = ray.get(assignments_ids)

    # Update centroids
    new_centroids = ray.get(update.remote(data, np.array(assignments), k))
    centroids = np.array(new_centroids)

print("Final Centroids:", centroids)

ray.shutdown()