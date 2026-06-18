
import numpy as np
import time
import matplotlib.pyplot as plt

# ================================
# PART A — GRADIENT DESCENT
# ================================

def scalar_function(x):
    return x**2 + 5

def scalar_gradient(x):
    # TODO: return derivative of f(x)
    return 2 * x


def gradient_descent_scalar(x0, alpha, steps=20):
    x = x0
    history = [x]
    for _ in range(steps):
        # TODO: update x using gradient descent
        gradients = scalar_gradient(x)
        x=x-alpha*gradients
        history.append(x)
    return history


def surface_function(v):
    x, y=v
    return x**2 + 3*y**2

def surface_gradient(v):
    # TODO: return gradient as numpy array
    x, y=v
    return np.array([2*x,6*y])


def gradient_descent_vector(v0, alpha, steps=20):
    v = v0.copy()
    history = []
    for _ in range(steps):
        # TODO: update v using gradient descent
        gradients = surface_gradient(v)
        v=v-alpha*gradients
        history.append(v.copy())

    return np.array(history)

def plot_trajectory(history):
    """
    Plots optimization trajectory for 2D gradient descent
    """
    history = np.array(history)
    plt.plot(history[:, 0], history[:, 1], marker='o')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gradient Descent Trajectory")
    plt.show()

# ================================
# PART B — KNN
# ================================

def generate_dataset(n=100, seed=42):
    np.random.seed(seed)
    class0 = np.random.multivariate_normal([0, 0], np.eye(2), n//2)
    class1 = np.random.multivariate_normal([2, 2], np.eye(2), n//2)
    X = np.vstack([class0, class1])
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y



def euclidean_distance_loop(x, y):
    # TODO: compute Euclidean distance using  loop
    sum_sq_diff = 0.0
    for i in range(len(x)):
        diff = x[i]-y[i]
        sum_sq_diff += diff*diff     #adding square of this difference to the sum
    return np.sqrt(sum_sq_diff)



def euclidean_distance_vectorized(x, y):
    # TODO: compute Euclidean distance using NumPy
    return np.sqrt(np.sum((x-y)**2))


def knn_predict(x_test, X_train, y_train, k):
    # TODO:
    # 1. compute distances (no loops)
    # 2. use np.argpartition (no full sort)
    # 3. majority vote
    distance_from_test = np.sqrt(np.sum((X_train - x_test) ** 2, axis=1))
    near_k_indices = np.argpartition(distance_from_test, kth=k - 1)[:k]
    near_k_classes = y_train[near_k_indices]
    predicted_class = int(np.bincount(near_k_classes).argmax())
    return predicted_class


# ================================
# PERFORMANCE COMPARISON UTILITIES
# ================================

def time_function(func, *args, repeats=1000):
    start = time.time()
    for _ in range(repeats):
        func(*args)
    end = time.time()
    return end - start

def compare_distance_performance(dim=1000):
    x = np.random.rand(dim)
    y = np.random.rand(dim)

    loop_time = time_function(euclidean_distance_loop, x, y)
    vec_time = time_function(euclidean_distance_vectorized, x, y)

    print(f"Loop-based distance time: {loop_time:.6f}s")
    print(f"Vectorized distance time: {vec_time:.6f}s")
    print(f"Speedup: {loop_time / vec_time:.2f}x")

def compare_knn_performance(n_samples=5000, n_features=10):
    X, y = generate_dataset(n_samples)
    x_test = X[0]
    k = 5

    start = time.time()
    for i in range(len(X)):
        euclidean_distance_loop(x_test, X[i])
    loop_time = time.time() - start

    start = time.time()
    _ = np.linalg.norm(X - x_test, axis=1)
    vec_time = time.time() - start

    print(f"KNN distance computation (loop): {loop_time:.6f}s")
    print(f"KNN distance computation (vectorized): {vec_time:.6f}s")
    if vec_time > 0:
        print(f"Speedup: {loop_time / vec_time:.2f}x")
    else:
        print("Speedup: Infinite (Vectorized code was instantaneous)")







if __name__ == "__main__":

     #  Task A1: Scalar Gradient Descent, Convergence vs. Divergence 
    print("--- Task A1 ---")

    initial_x = 10
    alphas = [0.1, 1.0, 1.1]

    plt.figure(figsize=(6, 4))

    for alpha in alphas:
        history = gradient_descent_scalar(x0=initial_x, alpha=alpha, steps=20)

        plt.plot(history, marker='o', label=f"alpha = {alpha}")
        print(f"Alpha {alpha}: Final x = {history[-1]:.4f}")

    plt.xlabel("Iteration")

    plt.ylabel("x value")

    plt.title("Gradient Descent for Different Learning Rates")

    plt.legend()

    plt.grid(True)

    plt.yscale("symlog")

    plt.savefig("Task_A1.png")

    plt.show()



   # Task A2: 2D Gradient Descent (10 steps)
print("\n--- Task A2 ---")

start_point = np.array([5.0, 5.0])
path = gradient_descent_vector(v0=start_point, alpha=0.1, steps=10)

path = np.vstack([start_point, path])

print("Iteration    x         y")
for i in range(len(path)):              # this will print 0..10
    print(f"{i:<9} {path[i, 0]:<8.3f} {path[i, 1]:<8.4f}")

# (keep your plot if you want)
x_vals = np.linspace(-6, 6, 100)
y_vals = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x_vals, y_vals)

Z = X**2 + 3 * Y**2

plt.figure(figsize=(6, 6))
plt.contour(X, Y, Z, levels=20)
plt.plot(path[:, 0], path[:, 1], marker='o')
plt.xlabel("x")
plt.ylabel("y")

plt.title("Gradient Descent on f(x,y) = x^2 + 3y^2")
plt.grid(True)
plt.gca().set_aspect('equal')
plt.show()



# Task A3: Divergence example (vertical exploding values) 
print("\n--- Task A3 ---")
print("Used 1.1 as learning rate for divergence as in first part it cause divergence")
print("Exploding Values")

exploding_steps = gradient_descent_scalar(x0=initial_x, alpha=1.1, steps=5)

if len(exploding_steps) == 5:
    exploding_steps = [initial_x] + list(exploding_steps)

for t, x_val in enumerate(exploding_steps):
    print(f"t = {t}    x{t} = {x_val:.4f}")





#-----------------------------------------------------------------------
# PART B: K-NEAREST NEIGHBORS (KNN)

all_features, all_labels = generate_dataset(n=100, seed=42)

rand_no_gen = np.random.default_rng(42)
shuffled_indices = rand_no_gen.permutation(len(all_features))
index_split = int(0.7 * len(all_features))

train_indices = shuffled_indices[:index_split]
test_indices = shuffled_indices[index_split:]

training_features = all_features[train_indices]
training_labels = all_labels[train_indices]
test_features = all_features[test_indices]
test_labels = all_labels[test_indices]



print("\n\nPART B: K-Nearest Neighbors (KNN)")
print()

print("Task B1: Euclidean Distance")
first_point = all_features[0]
second_point = all_features[1]

loop_distance = euclidean_distance_loop(first_point, second_point)
vector_distance = euclidean_distance_vectorized(first_point, second_point)

print("B1.1-> Loop distance:", loop_distance)
print("B1.2-> Vectorized distance:", vector_distance)
print("B1.3-> As you can see above, Same result")

dim = 1000
rand_point_a = np.random.rand(dim)
rand_point_b = np.random.rand(dim)

loop_time_seconds = time_function(euclidean_distance_loop, rand_point_a, rand_point_b, repeats=2500)
vec_time_seconds = time_function(euclidean_distance_vectorized, rand_point_a, rand_point_b, repeats=2500)

print("B1.4-> Loop time:", f"{loop_time_seconds:.6f}s")
print("B1.4-> Vectorized time:", f"{vec_time_seconds:.6f}s")
print("B1.4-> Speedup:", f"{(loop_time_seconds / vec_time_seconds):.2f}x")
print("B1.4-> Clarity: Loop is step-by-step; vectorized is shorter and usually faster.")
print()

print("Task B2: Distance Computation with Broadcasting")
single_test_point = test_features[0]

distances_to_train = np.linalg.norm(training_features - single_test_point, axis=1)

print("B2.1-> Test point:", single_test_point)
print("B2.1-> Distances shape:", distances_to_train.shape)
print("B2.1-> First 10 distances:", distances_to_train[:10])
print()

    #B2.2+ B2.3
    # training_features shape: (n_train, 2) -> n_train points, each point has 2 values (x, y)
    # single_test_point shape: (2,)        -> one point with 2 values [x, y]
    # training_features                    ->single_test_point broadcasts the (2,) point across all rows (no Python loop)
    # this produces (n_train, 2) diff vectors: diff[i] = training_features[i] - single_test_point
    # np.linalg.norm(..., axis=1) converts each diff row into one distance, giving (n_train,) distances

print("Task B3: Neighbor Selection (No Full Sorting)")
k_value1 = 10
nearest_indices = np.argpartition(distances_to_train, kth=k_value1 - 1)[:k_value1]

print("B3.1-> k:", k_value1)
print("B3.1-> Nearest indices (argpartition, unsorted):", nearest_indices)
print("B3.2-> Why not argsort: because it sorts every distance from smallest to largest,\n even though you will only use the first k after sorting. That extra work becomes wasteful when n is large")
print("B3.3-> Complexity: argsort= O(n log n), selection/partition= O(n) for top-k.")
print()

print("Task B4: KNN Prediction")
k_value2 = 10
predicted_class = knn_predict(single_test_point, training_features, training_labels, k_value2)

neighbor_indices_for_vote = np.argpartition(distances_to_train, kth=k_value2 - 1)[:k_value2]
neighbor_labels_for_vote = training_labels[neighbor_indices_for_vote]

# Get the indices of the k nearest training points (smallest distances) without sorting the whole array.
# kth=k_value2-1 sets the “cutoff” so the first k positions contain the k smallest distances (order inside k is not guaranteed).
# [:k_value2] just takes that first k group to use as neighbors for majority voting.

print("B4.1-> Neighbor labels:")
for index in range(len(neighbor_labels_for_vote)):
    print("Neighbor", index + 1, "class =", int(neighbor_labels_for_vote[index]))

print("B4.2-> Predicted class (majority vote):", int(predicted_class))

class_counts = np.bincount(training_labels, minlength=2)
print("Train class counts: class 0 =", class_counts[0], ", class 1 =", class_counts[1])

print()
    

print("Task B5: Effect of k")

def accuracy_for_k(num_neighbors):
    correct_predictions = 0
    for i in range(len(test_features)):
        predicted_label = knn_predict(test_features[i],training_features,training_labels,num_neighbors)
        if predicted_label == test_labels[i]:
            correct_predictions += 1
    return correct_predictions / len(test_features)

k_value_one = 1
k_value_five = 5
k_values_all_training = len(training_features)

k1_accuracy = accuracy_for_k(k_value_one)
k5_accuracy = accuracy_for_k(k_value_five)
kn_accuracy = accuracy_for_k(k_values_all_training)

print(f"B5.1-> Accuracy k=1: {k1_accuracy:.3f}")
print(f"B5.1-> Accuracy k=5: {k5_accuracy:.3f}")
print(f"B5.1-> Accuracy k=n_train({k_values_all_training}): {kn_accuracy:.3f} \n")


print("B5.2-> Overfitting happens at very small k (like k=1).")
print(f"B5.2-> Underfitting happens at very large k (like k=n_train={k_values_all_training}).\n")

majority_class = int(np.bincount(training_labels).argmax())
majority_count = int(np.bincount(training_labels)[majority_class])

print("B5.3-> Why: Small k uses only a few closest points, so it can follow noise/outliers and become unstable.")
print("B5.3-> Medium k (like 5) averages nearby votes, so it becomes more stable and often improves accuracy.")
print(f"B5.3-> When k=n_train, every test point votes using ALL training points,")
print(f"             so the model predicts the same 'global majority class' ({majority_class}, count={majority_count}) for almost every test point.")
print("             On roughly balanced data, that behaves close to guessing, so accuracy can drop toward almost 50%.")
