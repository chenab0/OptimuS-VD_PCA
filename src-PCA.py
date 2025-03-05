import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 
import os

os.makedirs('anim', exist_ok=True)



walking_1 = np.load('/Users/chenab/Downloads/hw2data 2/train/walking_1.npy')
walking_2 = np.load('/Users/chenab/Downloads/hw2data 2/train/walking_2.npy')
walking_3 = np.load('/Users/chenab/Downloads/hw2data 2/train/walking_3.npy')
walking_4 = np.load('/Users/chenab/Downloads/hw2data 2/train/walking_4.npy')
walking_5 = np.load('/Users/chenab/Downloads/hw2data 2/train/walking_5.npy')

jumping_1 = np.load('/Users/chenab/Downloads/hw2data 2/train/jumping_1.npy')
jumping_2 = np.load('/Users/chenab/Downloads/hw2data 2/train/jumping_2.npy')
jumping_3 = np.load('/Users/chenab/Downloads/hw2data 2/train/jumping_3.npy')
jumping_4 = np.load('/Users/chenab/Downloads/hw2data 2/train/jumping_4.npy')
jumping_5 = np.load('/Users/chenab/Downloads/hw2data 2/train/jumping_5.npy')
                  
running_1 = np.load('/Users/chenab/Downloads/hw2data 2/train/running_1.npy')
running_2 = np.load('/Users/chenab/Downloads/hw2data 2/train/running_2.npy')
running_3 = np.load('/Users/chenab/Downloads/hw2data 2/train/running_3.npy')
running_4 = np.load('/Users/chenab/Downloads/hw2data 2/train/running_4.npy')
running_5 = np.load('/Users/chenab/Downloads/hw2data 2/train/running_5.npy')

samples = [walking_1, walking_2, walking_3, walking_4, walking_5,
           jumping_1, jumping_2, jumping_3, jumping_4, jumping_5,
           running_1, running_2, running_3, running_4, running_5]

print(np.shape(walking_1))

X_train = np.hstack(samples)
print(np.shape(X_train))

X_train_T = X_train.T
pca = PCA()
pca.fit(X_train_T)

singular_values = pca.singular_values_

cumulative_energy = np.cumsum(singular_values**2)
total_energy = np.sum(singular_values**2)

thresholds = [0.70, 0.80, 0.90, 0.95]

for thresh in thresholds:
    num_modes = np.argmax(cumulative_energy/total_energy >= thresh) + 1
    print(f"PCA modes needed to approximate up to {int(thresh * 100)}% of the energy: {num_modes}")

plt.figure(figsize=(10,6))
modes = np.arange(1, len(cumulative_energy) + 1)
plt.plot(modes, cumulative_energy/total_energy * 100, 'k-', marker='o', label='Cumulative Energy')


colors = ['b', 'r', 'g', 'm'] 
for thresh, color in zip(thresholds, colors):
    plt.axhline(y=thresh*100, color=color, linestyle='--', label=f'{int(thresh*100)}% threshold')

plt.xlabel("Number of modes")
plt.ylabel("Energy percentage")
plt.title("Percentage Energy Captured vs Number of Modes")
plt.grid(True)
plt.legend()  
min_energy = (cumulative_energy/total_energy * 100).min() 
max_energy = (cumulative_energy/total_energy * 100).max()
plt.ylim(min_energy - 5, max_energy + 5)
plt.show()



pca_2d = PCA(2)
X_train_2d = pca_2d.fit_transform(X_train_T)  # shape: (1500, 2)
print(np.shape(X_train_2d))

pca_3d = PCA(3)
X_train_3d = pca_3d.fit_transform(X_train_T)  # shape: (1500, 3)
print(np.shape(X_train_3d))

num_timesteps = 100
num_samples = len(samples)  #

colors = ['r', 'r', 'r', 'r', 'r', 'g', 'g', 'g', 'g', 'g', 'b', 'b', 'b', 'b', 'b']

plt.figure(figsize=(8, 6))
added_labels = set()
for i in range(num_samples):
    start = i * num_timesteps
    end = start + num_timesteps
    traj_2d = X_train_2d[start:end, :] 
    labels = None
    if(i == 0):
        label = 'Walking'
        walking_label_added = True
    elif(i == 5):
        label = 'Jumping'
        jumping_label_added = True
    elif(i == 10):
        label = 'Running'
        running_label_added = True
    if label not in added_labels:
        plt.plot(traj_2d[:, 0], traj_2d[:, 1], color=colors[i], label=label)
        added_labels.add(label)
    else:
        plt.plot(traj_2d[:, 0], traj_2d[:, 1], color=colors[i])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Trajectories of 2D Projection')
plt.grid(True)
plt.legend()
plt.show()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
added_labels = set()
for i in range(num_samples):
    start = i * num_timesteps
    end = start + num_timesteps
    traj_3d = X_train_3d[start:end, :]
    labels = None
    if(i == 0):
        label = 'Walking'
        walking_label_added = True
    elif(i == 5):
        label = 'Jumping'
        jumping_label_added = True
    elif(i == 10):
        label = 'Running'
    if label not in added_labels:
        ax.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], color= colors[i], label = label)
        added_labels.add(label)
    else:
        ax.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], color= colors[i])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Trajectories of 3D Projection')
plt.legend()
plt.show()


ground_truth_vector = np.hstack([np.zeros(num_timesteps*5), np.ones(num_timesteps*5), np.full(5 * num_timesteps, 2)])
print(np.shape(ground_truth_vector))

walking_index = np.arange(0, 5 * num_timesteps)
jumping_index = np.arange(5 * num_timesteps, 10 * num_timesteps)
running_index = np.arange(10 * num_timesteps, 15 * num_timesteps)


def getSampleCentroids(k, X_train):
    pca_k = PCA(k)
    X_train_kd = pca_k.fit_transform(X_train_T)
    
    centroid_walking = np.mean(X_train_kd[walking_index, :], axis=0)
    centroid_jumping = np.mean(X_train_kd[jumping_index, :], axis=0)
    centroid_running = np.mean(X_train_kd[running_index, :], axis=0)
    
    return [centroid_walking, centroid_jumping, centroid_running]
    
centroids_2d = getSampleCentroids(2, X_train)
centroid_walking = centroids_2d[0]
centroid_jumping = centroids_2d[1]
centroid_running = centroids_2d[2]

print("\nCentroids in 2D PCA space:")
print("Walking centroid (label 0):", centroids_2d[0])
print("Jumping centroid (label 1):", centroids_2d[1])
print("Running centroid (label 2):", centroids_2d[2])

plt.figure(figsize=(8, 6))
for i in range(num_samples):
    start = i * num_timesteps
    end = start + num_timesteps
    traj_2d = X_train_2d[start:end, :] 
    plt.plot(traj_2d[:, 0], traj_2d[:, 1], color= colors[i], alpha = 0.2)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Centroid Visualization in 2D')
plt.grid(True)

plt.scatter(centroid_walking[0], centroid_walking[1], color='r', marker='o', s=150, label='Centroid Walking')
plt.scatter(centroid_jumping[0], centroid_jumping[1], color='g', marker='o', s=150, label='Centroid Jumping')
plt.scatter(centroid_running[0], centroid_running[1], color='b', marker='o', s=150, label='Centroid Running')
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score


k_values = np.arange(50)

print("\nClassification accuracy for various values of k:")

accuracy = [] 

for k in k_values:
    pca_k = PCA(n_components=k)
    X_train_kd = pca_k.fit_transform(X_train_T)  
    

    centroids_kd = getSampleCentroids(k, X_train)
    
    trained_labels = np.empty(X_train_kd.shape[0], dtype=int)
    for i in range(X_train_kd.shape[0]):
        point = X_train_kd[i, :]
        d_walk = np.linalg.norm(point - centroids_kd[0])
        d_jump = np.linalg.norm(point - centroids_kd[1])
        d_run  = np.linalg.norm(point - centroids_kd[2])
        distance_array = [d_walk, d_jump, d_run]
        trained_labels[i] = np.argmin(distance_array)
    
    acc = accuracy_score(trained_labels.ravel(), ground_truth_vector)
    accuracy.append(acc)
    print(f"Accuracy for k = {k}: {acc * 100:.2f}%")

plt.figure(figsize=(8, 6))
plt.plot(k_values, [acc * 100 for acc in accuracy])
plt.grid(True)
plt.title("Accuracy vs k modes")
plt.xlabel("k")
plt.ylabel("accuracy")

walking_test = np.load('/Users/chenab/Downloads/hw2data 2/test/walking_1t.npy')
jumping_test = np.load('/Users/chenab/Downloads/hw2data 2/test/jumping_1t.npy')
running_test = np.load('/Users/chenab/Downloads/hw2data 2/test/running_1t.npy')


test_samples = [walking_test, jumping_test, running_test]
ground_truth_vector_test = np.hstack([np.zeros(num_timesteps), np.ones(num_timesteps), np.full(num_timesteps, 2)])


X_test = np.hstack(test_samples)
X_test_T = X_test.T

k_values = np.arange(100)
accuracy = []
for k in k_values:
    pca_k = PCA(k)
    X_train_kd = pca_k.fit_transform(X_train_T)  
    X_test_kd = pca_k.transform(X_test_T)
    
    centroids_kd = getSampleCentroids(k, X_train)
    
    test_labels = np.empty(X_test_kd.shape[0], dtype=int)
    for i in range(X_test_kd.shape[0]):
        point = X_test_kd[i, :]
        d_walk = np.linalg.norm(point - centroids_kd[0])
        d_jump = np.linalg.norm(point - centroids_kd[1])
        d_run = np.linalg.norm(point - centroids_kd[2])
        distance_array = [d_walk, d_jump, d_run]
        test_labels[i] = np.argmin(distance_array)
        
    acc = accuracy_score(ground_truth_vector_test, test_labels.ravel())
    accuracy.append(acc)
    print(f"Accuracy for k = {k}: {acc * 100:.2f}%")

plt.plot(k_values, [acc * 100 for acc in accuracy])
plt.grid(True)
plt.title("Accuracy vs k modes for Test data")
plt.xlabel("k")
plt.ylabel("accuracy")


####BONUS KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

n_neighbors = 5


k_values = np.arange(1, 51)

train_accuracy_knn = []
test_accuracy_knn = []

for k in k_values:
    pca = PCA(k)
    X_train_kd = pca.fit_transform(X_train_T)
    X_test_kd = pca.transform(X_test_T)

    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train_kd, ground_truth_vector)
    
    train_pred = knn.predict(X_train_kd)
    test_pred = knn.predict(X_test_kd)
    
    train_acc = accuracy_score(ground_truth_vector, train_pred)
    test_acc = accuracy_score(ground_truth_vector_test, test_pred)
    
    train_accuracy_knn.append(train_acc)
    test_accuracy_knn.append(test_acc)
    
    print(f"PCA components: {k}, "
          f"KNN Train Accuracy: {train_acc*100:.2f}%, "
          f"KNN Test Accuracy: {test_acc*100:.2f}%")