from keras.datasets import mnist
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load MNIST database
(train_images_raw, train_labels), (test_images_raw, test_labels) = mnist.load_data()

# Get image parameters
n_train, h, w = train_images_raw.shape
n_test, h, w = test_images_raw.shape

# Vectorize the images
train_images = [np.reshape(image, -1) for image in train_images_raw]
test_images = [np.reshape(image, -1) for image in test_images_raw]

# Perform PCA, retaining 95% of variance in data
pca = PCA(n_components = 0.95, svd_solver = 'full')
train_images_pca = pca.fit_transform(train_images)
test_images_pca = pca.transform(test_images)

# Predict labels for test images using kNN with 1 neighbor
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(train_images_pca, train_labels)
predicted_labels = knn.predict(test_images_pca)

# Measure Accuracy
acc = accuracy_score(test_labels, predicted_labels)
print(f"Accuracy: {acc*100}%")