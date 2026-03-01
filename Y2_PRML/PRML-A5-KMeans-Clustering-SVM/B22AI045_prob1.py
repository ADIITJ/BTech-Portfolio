# Assignment 5
## Q1
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
### a)
def computeCentroid(feat):
    return np.mean(feat, axis = 0)
### b)
def mykmeans(X, k):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(100):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        new_centroids = np.array([computeCentroid(X[labels == i]) for i in range(k)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids

def compress_image(image, k):
    pixels = np.reshape(image, (-1, 3))
    centroids = mykmeans(pixels, k)
    distances = np.sqrt(((pixels[:, np.newaxis] - centroids)**2).sum(axis=2))
    compressed_pixels = centroids[np.argmin(distances, axis=1)]
    compressed_image = np.reshape(compressed_pixels, image.shape)
    return compressed_image
### c)
image_path = 'test.png'
image = Image.open(image_path)

image_np = np.array(image)

image_reshaped = image_np.reshape(-1, 3)
print(image_np.shape)
print(image_reshaped.shape)
compressed_image2 = compress_image(image_np, 3)
compressed_image2 = compressed_image2.astype(np.uint8)
for k in [1, 2, 4, 8]:
    plt.imshow(compress_image(image_np, k).astype(np.uint8))
    plt.title(f'Compressed image with k={k}')
    plt.show()
### d)
from sklearn.cluster import KMeans

def compress_image_sklearn(image, k):
    pixels = np.reshape(image, (-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = np.reshape(compressed_pixels, image.shape)
    return compressed_image

compressed_image1 = compress_image_sklearn(image_np, 3)

compressed_image1 = compressed_image1.astype(np.uint8)

fig, ax = plt.subplots(1, 3, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(compressed_image1)
ax[1].set_title('sklearn ({} colors)'.format(3))
ax[1].axis('off')

ax[2].imshow(compressed_image2)
ax[2].set_title('mykmeans ({} colors)'.format(3))
ax[2].axis('off')

plt.show()
#### Difference is in execution time.

import time
start_time = time.time()
sklearn_lib = compress_image_sklearn(image_np, 3)
end_time = time.time()
elapsed_time3 = end_time - start_time
print("Time taken for compressing image with sklearn: {:.4f} seconds".format(elapsed_time3))

start_time = time.time()
my_sklearn = compress_image(image_np, 3)
end_time = time.time()
elapsed_time4 = end_time - start_time
print("Time taken for compressing image with mykmeans: {:.4f} seconds".format(elapsed_time4))

plt.imshow(-compressed_image2+compressed_image1)
plt.title(f'sklearn_kmeans_img-mykmeans_img with k=3')
plt.show()
### e)
from scipy.ndimage import gaussian_filter
def compress_blur_image(image, k):
    blurred_image = gaussian_filter(image, sigma=1)
    pixels = np.reshape(blurred_image, (-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = np.reshape(compressed_pixels, blurred_image.shape)
    return compressed_image
compressed_image = compress_blur_image(image_np, 3)

compressed_image = compressed_image.astype(np.uint8)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(image)
ax[0, 0].set_title('Original')
ax[0, 0].axis('off')

ax[0, 1].imshow(compressed_image)
ax[0, 1].set_title('Compressed Image with Gaussian filtering')
ax[0, 1].axis('off')

ax[1, 0].imshow(compressed_image1)
ax[1, 0].set_title('sklearn ({} colors)'.format(3))
ax[1, 0].axis('off')

ax[1, 1].imshow(compressed_image2)
ax[1, 1].set_title('mykmeans ({} colors)'.format(3))
ax[1, 1].axis('off')

plt.show()

plt.show()

#### Comparing time
start_time = time.time()
compressed_image = compress_blur_image(image_np, 3)
end_time = time.time()
elapsed_time1 = end_time - start_time
print("Time taken for compressing image with Gaussian filter: {:.4f} seconds".format(elapsed_time1))

start_time = time.time()
compressed_image1 = compress_image_sklearn(image_np, 3)
end_time = time.time()
elapsed_time3 = end_time - start_time
print("Time taken for compressing image with sklearn: {:.4f} seconds".format(elapsed_time3))

start_time = time.time()
compressed_image2 = compress_image(image_np, 3)
end_time = time.time()
elapsed_time4 = end_time - start_time
print("Time taken for compressing image with mykmeans: {:.4f} seconds".format(elapsed_time4))


print("Time taken for bluring is : {:.4f} seconds".format(elapsed_time1-elapsed_time3))
