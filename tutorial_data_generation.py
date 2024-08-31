import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

def generate_mnist_clouds():
    """
    Generates 2D point clouds from the MNIST dataset using non-zero pixel coordinates.

    Returns:
    - pclouds: List of 2D arrays, each containing the point cloud for one image
    - masses: List of 1D arrays, each containing the normalized masses for the corresponding point cloud
    - labels: List of integers, each corresponding to the label of the associated point cloud
    - class_clouds: List of lists of 2D arrays, where each sublist contains the point clouds for one class
    """
    # Load MNIST data
    (X_train, y_train), _ = mnist.load_data()
    
    pclouds = []
    masses = []
    labels = []
    class_clouds = [[] for _ in range(10)]  # List of lists, one for each class (0-9)
    
    # Loop through each class (0-9)
    for class_id in np.unique(y_train):
        class_data = X_train[y_train == class_id]
        
        # Loop through each image in the class
        for img in class_data:
            # Extract the coordinates of non-zero pixels
            pixel_indices = np.array(np.nonzero(img)).T
            # Extract the corresponding pixel values (masses)
            pixel_values = img[img > 0].astype(float)
            # Normalize the masses so that they sum to 1
            pixel_values /= pixel_values.sum()
            
            # Append the point cloud, its mass, and corresponding label
            pclouds.append(pixel_indices)
            masses.append(pixel_values)
            labels.append(class_id)
            
            # Append the point cloud to the appropriate class list
            class_clouds[class_id].append(pixel_indices)
    
    # Convert lists to numpy arrays for easier indexing
    pclouds = np.array(pclouds, dtype=object)
    masses = np.array(masses, dtype=object)
    labels = np.array(labels)
    class_clouds = [np.array(cloud_list, dtype=object) for cloud_list in class_clouds]
    
    return pclouds, masses, labels, class_clouds


def generate_gauss_point_clouds(num_classes, n_samples_per_cloud, n_features, n_clouds_per_class, base_means=None, noise_std=0.5):
    """
    Generates multiple point clouds per class with added noise to the mean.

    Parameters:
    - num_classes: Number of different classes
    - n_samples_per_cloud: Number of samples in each point cloud
    - n_features: Number of features (dimensions) for each point cloud
    - n_clouds_per_class: Number of point clouds per class
    - base_means: means for each of the classes
    - noise_std: Standard deviation of the noise added to the mean of each cloud

    Returns:
    - pclouds: List of 2D arrays, each containing the point cloud for one generated sample
    - labels: List of integers, each corresponding to the label of the associated point cloud
    - class_clouds: List of lists of 2D arrays, where each sublist contains the point clouds for one class
    """
    if base_means is None:
        base_means = [3 * np.random.randn(n_features) for _ in range(num_classes)]
    
    pclouds = []
    labels = []
    class_clouds = [[] for _ in range(num_classes)]  # List of lists, one for each class
    
    for class_id, base_mean in enumerate(base_means):
        # Generate multiple point clouds for each class
        for _ in range(n_clouds_per_class):
            # Add noise to the mean
            noisy_mean = base_mean + np.random.randn(n_features) * noise_std
            # Generate a point cloud around the noisy mean
            pcloud = np.random.randn(n_samples_per_cloud, n_features) + noisy_mean
            # Append the point cloud and corresponding label
            pclouds.append(pcloud)
            labels.append(class_id)
            # Add this point cloud to the list of clouds for the current class
            class_clouds[class_id].append(pcloud)
    
    # Convert lists to numpy arrays for easier indexing
    pclouds = np.array(pclouds, dtype=object)
    labels = np.array(labels)
    class_clouds = [np.array(cloud_list, dtype=object) for cloud_list in class_clouds]
    
    return pclouds, labels, class_clouds
