import numpy as np
from imblearn.over_sampling import RandomOverSampler  # For balancing imbalanced datasets by oversampling
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # For Linear Discriminant Analysis (LDA)

class LOTDimensionalityReduction:
    @staticmethod
    def balance_data(pclouds, labels):
        """
        Balance the dataset by oversampling minority classes.

        Parameters
        ----------
        pclouds : np.array
            A 2D array where each row represents a point cloud (data point).
            Shape: (number_of_point_clouds, number_of_features)
        
        labels : np.array
            A 1D array of labels corresponding to the point clouds in `pclouds`.
            Shape: (number_of_point_clouds,)

        Returns
        -------
        pclouds_balanced : np.array
            The balanced point clouds after oversampling.
        
        labels_balanced : np.array
            The corresponding labels for the balanced point clouds.
        """
        # Use RandomOverSampler to balance the dataset by oversampling the minority class(es)
        return RandomOverSampler().fit_resample(pclouds, labels)

    @staticmethod
    def lda_reduction(pclouds, labels, n_components=3):
        """
        Perform Linear Discriminant Analysis (LDA) to reduce dimensionality.

        Parameters
        ----------
        pclouds : np.array
            A 2D array where each row represents a point cloud (data point).
            Shape: (number_of_point_clouds, number_of_features)
        
        labels : np.array
            A 1D array of labels corresponding to the point clouds in `pclouds`.
            Shape: (number_of_point_clouds,)
        
        n_components : int, optional
            Number of components to keep after LDA. Default is 3.

        Returns
        -------
        T_lda : np.array
            The point clouds transformed to the new space defined by the LDA components.
            Shape: (number_of_point_clouds_balanced, n_components)
        
        labels_balanced : np.array
            The corresponding labels for the transformed point clouds.
        """
        # Create an LDA object with the desired number of components
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        
        # Balance the dataset before performing LDA
        pclouds_balanced, labels_balanced = LOTDimensionalityReduction.balance_data(pclouds, labels)
        
        # Fit the LDA model on the balanced data and transform the point clouds
        T_lda = lda.fit(pclouds_balanced, labels_balanced).transform(pclouds_balanced)
        
        # Return the transformed data and the balanced labels
        return T_lda, labels_balanced

    @staticmethod
    def pca_reduction(pclouds, labels):
        """
        Perform Principal Component Analysis (PCA) to reduce dimensionality.

        Parameters
        ----------
        pclouds : np.array
            A 2D array where each row represents a point cloud (data point).
            Shape: (number_of_point_clouds, number_of_features)
        
        labels : np.array
            A 1D array of labels corresponding to the point clouds in `pclouds`.
            Shape: (number_of_point_clouds,)

        Returns
        -------
        U : np.array
            The left singular vectors (principal components) of the centered data.
            Shape: (number_of_point_clouds_balanced, min(number_of_point_clouds_balanced, number_of_features))
        
        S : np.array
            The singular values corresponding to the principal components.
            Shape: (min(number_of_point_clouds_balanced, number_of_features),)
        
        Vh : np.array
            The right singular vectors, which are the principal axes in feature space.
            Shape: (min(number_of_point_clouds_balanced, number_of_features), number_of_features)
        
        labels_balanced : np.array
            The corresponding labels for the balanced point clouds.
        """
        # Balance the dataset before performing PCA
        pclouds_balanced, labels_balanced = LOTDimensionalityReduction.balance_data(pclouds, labels)
        
        # Calculate the mean of the balanced point clouds
        mu = np.mean(pclouds_balanced, axis=0)
        
        # Center the point clouds by subtracting the mean
        centered_data = np.array(pclouds_balanced - mu, dtype=np.float64)
        
        # Perform Singular Value Decomposition (SVD) on the centered data
        U, S, Vh = np.linalg.svd(centered_data, full_matrices=False)
        
        # Return the SVD results (principal components) and the balanced labels
        return U, S, Vh, labels_balanced
