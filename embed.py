import numpy as np
import ot  # Library for Optimal Transport computations

class LOTEmbedding:
    # Define a small epsilon value to avoid division by zero errors
    EPISLON = np.finfo(float).eps

    @staticmethod
    def calc_OTmap(xr, xt, a=None, b=None, M=None, sinkhorn=False, lambd=1, normalize_T=False):
        """
        Compute the Optimal Transport (OT) map between a reference point cloud and a target point cloud.

        Parameters:
        -----------
        xr : ndarray
            Reference point cloud with shape (m, dr), where m is the number of points and dr is the dimensionality.
        
        xt : ndarray
            Target point cloud with shape (n, dt), where n is the number of points and dt is the dimensionality.
        
        a : ndarray, optional
            Mass distribution for the reference point cloud. If None, uniform distribution is used.
        
        b : ndarray, optional
            Mass distribution for the target point cloud. If None, uniform distribution is used.
        
        M : ndarray, optional
            Cost matrix representing pairwise distances between points in xr and xt. If None, it is computed.
        
        sinkhorn : bool, optional
            Whether to use the Sinkhorn algorithm for regularized OT. Default is False (use LP for exact OT).
        
        lambd : float, optional
            Regularization parameter for the Sinkhorn algorithm. Default is 1.
        
        normalize_T : bool, optional
            Whether to normalize the resulting transport map by the square root of the number of reference points.
            Default is False.

        Returns:
        --------
        T : ndarray
            The barycentric projection of the target point cloud onto the reference point cloud space.
            Shape: (m, dt)
        """
        # Get the dimensions of the reference and target point clouds
        m, dr = xr.shape
        n, dt = xt.shape
        
        # If mass distributions are not provided, assume uniform distributions
        if a is None:
            a = np.ones((m,)) / m  # Uniform distribution for reference point cloud
        if b is None:
            b = np.ones((n,)) / n  # Uniform distribution for target point cloud

        # If the cost matrix is not provided, compute it based on the Euclidean distance
        if M is None:
            assert dr == dt  # Ensure that the dimensionalities of xr and xt match
            M = ot.dist(xr, xt)  # Compute pairwise distance matrix between xr and xt

        # Compute the Optimal Transport plan G
        if sinkhorn:
            # Use the Sinkhorn algorithm for entropic regularization
            G = ot.sinkhorn(a, b, M, lambd)
        else:
            # Use the exact Linear Programming (LP) method to compute the OT plan
            G = ot.emd(a, b, M)

        # Normalize each row of the transport plan G to obtain a stochastic matrix
        Gstochastic = G / (G.sum(axis=1)[:, None] + LOTEmbedding.EPISLON)
        
        # Compute the barycentric projection T by applying the transport map to the target point cloud xt
        T = Gstochastic @ xt  # Shape: (m, dt)

        # Optionally normalize T by the square root of the number of points in xr
        if normalize_T:
            T = T / np.sqrt(m)
        
        return T  # Return the barycentric projection

    @staticmethod
    def embed_point_clouds(xr, xt_lst, r_mass=None, xt_masses=None, sinkhorn=False, lambd=1, normalize_T=False):
        """
        Embed a list of target point clouds into the reference point cloud space using LOT embeddings.

        Parameters:
        -----------
        xr : ndarray
            Reference point cloud to which the other point clouds will be compared. Shape: (m, dr)
        
        xt_lst : ndarray
            A list or array of target point clouds to embed. Shape: (num_point_clouds, n, dt)
        
        r_mass : ndarray, optional
            An array assigning mass to each point in the reference point cloud xr. Default is None (uniform mass).
        
        xt_masses : ndarray, optional
            A list of arrays assigning mass to each point in the corresponding target point cloud in xt_lst.
            Default is None (uniform mass).
        
        sinkhorn : bool, optional
            Whether to use the Sinkhorn algorithm for OT calculation. Default is False (use exact OT).
        
        lambd : float, optional
            Regularization parameter for the Sinkhorn algorithm. Default is 1.
        
        normalize_T : bool, optional
            Whether to normalize the transport maps by the square root of the number of reference points.
            Default is False.

        Returns:
        --------
        pclouds : ndarray
            A stacked array of flattened LOT embeddings of the target point clouds.
            Shape: (num_point_clouds, m*dt)
        """
        # Initialize an empty list to store the embeddings of each point cloud
        pclouds = []

        # Iterate over each target point cloud in the list xt_lst
        for j, xt in enumerate(xt_lst):
            # Determine the mass distribution for the reference and target point clouds
            a = None  # Mass for the reference point cloud
            if r_mass is not None: 
                a = r_mass
            b = None  # Mass for the target point cloud
            if xt_masses is not None: 
                b = xt_masses[j]
            
            # Compute the LOT embedding (barycentric projection) for the current target point cloud
            T = LOTEmbedding.calc_OTmap(xr, xt, a=a, b=b, sinkhorn=sinkhorn, lambd=lambd, normalize_T=normalize_T)
            
            # Flatten the embedding matrix T and add it to the list
            pclouds.append(T.reshape(-1))

        # Stack the flattened embeddings into a single array and return it
        return np.stack(pclouds)  # Shape: (num_point_clouds, m*dt)
