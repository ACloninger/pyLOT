import numpy as np
import ot

class LOTEmbedding:
    EPISLON = np.finfo(float).eps

    @staticmethod
    def calc_OTmap(xr, xt, a=None, b=None, M=None, sinkhorn=False, lambd=1, normalize_T=False):
        m, dr = xr.shape
        n, dt = xt.shape
        if a is None:
            a = np.ones((m,)) / m
        if b is None:
            b = np.ones((n,)) / n

        if M is None:
            assert dr == dt
            M = ot.dist(xr, xt) # cost matrix for Wasserstein distance computation

        if sinkhorn:
            # compute Sinkhorn plan
            G = ot.sinkhorn(a, b, M, lambd)
        else:
            # compute LP plan
            G = ot.emd(a, b, M)

        # row-normalize plan
        Gstochastic = G / (G.sum(axis=1)[:, None] + LOTEmbedding.EPISLON)
        T = Gstochastic @ xt # calculate barycentric projection

        if normalize_T:
            T = T / np.sqrt(m)
        return T
    

    @staticmethod
    def embed_point_clouds(xr, xt_lst, sinkhorn=False, lambd=1):
        """
        Embeds a list of point clouds using LOT embeddings.

        Parameters:
        -----------
        xr : ndarray
            Reference point cloud to which the other point clouds will be compared.
        xt_lst : ndarray
            A list or ndarray of target point clouds to embed.
        sinkhorn : bool, optional
            Whether to use the Sinkhorn algorithm for OT calculation. Default is False.
        lambd : float, optional
            Regularization parameter for Sinkhorn algorithm. Default is 1.

        Returns:
        --------
        pclouds : ndarray
            Matrix of flattened LOT embeddings of the target point clouds.
        """
        pclouds = []

        for xt in xt_lst:
            T = LOTEmbedding.calc_OTmap(xr, xt, sinkhorn=sinkhorn, lambd=lambd)
            pclouds.append(T.reshape(-1))

        return np.stack(pclouds)