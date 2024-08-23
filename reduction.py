import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LOTDimensionalityReduction:
    @staticmethod
    def balance_data(pclouds, labels):
        return RandomOverSampler().fit_resample(pclouds, labels)

    @staticmethod
    def lda_reduction(pclouds, labels, n_components=3):
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        pclouds_balanced, labels_balanced =  LOTDimensionalityReduction.balance_data(pclouds, labels)
        T_lda = lda.fit(pclouds_balanced, labels_balanced).transform(pclouds_balanced)
        return T_lda

    @staticmethod
    def pca_reduction(pclouds, labels):
        pclouds_balanced, labels_balanced =  LOTDimensionalityReduction.balance_data(pclouds, labels)
        mu = np.mean(pclouds_balanced, axis=0)
        U, S, Vh = np.linalg.svd(pclouds_balanced - mu, full_matrices=False)
        return U, S, Vh
