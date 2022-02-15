from abc import ABC, abstractmethod
import numpy as np
from sklearn_lvq import GmlvqModel

from lvq import build_pairwise_lvq_classifiers


class RejectOption(ABC):
    def __init__(self, threshold, **kwds):
        self.threshold = threshold

        super().__init__(**kwds)

    @abstractmethod
    def criterion(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        return self.reject(x)

    def reject(self, x):
        return self.criterion(x) < self.threshold


class LvqRejectOption(RejectOption):
    def __init__(self, lvq_wrapped_model, **kwds):
        self.lvq_model = lvq_wrapped_model

        super().__init__(**kwds)

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError()


class LvqRejectRelSim(LvqRejectOption):
    def __init__(self, **kwds):
        super().__init__(**kwds)

    def fit(self, X, y):
        pass

    def criterion(self, x):
        distances_to_prototypes = [self.lvq_model.dist(self.lvq_model.prototypes[i], x) for i in range(len(self.lvq_model.prototypes))]

        pi_idx = np.argmin(distances_to_prototypes)
        dp = distances_to_prototypes[pi_idx]

        pi_label = self.lvq_model.prototypes_labels[pi_idx]
        other_prototypes_idx = np.where(self.lvq_model.prototypes_labels != pi_label)[0]
        dm = np.min([distances_to_prototypes[idx] for idx in other_prototypes_idx])

        return (dm - dp) / (dm + dp)


class LvqRejectDistDecisionBoundary(LvqRejectOption):
    def __init__(self, **kwds):
        super().__init__(**kwds)

    def fit(self, X, y):
        pass

    def criterion(self, x):
        distances_to_prototypes = [self.lvq_model.dist(self.lvq_model.prototypes[i], x) for i in range(len(self.lvq_model.prototypes))]

        pi_idx = np.argmin(distances_to_prototypes)
        p_i = self.lvq_model.prototypes[pi_idx]
        dp = distances_to_prototypes[pi_idx]
        pi_label = self.lvq_model.prototypes_labels[pi_idx]

        other_prototypes_idx = np.where(self.lvq_model.prototypes_labels != pi_label)[0]
        pj_idx = np.argmin([distances_to_prototypes[idx] for idx in other_prototypes_idx])
        p_j = self.lvq_model.prototypes[other_prototypes_idx[pj_idx]]
        dm = distances_to_prototypes[other_prototypes_idx[pj_idx]]

        return np.abs(dp - dm) / (2. * np.linalg.norm(p_i - p_j)**2)


class LvqRejectProbabilistic(LvqRejectOption):
    def __init__(self, pairwise_lvq_classifier_class=GmlvqModel, **kwds):
        self.pairwise_lvq_classifier_class = pairwise_lvq_classifier_class
        self.pairwise_wrapped_lvq_models = None
        self.num_classes = None

        super().__init__(**kwds)

    def fit(self, X_train, y_train):
        self.num_classes = len(np.unique(y_train))
        self.pairwise_wrapped_lvq_models = build_pairwise_lvq_classifiers(self.pairwise_lvq_classifier_class, X_train, y_train)

    def __compute_prob_ij(self, x, i, j):
        lvq_model_ij_data = self.pairwise_wrapped_lvq_models[i][j]
        lvq_model_ij, alpha, beta = lvq_model_ij_data["model"], lvq_model_ij_data["alpha"], lvq_model_ij_data["beta"]

        distances_to_prototypes = [lvq_model_ij.dist(lvq_model_ij.prototypes[i], x) for i in range(len(lvq_model_ij.prototypes))]

        pi_idx = np.argmin(distances_to_prototypes)
        dp = distances_to_prototypes[pi_idx]
        pi_label = lvq_model_ij.prototypes_labels[pi_idx]

        other_prototypes_idx = np.where(lvq_model_ij.prototypes_labels != pi_label)[0]
        pj_idx = np.argmin([distances_to_prototypes[idx] for idx in other_prototypes_idx])
        dm = distances_to_prototypes[other_prototypes_idx[pj_idx]]

        r_ij = (dm - dp) / (dm + dp)

        return 1. / (1. + np.exp(alpha * r_ij + beta))

    def __compute_prob_i(self, x, i):
        other_labels = list(range(self.num_classes));other_labels.remove(i)

        return 1. / (np.sum([1. / self.__compute_prob_ij(x, i, j) for j in other_labels]) - self.num_classes + 2)

    def criterion(self, x):
        return np.max([self.__compute_prob_i(x, i) for i in range(self.num_classes)])
