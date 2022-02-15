import numpy as np
import sklearn_lvq


from platt_scaling import calc_mu
from train_sigmoid import train_sigmoid


class LvqWrapper():
    def __init__(self, lvq_model, **kwds):
        self.lvq_model = lvq_model
        self.prototypes, self.prototypes_labels, self.dist_mat = self.__wrap_lvq_model()
        
        super().__init__(**kwds)

    def __wrap_lvq_model(self):
        dist_mat = None
        if isinstance(self.lvq_model, sklearn_lvq.GlvqModel) or isinstance(self.lvq_model, sklearn_lvq.RslvqModel):
            dist_mat = np.eye(self.lvq_model.w_.shape[1])
        elif isinstance(self.lvq_model, sklearn_lvq.GmlvqModel) or isinstance(self.lvq_model, sklearn_lvq.MrslvqModel):
            dist_mat = np.dot(self.lvq_model.omega_.T, self.lvq_model.omega_)
        else:
            raise TypeError(f"'lvq_model' must be an instance of 'sklearn_lvq.GlvqModel', 'sklearn_lvq.GmlvqModel', 'sklearn_lvq.RslvqModel' or 'sklearn_lvq.MrslvqModel' but not of {type(self.lvq_model)}")

        return self.lvq_model.w_, self.lvq_model.c_w_, dist_mat

    def dist(self, a, b):
        return (a-b).T @ self.dist_mat @ (a-b)


def build_pairwise_lvq_classifiers(lvq_class, X_train, y_train, n_prototypes_per_class=2):
    results = {}
    
    # Get unique labels
    labels_unqiue = np.unique(y_train)

    # Consider all possible pairs of labels
    for label_i in labels_unqiue:
        for label_j in labels_unqiue:
            if label_i == label_j:
                continue
            if (label_i in results and label_j in results[label_i]) or (label_j in results and label_i in results[label_j]):
                continue

            if label_i not in results:
                results[label_i] = {}
            if label_j not in results:
                results[label_j] = {}

            # Get training data
            idx = list(filter(lambda i: y_train[i] == label_i or y_train[i] == label_j, range(len(y_train))))
            X_train_ = X_train[idx, :]
            y_train_ = y_train[idx]

            # Encode labels as 0 and 1
            idx_i = y_train_ == label_i
            idx_j = y_train_ == label_j
            y_train_[idx_i] = 0
            y_train_[idx_j] = 1

            # Fit classifier 
            model = lvq_class(prototypes_per_class=n_prototypes_per_class)
            model.fit(X_train_, y_train_)

            # Platt-Scaling
            #alpha = 1.
            #beta = 0.
            target =  model.predict(X_train_) == 1
            prior1 = sum(target)
            prior0 = sum(np.invert(target))
            mu = calc_mu(X_train_, model, target)
            alpha, beta = train_sigmoid(mu, target, prior1, prior0)

            # Store
            wrapped_model = LvqWrapper(model)
            results[label_i][label_j] = {"model": wrapped_model, "alpha": alpha, "beta": beta}
            results[label_j][label_i] = {"model": wrapped_model, "alpha": alpha, "beta": beta}

    return results
