import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn_lvq import GlvqModel, GmlvqModel

from train_sigmoid import train_sigmoid


def calc_mu(X, GMLVQ_model, posClass=None):
    # calculate μ for a given GMLVQ model for the zscored data

#     distances = zeros(size(data,1),length(GMLVQ_model.c_w));
#     for i = 1:size(GMLVQ_model.w,1)
#         distances(:,i) = sum((bsxfun(@minus, data, GMLVQ_model.w(i,:))*GMLVQ_model.omega').^2, 2);
#     end
    distances = GMLVQ_model._compute_distance(X)
    # distances = computeDistance(X,GMLVQ_model.w,GMLVQ_model)  # for each point compute distance to prototypes
    # classify all datapoinst, if no class-label is given
    if posClass is None:
        idx = np.argmin(distances, 1)
        posClass = GMLVQ_model.c_w_[idx]
    
    # d+ is the nearest prototyp and d- the nearest from a different class
    # as d+
    if len(posClass) == len(X):
        mu = np.zeros(len(posClass))
        for pC in np.unique(posClass):

            distNeg = np.min(distances[posClass==pC,:][:,GMLVQ_model.c_w_!=pC], 1)
            distPos = np.min(distances[posClass==pC,:][:, GMLVQ_model.c_w_==pC], 1)
            # Speicher die Klassifikation der einzelnen Messer in estimatedClasses ab
            mu[posClass==pC] = (distNeg - distPos) / (distNeg + distPos)
    else:
    # d+ is the nearest prototyp for the given class posClass and d- the
    # nearest for a different one
        distNeg = np.min(distances[:, GMLVQ_model.c_w_!=posClass], 1)
        distPos = np.min(distances[:, GMLVQ_model.c_w_==posClass], 1)
        # Speicher die Klassifikation der einzelnen Messer in estimatedClasses ab
        mu = (distNeg - distPos) / (distNeg + distPos)

    return mu

def platt_scaling_example():
    X, y = load_iris(return_X_y=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Fit LVQ model
    model = GmlvqModel(prototypes_per_class=3, random_state=444)
    model.fit(X_train, y_train)
    print("single model score: ", model.score(X_test, y_test))

    c = 2  # compute for class 2
    target = model.predict(X_test) == c
    prior1 = sum(target)
    prior0 = sum(np.invert(target))

    # Calculate mu for points in test
    mu = calc_mu(X_test, model, target)
    print('mu:', mu)

    # Compute A and B of sigmoid
    A, B = train_sigmoid(mu, target, prior1, prior0)
    print('A B:', A, B)

if __name__ == "__main__":
    platt_scaling_example()
