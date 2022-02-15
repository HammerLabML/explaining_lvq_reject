import os
import random
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn_lvq import GlvqModel, GmlvqModel

from reject_option import (LvqRejectDistDecisionBoundary,
                           LvqRejectProbabilistic, LvqRejectRelSim)
from counterfactual import (LvqRejectOptionDistDecisionBoundaryCounterfactual,
                            LvqRejectOptionProbaCounterfactual,
                            LvqRejectOptionRelSimCounterfactual,
                            LvqRejectOptionProbaCounterfactualBlackBox)
from data_preparation import scale_standardize_data

lvq_parameters = {'prototypes_per_class':[1,2,3,4,5]}
reject_thresholds = [0.001, 0.005, 0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 1, 2, 3, 4, 5, 7, 10, 100, 1000]

n_folds = 5
non_zero_threshold = 1e-5
non_zero_threshold_sparsity = 1e-5

def load_data(data_desc, data_folder="../../", scaling=True):
    if data_desc == "iris":
        X, y = load_iris(return_X_y=True)
        if scaling is True:
            X = scale_standardize_data(X)
        return X, y
    elif data_desc == "breastcancer":
        X, y = load_breast_cancer(return_X_y=True)
        if scaling is True:
            X = scale_standardize_data(X)
        return X, y
    elif data_desc == "wine":
        X, y = load_wine(return_X_y=True)
        if scaling is True:
            X = scale_standardize_data(X)
        return X, y
    elif data_desc == "flip":
        flip_data = np.load(os.path.join(data_folder, f"datasets_formatted/flip.npz"))
        if scaling is False:
            flip_data = np.load(os.path.join(data_folder, f"datasets_formatted/flip_notscaled.npz"))
        return flip_data['X'], flip_data['y']
    elif data_desc == "t21":
        t21_data = np.load(os.path.join(data_folder, f"datasets_formatted/t21.npz"))
        if scaling is False:
            t21_data = np.load(os.path.join(data_folder, f"datasets_formatted/t21_notscaled.npz"))
        return t21_data['X'], t21_data['y']
    elif data_desc == "coil-20":
        coil_20_data = np.load(os.path.join(data_folder, f"datasets_formatted/coil_20.npz"))
        return coil_20_data['X'], coil_20_data['y']
    elif data_desc == "usps":
        usps_data = np.load(os.path.join(data_folder, "datasets_formatted/usps.npz"))
        return usps_data['X'], usps_data['y']
    elif data_desc == "letter":
        letter_data = np.load(os.path.join(data_folder, "datasets_formatted/letter.npz"), allow_pickle=True)
        return letter_data['X'], letter_data['y']
    elif data_desc == "mce-nose":
        mce_nose_data = np.load(os.path.join(data_folder, "datasets_formatted/mce_nose.npz"))
        return mce_nose_data['X'], mce_nose_data['y']
    elif data_desc == "motion":
        motion_data = np.load(os.path.join(data_folder, "datasets_formatted/motion.npz"))
        return motion_data['X'], motion_data['y']
    elif data_desc == "image-segment":
        image_segment_data = np.load(os.path.join(data_folder, "datasets_formatted/image_segment.npz"), allow_pickle=True)
        return image_segment_data['X'], image_segment_data['y']
    else:
        raise ValueError(f"Unkown data set {data_desc}")


def get_lvq_model_class(lvq_desc):
    if lvq_desc == "glvq":
        return GlvqModel
    elif lvq_desc == "gmlvq":
        return GmlvqModel
    else:
        raise ValueError(f"Unkown LVQ model {lvq_desc}")


def get_reject_option_class(reject_desc):
    if reject_desc == "relsim":
        return LvqRejectRelSim, LvqRejectOptionRelSimCounterfactual
    elif reject_desc == "distdecbound":
        return LvqRejectDistDecisionBoundary, LvqRejectOptionDistDecisionBoundaryCounterfactual
    elif reject_desc == "proba":
        return LvqRejectProbabilistic, LvqRejectOptionProbaCounterfactual
    elif reject_desc == "proba_blackboxcf":
        return LvqRejectProbabilistic, LvqRejectOptionProbaCounterfactualBlackBox
    else:
        raise ValueError(f"Unknown reject option {reject_desc}")


def closest_sample_counterfactual(X_train, y_reject, x_orig):
    # Compute l1 distance to each sample in the training set
    d = np.linalg.norm(X_train - x_orig, ord=1, axis=1)

    # Select the closest sample which is not rejected
    indices = np.flip(np.argsort(d))
    idx = list(filter(lambda i: i not in y_reject, indices))[0]
    
    return X_train[idx, :]


def evaluate_sparsity(xcf, x_orig):
    return evaluate_sparsity_ex(xcf - x_orig)

def evaluate_sparsity_ex(x):
    return np.sum(np.abs(x[i]) > non_zero_threshold_sparsity for i in range(x.shape[0]))    # Count non-zero features (smaller values are better!)


def evaluate_featureoverlap(xcf1, xcf2, x_orig):
    # Find non-zero features
    a = np.array([np.abs(xcf1[i] - x_orig[i]) > non_zero_threshold for i in range(xcf1.shape[0])]).astype(np.int)
    b = np.array([np.abs(xcf2[i] - x_orig[i]) > non_zero_threshold for i in range(xcf2.shape[0])]).astype(np.int)
    
    # Look for overlaps
    return np.sum(a + b == 2)


def select_random_feature_subset(n_features, size=0.3):
    n_subset_size = int(n_features * size)

    return random.sample(range(n_features), n_subset_size)


def apply_perturbation(X, features_idx, noise_size=1.):
    scale = noise_size  # Scale/amount/variance of noise
    X[:, features_idx] += np.random.normal(scale=scale, size=(X.shape[0], len(features_idx)))
    return X


def evaluate_perturbed_features_recovery(xcf, x_orig, perturbed_features_idx):
    return evaluate_perturbed_features_recovery_ex(np.abs(xcf - x_orig), perturbed_features_idx)

def evaluate_perturbed_features_recovery_ex(x, perturbed_features_idx):
    indices = np.argwhere(x > non_zero_threshold)

    # Compute confusion matrix
    tp = np.sum([idx in perturbed_features_idx for idx in indices]) / len(indices)
    fp = np.sum([idx not in perturbed_features_idx for idx in indices]) / len(indices)
    tn = np.sum([idx not in perturbed_features_idx for idx in filter(lambda i: i not in indices, range(x.shape[0]))]) / len(indices)
    fn = np.sum([idx in perturbed_features_idx for idx in filter(lambda i: i not in indices, range(x.shape[0]))]) / len(indices)

    if len(indices) != 0:
        return tp / (tp + fn)  # Compute recall
    else:
        return 0


def compute_export_perturbed_features_recovery_results(perturbed_features_recovery_counterfactual, perturbed_features_recovery_closestsample, perturbed_features_recovery_blackbox):
    # Compute final statistics (TODO: Compute statsitics over all folds directly or per fold?)
    perturbed_features_recovery_counterfactual_mean, perturbed_features_recovery_counterfactual_var = np.mean(perturbed_features_recovery_counterfactual), np.var(perturbed_features_recovery_counterfactual)
    perturbed_features_recovery_closestsample_mean, perturbed_features_recovery_closestsample_var = np.mean(perturbed_features_recovery_closestsample), np.var(perturbed_features_recovery_closestsample)
    perturbed_features_recovery_blackbox_mean, perturbed_features_recovery_blackbox_var = np.mean(perturbed_features_recovery_blackbox), np.var(perturbed_features_recovery_blackbox)

    # Export
    print(f"Perturbed features recovery counterfactual: {perturbed_features_recovery_counterfactual_mean} \pm {perturbed_features_recovery_counterfactual_var}")
    print(f"Perturbed features recovery closest sample: {perturbed_features_recovery_closestsample_mean} \pm {perturbed_features_recovery_closestsample_var}")
    print(f"Perturbed features recovery black-box: {perturbed_features_recovery_blackbox_mean} \pm {perturbed_features_recovery_blackbox_var}")
    
    # LaTeX export
    print(f"${np.round(perturbed_features_recovery_counterfactual_mean, 2)} \pm {np.round(perturbed_features_recovery_counterfactual_var, 2)}$ &\
        ${np.round(perturbed_features_recovery_closestsample_mean, 2)} \pm {np.round(perturbed_features_recovery_closestsample_var, 2)}$ &\
        ${np.round(perturbed_features_recovery_blackbox_mean, 2)} \pm {np.round(perturbed_features_recovery_blackbox_var, 2)}$")


def compute_export_results(black_box_feasibility, white_box_feasibility, sparsity_blackbox_counterfactual, sparsity_closest_sample, sparsity_counterfactual, overlap_closestsample_counterfactual, overlap_closestsample_blackbox, overlap_counterfactual_blackbox):
    # Compute final statistics (TODO: Compute statsitics over all folds directly or per fold?)
    black_box_feasibility_mean, black_box_feasibility_var  = np.mean(black_box_feasibility), np.var(black_box_feasibility)
    white_box_feasibility_mean, white_box_feasibility_var = np.mean(white_box_feasibility), np.var(white_box_feasibility)
    sparsity_blackbox_counterfactual_mean, sparsity_blackbox_counterfactual_var = np.mean(sparsity_blackbox_counterfactual), np.var(sparsity_blackbox_counterfactual)
    sparsity_closest_sample_mean, sparsity_closest_sample_var = np.mean(sparsity_closest_sample), np.var(sparsity_closest_sample)
    sparsity_counterfactual_mean, sparsity_counterfactual_var = np.mean(sparsity_counterfactual), np.var(sparsity_counterfactual)
    overlap_closestsample_counterfactual_mean, overlap_closestsample_counterfactual_var = np.mean(overlap_closestsample_counterfactual), np.var(overlap_closestsample_counterfactual)
    overlap_closestsample_blackbox_mean, overlap_closestsample_blackbox_var = np.mean(overlap_closestsample_blackbox), np.var(overlap_closestsample_blackbox)
    overlap_counterfactual_blackbox_mean, overlap_counterfactual_blackbox_var = np.mean(overlap_counterfactual_blackbox), np.var(overlap_counterfactual_blackbox)

    # Export
    print(f"Black-box feasibility: {black_box_feasibility_mean} \pm {black_box_feasibility_var}")
    print(f"Black-box sparsity: {sparsity_blackbox_counterfactual_mean} \pm {sparsity_blackbox_counterfactual_var}")
    print(f"Closest sample sparsity: {sparsity_closest_sample_mean} \pm {sparsity_closest_sample_var}")
    print(f"Counterfactual sparsity: {sparsity_counterfactual_mean} \pm {sparsity_counterfactual_var}")
    print(f"Counterfactual feasibility: {white_box_feasibility_mean} \pm {white_box_feasibility_var}")
    print(f"Overlap closest sample vs. counterfactual: {overlap_closestsample_counterfactual_mean} \pm {overlap_closestsample_counterfactual_var}")
    print(f"Overlap closest sample vs. black-box: {overlap_closestsample_blackbox_mean} \pm {overlap_closestsample_blackbox_var}")
    print(f"Overlap counterfactual vs. black-box: {overlap_counterfactual_blackbox_mean} \pm {overlap_counterfactual_blackbox_var}")
   
    # LaTeX export
    print(f"${np.round(black_box_feasibility_mean, 2)} \pm {np.round(black_box_feasibility_var, 2)}$ &\
        ${np.round(sparsity_blackbox_counterfactual_mean, 2)} \pm {np.round(sparsity_blackbox_counterfactual_var, 2)}$ &\
        ${np.round(sparsity_closest_sample_mean)} \pm {np.round(sparsity_closest_sample_var)}$ &\
        ${np.round(sparsity_counterfactual_mean, 2)} \pm {np.round(sparsity_counterfactual_var, 2)}$")
    print(f"${np.round(overlap_closestsample_counterfactual_mean, 2)} \pm {np.round(overlap_closestsample_counterfactual_var, 2)}$ &\
        ${np.round(overlap_closestsample_blackbox_mean, 2)} \pm {np.round(overlap_closestsample_blackbox_var, 2)}$ &\
        ${np.round(overlap_counterfactual_blackbox_mean, 2)} \pm {np.round(overlap_counterfactual_blackbox_var, 2)}$")