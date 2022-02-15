import sys
import os
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from lvq import LvqWrapper
from counterfactual import LvqRejectOptionCounterfactualBlackBoxSolver
from modelselection import ExplainableRejectOptionGridSearchCV

from utils import *


if __name__ == "__main__":
    # - Load data (given as an argument)
    # K-Fold:
    #  - Hyperparameter tuning LVQ and reject option (type of LVQ and reject option are given as arguments)
    #  - Fit LVQ and reject option
    #  - Select random subset of features that are going to be perturbed
    #  - Apply perturbation (e.g. adding noise) to the feature subset (make sure that enough samples are rejected due to the perturbation!)
    #  For all test samples that are rejected due to the perturbation:
    #   - Compute counterfactual with our proposed method (convex program)
    #   - Compute counterfactual using black-box method & check feasibility (i.e. validity)
    #   - Select closest sample from the training set that is not rejected
    #   - Evaluate sparsity (l0-norm) of all three counterfactuals
    #   - Evaluate feature overlap
    #   - Check for each counterfactual (method) if and how many of the perturbed features are recovered

    if len(sys.argv) != 4:
        print("Usage: <dataset> <lvq-model> <reject-option>")
        os._exit(1)

    # Specifications (provided as an input by the user)
    data_desc = sys.argv[1]
    lvq_desc = sys.argv[2]
    reject_desc = sys.argv[3]

    # Load data
    X, y = load_data(data_desc, scaling=reject_desc != "proba" and reject_desc != "proba_blackboxcf")
    print(X.shape)

    # Get lvq model and reject option
    lvq_model_class = get_lvq_model_class(lvq_desc)
    reject_option_class, reject_option_counterfactual_method = get_reject_option_class(reject_desc)

    # Results/Statistics
    black_box_feasibility = []
    white_box_feasibility = []
    sparsity_closest_sample = []
    sparsity_counterfactual = []
    sparsity_blackbox_counterfactual = []
    overlap_closestsample_counterfactual = []
    overlap_closestsample_blackbox = []
    overlap_counterfactual_blackbox = []
    perturbed_features_recovery_counterfactual = []
    perturbed_features_recovery_blackbox = []
    perturbed_features_recovery_closestsample = []

    # In case of an extremly large majority class, perform simple downsampling
    if data_desc == "t21":
        rus = RandomUnderSampler()
        X, y = rus.fit_resample(X, y)

    # K-Fold
    for train_index, test_index in KFold(n_splits=n_folds, shuffle=True, random_state=None).split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # If necessary (in case of an highly imbalanced data set), apply Synthetic Minority Over-sampling Technique (SMOTE)
        if data_desc == "flip":
            sm = SMOTE(k_neighbors=1)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            X_test, y_test = sm.fit_resample(X_test, y_test)

        # Hyperparameter tuning
        model_search = ExplainableRejectOptionGridSearchCV(lvq_model=lvq_model_class, lvq_parameters=lvq_parameters, reject_option_model=reject_option_class, rejection_thresholds=reject_thresholds)
        best_params = model_search.fit(X_train, y_train)

        # Fit & evaluate model and reject option
        print(best_params["lvq_params"])
        model = lvq_model_class(**best_params["lvq_params"])
        model.fit(X_train, y_train)
        print(f"Model score: {model.score(X_train, y_train)}, {model.score(X_test, y_test)}")

        print(f'Rejection threshold: {best_params["rejection_threshold"]}')
        reject_option = reject_option_class(lvq_wrapped_model=LvqWrapper(model), threshold=best_params["rejection_threshold"])
        reject_option.fit(X_train, y_train)   # Fit/Calibrate reject option

        cf_method = reject_option_counterfactual_method(reject_option_model=reject_option)
        cf_blackbox_method = LvqRejectOptionCounterfactualBlackBoxSolver(reject_option_model=reject_option)

        # Select random subset of features which are going to be perturbed
        perturbed_features_idx = select_random_feature_subset(X_train.shape[1])
        print(f"Perturbed features: {perturbed_features_idx}")

        # For each sample in the test set, check if it is rejected
        y_rejects = []
        for i in range(X_test.shape[0]):
            x = X_test[i,:]
            if reject_option(x):
                y_rejects.append(i)
        print(f"{len(y_rejects)}/{X_test.shape[0]} are rejected")

        # Find all samples in the test set that are rejected because of the perturbation
        noise_size = 1.
        X_test = apply_perturbation(X_test, perturbed_features_idx, noise_size)  # Apply perturbation
        
        y_rejects_due_to_perturbations = []
        for i in range(X_test.shape[0]):    # Check which samples are now rejected
            x = X_test[i,:]
            if reject_option(x) and i not in y_rejects:
                y_rejects_due_to_perturbations.append(i)
        print(f"{len(y_rejects_due_to_perturbations)}/{X_test.shape[0]} are rejected due to perturbations")

        # Compute counterfactual explanations for all rejected test samples
        black_box_feasibility_ = [];white_box_feasibility_ = []
        for idx in y_rejects_due_to_perturbations:
            try:
                x_orig = X_test[idx,:]

                # Counterfactual by black-box method
                xcf_blackbox = cf_blackbox_method.compute_counterfactual_explanation(x_orig)
                if xcf_blackbox is None:
                    black_box_feasibility_.append(0)
                else:
                    black_box_feasibility_.append(1. / len(y_rejects_due_to_perturbations))

                # Counterfactual by our proposed convex program
                xcf = cf_method.compute_counterfactual_explanation(x_orig)
                if xcf is None or reject_option(xcf):  # Sanity check -- should not happen. It it happens, then because of numerical reasons!
                    white_box_feasibility_.append(0)
                    print("Sanity check: xcf is rejected")
                    continue
                else:
                    white_box_feasibility_.append(1. / len(y_rejects_due_to_perturbations))

                # Closest sample from the training set as counterfactual
                xcf_training_sample = closest_sample_counterfactual(X_train, y_rejects, x_orig)

                # Evaluation
                # Are perturbed features recovered?
                perturbed_features_recovery_counterfactual.append(evaluate_perturbed_features_recovery(xcf, x_orig, perturbed_features_idx))
                perturbed_features_recovery_closestsample.append(evaluate_perturbed_features_recovery(xcf_training_sample, x_orig, perturbed_features_idx))
                if xcf_blackbox is not None:
                    perturbed_features_recovery_blackbox.append(evaluate_perturbed_features_recovery(xcf_blackbox, x_orig, perturbed_features_idx))

                # Sparsity -- i.e. "complexity" of the explanation
                sparsity_closest_sample.append(evaluate_sparsity(xcf_training_sample, x_orig))
                sparsity_counterfactual.append(evaluate_sparsity(xcf, x_orig))
                if xcf_blackbox is not None:
                    sparsity_blackbox_counterfactual.append(evaluate_sparsity(xcf_blackbox, x_orig))

                # Feature overlap between different explanations
                overlap_closestsample_counterfactual.append(evaluate_featureoverlap(xcf, xcf_training_sample, x_orig))
                if xcf_blackbox is not None:
                    overlap_closestsample_blackbox.append(evaluate_featureoverlap(xcf_training_sample, xcf_blackbox, x_orig))
                    overlap_counterfactual_blackbox.append(evaluate_featureoverlap(xcf, xcf_blackbox, x_orig))
            except Exception as ex:
                print(ex)
        
        white_box_feasibility.append(np.sum(white_box_feasibility_))
        black_box_feasibility.append(np.sum(black_box_feasibility_))
    
    # Compute and export final statistics
    compute_export_results(black_box_feasibility, white_box_feasibility, sparsity_blackbox_counterfactual, sparsity_closest_sample, sparsity_counterfactual, overlap_closestsample_counterfactual, overlap_closestsample_blackbox, overlap_counterfactual_blackbox)
    compute_export_perturbed_features_recovery_results(perturbed_features_recovery_counterfactual, perturbed_features_recovery_closestsample, perturbed_features_recovery_blackbox)
