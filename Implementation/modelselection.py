from itertools import count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from kneed import KneeLocator
from sklearn.metrics import auc
from sklearn.model_selection import (KFold, ParameterGrid,
                                     cross_val_score, train_test_split)

from counterfactual import (LvqRejectOptionCounterfactualBlackBoxSolver,
                            LvqRejectOptionProbaCounterfactual)
from lvq import LvqWrapper


def check_rejections(X_test, reject_option_model):
    # For each sample in the test set, check if it is rejected
    y_rejects = []
    for i in range(X_test.shape[0]):
        x = X_test[i,:]
        if reject_option_model(x):
            y_rejects.append(i)

    return y_rejects

def rate_of_lists(part, whole):
    return len(part) / len(whole)

def find_nearest(array, value):
    return (np.abs(array - value)).argmin()

def compute_threshold_knee_point(x, y, thresholds):
    x = np.array(x)
    y = np.array(y)
    thresholds = np.array(thresholds)

    # Filter duplicate x,y coordinates
    # Remove non duplicate rejection_rates
    # x, indices = np.unique(x, return_index=True)
    # y = y[indices]
    # thresholds = thresholds[indices]
    
    # remove non-increasing elements
    # indices = [0] + [idx for idx in range(1, len(x)) if y[idx] >= y[idx-1]]
    # x = x[indices]
    # y = y[indices]
    # thresholds = thresholds[indices]

    # compute knee point rejection rate
    kneedle = KneeLocator(x, y, S=1.0, curve="concave", online=True)

    if kneedle.knee is not None:
        # TODO find/approximate rejection_threshold for the knee point rejection rate
        idx_nearest = find_nearest(x, kneedle.knee)
        # kneedle.plot_knee()
        # plt.plot(kneedle.x_difference, kneedle.y_difference)
        # kneedle.plot_knee_normalized()
        return thresholds[idx_nearest]

    return None

def sort_increasing(rejection_rates, accuracies):
    rejection_rates = np.array(rejection_rates)
    accuracies = np.array(accuracies)
    
    indices = np.argsort(rejection_rates)

    rejection_rates = rejection_rates[indices]
    accuracies = accuracies[indices]
    return rejection_rates, accuracies

class ExplainableRejectOptionGridSearchCVSingleThreaded():

    def __init__(self, lvq_model, lvq_parameters, reject_option_model, rejection_thresholds, counterfactual_model, cv=5):
        # TODO type check lvq and reject model class
        self.lvq_model = lvq_model
        self.lvq_parameters = lvq_parameters
        self.reject_option_model = reject_option_model
        self.rejection_thresholds = rejection_thresholds
        self.counterfactual_model = counterfactual_model
        self.cv = cv

    def fit(self, X, y):
        self.X = X
        self.y = y

        # Generate folds
        self.folds = KFold(n_splits=self.cv)

        # Fit LVQ models
        self.lvq_models = self._generate_lvq_models()

        # Run rejection/counterfactual
        self._generate_rejection_models()

        return self.best_model_params()

    def _generate_lvq_models(self):
        X, y = self.X, self.y
        models = []
        for params in ParameterGrid(self.lvq_parameters):
            fold_scores = []
            fold_models = []
            for train, validation in self.folds.split(X):
                # X_train, y_train, X_validation, y_validation
                model = self.lvq_model(prototypes_per_class=params['prototypes_per_class'], random_state=444)
                model.fit(X[train], y[train])
                score = model.score(X[validation], y[validation])
                fold_scores.append(score)
                fold_models.append(model)

            avg_score = sum(fold_scores)/len(fold_scores)
            std_score = np.std(fold_scores)
            models.append({'lvq_params': params, 'fold_models': fold_models, 'lvq_accuracy_score': fold_scores, 'avg_accuracy_score': avg_score, 'std_accuracy_score': std_score})
        
        return models

    def _generate_rejection_models(self):
        X, y = self.X, self.y
        for idx, lvq_model in enumerate(self.lvq_models):
            fold_rejections = []
            fold_idx = 0
            fold_models = lvq_model['fold_models']
            for train, validation in self.folds.split(X):
                fold_model = fold_models[fold_idx]
                accuracies = [lvq_model['lvq_accuracy_score'][fold_idx]]
                rejection_rates = [0]

                X_train, y_train, X_validation, y_validation = X[train], y[train], X[validation], y[validation]
                for threshold in self.rejection_thresholds:
                    reject_option_model = self.reject_option_model(lvq_wrapped_model=LvqWrapper(fold_model), threshold=threshold)
                    reject_option_model.fit(X_train, y_train)   # Fit/Calibrate reject option
                    y_reject = check_rejections(X_validation, reject_option_model)

                    # Compute counterfactual explanations (for each fold/threshold)
                    # counterfactuals[threshold] = self._compute_counterfactuals(X_validation, y_reject, reject_option_model)

                    index = list(range(len(y_validation)))
                    index = [x for x in index if x not in y_reject]
                    if len(X_validation[index]) == 0:  # Stop if all elements are rejected for threshold
                        break

                    reject_rate = rate_of_lists(y_reject, X_validation)
                    rejection_rates.append(reject_rate)

                    # Compute accuracy of model
                    new_score = fold_model.score(X_validation[index], y_validation[index])
                    accuracies.append(new_score)
                
                if rejection_rates[-1] != 1:  # Add final point of ARC curve (by defintion of ARC)
                    accuracies.append(1)
                    rejection_rates.append(1)

                # Compute auc for ARC
                sorted_rates, sorted_accuracies = sort_increasing(rejection_rates, accuracies)
                au_arc_score = auc(sorted_rates, sorted_accuracies)

                rejection_model_outputs = {'accuracies': accuracies, 'rejection_rates': rejection_rates, 'au_arc_score': au_arc_score}
                fold_rejections.append(rejection_model_outputs)
                fold_idx += 1
            
            # Compute average area under ARC for lvq params
            average_au_arc = sum([x['au_arc_score'] for x in fold_rejections])/len(fold_rejections)
            std_au_arc = np.std([x['au_arc_score'] for x in fold_rejections])
            self.lvq_models[idx]['fold_rejection_outputs'] = fold_rejections
            self.lvq_models[idx]['avg_au_arc'] = average_au_arc
            self.lvq_models[idx]['std_au_arc'] = std_au_arc


    def _compute_counterfactuals(self, X_validation, y_reject, reject_option_model):
        cf_explanation = self.counterfactual_model(reject_option_model=reject_option_model)
        counterfactuals = []
        for idx in y_reject:
            x_orig = X_validation[idx,:]

            xcf = cf_explanation.compute_counterfactual_explanation(x_orig)
            counterfactuals.append((idx, xcf))
        return counterfactuals

    def fit_reject_model(self, reject_option_model, rejection_thresholds):
        if rejection_thresholds is not None:
            self.rejection_thresholds = rejection_thresholds

        self.reject_option_model = reject_option_model

        if self.lvq_models is None:
            self.fit()
        else:
            self._generate_rejection_models()

    def refit_counterfactual_model(self, counterfactual_model):
        raise NotImplementedError

    def best_model_params(self):
        # select best lvq model (by avg accuracy)
        df = self.results_df()
        idx_best_model = df['avg_accuracy_score'].idxmax()

        lvq_model_output = self.lvq_models[idx_best_model]
        lvq_params = lvq_model_output['lvq_params']

        # select best rejection threshold
        fold_rejection_output = lvq_model_output['fold_rejection_outputs']
        best_rejection_threshold_folds = [compute_threshold_knee_point(fold['rejection_rates'], fold['accuracies'], self.rejection_thresholds) for fold in fold_rejection_output]
        best_rejection_threshold_folds = list(filter(None, best_rejection_threshold_folds))
        best_rejection_threshold = np.mean(best_rejection_threshold_folds)  # Compute best threshold as mean of thresholds

        return {'lvq_params': lvq_params, 'rejection_threshold': best_rejection_threshold}

    def compute_inspect_counterfactuals(self, idx, rejection_thresholds=None, counterfactual_model_type=LvqRejectOptionCounterfactualBlackBoxSolver):
        """Computes counterfactuals for model (for all folds) at idx in lvq_models"""
        lvq_model = self.lvq_models[idx]
        
        if rejection_thresholds is None:
            rejection_thresholds = self.rejection_thresholds

        X, y = self.X, self.y

        fold_rejections = []
        fold_idx = 0
        fold_models = lvq_model['fold_models']
        for _, validation in self.folds.split(X):
            fold_model = fold_models[fold_idx]
            accuracies = [lvq_model['lvq_accuracy_score'][fold_idx]]
            rejection_rates = [0]
            counterfactual_output = {}

            X_validation, y_validation = X[validation], y[validation]
            for threshold in self.rejection_thresholds:
                reject_option_model = self.reject_option_model(lvq_wrapped_model=LvqWrapper(fold_model), threshold=threshold)
                reject_option_model.fit(X_validation, y_validation)   # Fit/Calibrate reject option
                y_reject = check_rejections(X_validation, reject_option_model)

                # Compute counterfactual explanations (for each fold/threshold)
                cf_explanation = counterfactual_model_type(reject_option_model=reject_option_model)
                counterfactuals = []
                for idx in y_reject:
                    x_orig = X_validation[idx,:]

                    xcf = cf_explanation.compute_counterfactual_explanation(x_orig)
                    counterfactuals.append((idx, xcf, x_orig))

                counterfactual_output[threshold] = counterfactuals

                index = list(range(len(y_validation)))
                index = [x for x in index if x not in y_reject]
                if len(X_validation[index]) == 0:  # Stop if all elements are rejected for threshold
                    break

                reject_rate = rate_of_lists(y_reject, X_validation)
                rejection_rates.append(reject_rate)

                # Compute accuracy of model
                new_score = fold_model.score(X_validation[index], y_validation[index])
                accuracies.append(new_score)
            
            if rejection_rates[-1] != 1:  # Add final point of ARC curve (by defintion of ARC)
                accuracies.append(1)
                rejection_rates.append(1)

            # Compute auc for ARC
            sorted_rates, sorted_accuracies = sort_increasing(rejection_rates, accuracies)
            au_arc_score = auc(sorted_rates, sorted_accuracies)

            rejection_model_outputs = {'fold_idx': fold_idx, 'accuracies': accuracies, 'rejection_rates': rejection_rates, 'counterfactual_output': counterfactual_output, 'au_arc_score': au_arc_score}
            fold_rejections.append(rejection_model_outputs)
            fold_idx += 1

        return fold_rejections

    def plot_arc_curve(self, lvq_model_idx=None, lvq_params=None):
        if lvq_model_idx is not None:
            model_output = self.lvq_models[lvq_model_idx]
        elif lvq_params is not None:
            raise NotImplementedError
        else:
            raise ValueError("Supply either index of lvq model or lvq hyperparameter setting")

        plt.figure()
        for idx, rejection_output in enumerate(model_output['fold_rejection_outputs']):
            plt.plot(
                rejection_output['rejection_rates'], 
                rejection_output['accuracies'],
                label="ARC of fold "+str(idx)
                )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Rejection Rate")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")

    def results_df(self):
        output_keys = ['lvq_params', 'avg_accuracy_score', 'std_accuracy_score', 'avg_au_arc', 'std_au_arc']
        filter_keys_from_dict = lambda x, keys: { key: x[key] for key in keys }
        model_scores = [filter_keys_from_dict(model_ouput, output_keys) for model_ouput in self.lvq_models]
        return pd.DataFrame(model_scores) 




#############
# Multiprocess Optimized version
#
#
def generate_lvq_model(lvq_model, params, X_train, y_train, X_validation, y_validation):
        model = lvq_model(prototypes_per_class=params['prototypes_per_class'], random_state=444)
        model.fit(X_train, y_train)
        score = model.score(X_validation, y_validation)
        return model, score

def reject_threshold_performance(reject_option_model, lvq_model, threshold, X_train, y_train, X_validation, y_validation):
    reject_option_model = reject_option_model(lvq_wrapped_model=LvqWrapper(lvq_model), threshold=threshold)
    reject_option_model.fit(X_train, y_train)   # Fit/Calibrate reject option
    y_reject = check_rejections(X_validation, reject_option_model)

    index = list(range(len(y_validation)))
    index = [x for x in index if x not in y_reject]
    if len(X_validation[index]) == 0:  # Stop if all elements are rejected for threshold
        return 

    reject_rate = rate_of_lists(y_reject, X_validation)

    # Compute accuracy of model
    new_score = lvq_model.score(X_validation[index], y_validation[index])
    return {'rejection_rate': reject_rate, 'reject_lvq_accuracy': new_score}

@ray.remote
def reject_lvq_cv_performance(trial_settings, lvq_parameters, X, y):
    """Ray Remote function used for multiprocessing/threading all gridsearch trials"""
    lvq_model = trial_settings.lvq_model
    reject_option_model = trial_settings.reject_option_model
    rejection_thresholds = trial_settings.rejection_thresholds 
    cv = trial_settings.cv

    # Generate folds
    folds = KFold(n_splits=cv)
    fold_accuracy_scores = []
    fold_models = []
    fold_rejections = []
    for train, validation in folds.split(X):
        X_train, y_train, X_validation, y_validation = X[train], y[train], X[validation], y[validation]
        model, lvq_accuracy_score = generate_lvq_model(lvq_model, lvq_parameters, X_train, y_train, X_validation, y_validation)
        fold_accuracy_scores.append(lvq_accuracy_score)
        fold_models.append(model)

        # compute rejects
        accuracies = [lvq_accuracy_score]
        rejection_rates = [0]
        rejection_thresholds_performance = [reject_threshold_performance(reject_option_model, model, threshold, X_train, y_train, X_validation, y_validation) for threshold in rejection_thresholds]
        
        for x in rejection_thresholds_performance:
            if x is not None:
                rejection_rates.append(x['rejection_rate'])
                accuracies.append(x['reject_lvq_accuracy'])

        if rejection_rates[-1] != 1:  # Add final point of ARC curve (by defintion of ARC)
            accuracies.append(1)
            rejection_rates.append(1)

        # Compute auc for ARC
        sorted_rates, sorted_accuracies = sort_increasing(rejection_rates, accuracies)
        au_arc_score = auc(sorted_rates, sorted_accuracies)
        
        rejection_model_outputs = {'accuracies': accuracies, 'rejection_rates': rejection_rates, 'au_arc_score': au_arc_score}
        fold_rejections.append(rejection_model_outputs)


    average_au_arc = sum([x['au_arc_score'] for x in fold_rejections])/len(fold_rejections)
    std_au_arc = np.std([x['au_arc_score'] for x in fold_rejections])
    
    avg_score = sum(fold_accuracy_scores)/len(fold_accuracy_scores)
    std_score = np.std(fold_accuracy_scores)

    return {'lvq_params': lvq_parameters, 
            'fold_models': fold_models, 
            'lvq_accuracy_score': fold_accuracy_scores, 
            'avg_accuracy_score': avg_score, 
            'std_accuracy_score': std_score,
            'fold_rejection_outputs': fold_rejections,
            'avg_au_arc': average_au_arc,
            'std_au_arc': std_au_arc}


class ExplainableRejectOptionGridSearchCV():
    def __init__(self, lvq_model, lvq_parameters, reject_option_model, rejection_thresholds, cv=5):
        self.lvq_model = lvq_model
        self.lvq_parameters_grid = lvq_parameters
        self.reject_option_model = reject_option_model
        self.rejection_thresholds = rejection_thresholds
        self.cv = cv

        ray.shutdown()
        ray.init()

    def fit(self, X, y):
        # self.X, self.y = X, y
        self.lvq_models = self._generate_lvq_models(X, y)  # Fit LVQ and rejection models
        return self.best_model_params()  # Return best score

    def _generate_lvq_models(self, X, y):
        # Load datasets into ray shared memory store (optimized for numpy arrays, works best for large datasets)
        X_id = ray.put(X)
        y_id = ray.put(y)
        
        models = []
        for params in ParameterGrid(self.lvq_parameters_grid):
            models.append(reject_lvq_cv_performance.remote(self, params, X_id, y_id))
        
        model_ouputs = ray.get(models) # Evaluate remote functions (i.e. trials with different parameterizations)
        return model_ouputs

    def best_model_params(self):
        # select best lvq model (by avg accuracy)
        df = self.results_df()
        idx_best_model = df['avg_accuracy_score'].idxmax()

        lvq_model_output = self.lvq_models[idx_best_model]
        lvq_params = lvq_model_output['lvq_params']

        # select best rejection threshold
        fold_rejection_output = lvq_model_output['fold_rejection_outputs']
        best_rejection_threshold_folds = [compute_threshold_knee_point(fold['rejection_rates'], fold['accuracies'], self.rejection_thresholds) for fold in fold_rejection_output]
        best_rejection_threshold_folds = list(filter(None, best_rejection_threshold_folds))
        best_rejection_threshold = np.mean(best_rejection_threshold_folds)  # Compute best threshold as mean of thresholds

        return {'lvq_params': lvq_params, 'rejection_threshold': best_rejection_threshold}

    def plot_arc_curve(self, lvq_model_idx=None, lvq_params=None):
        if lvq_model_idx is not None:
            model_output = self.lvq_models[lvq_model_idx]
        elif lvq_params is not None:
            raise NotImplementedError
        else:
            raise ValueError("Supply either index of lvq model or lvq hyperparameter setting")

        plt.figure()
        for idx, rejection_output in enumerate(model_output['fold_rejection_outputs']):
            plt.plot(
                rejection_output['rejection_rates'], 
                rejection_output['accuracies'],
                label="ARC of fold "+str(idx)
                )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Rejection Rate")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")

    def results_df(self):
        output_keys = ['lvq_params', 'avg_accuracy_score', 'std_accuracy_score', 'avg_au_arc', 'std_au_arc']
        filter_keys_from_dict = lambda x, keys: { key: x[key] for key in keys }
        model_scores = [filter_keys_from_dict(model_ouput, output_keys) for model_ouput in self.lvq_models]
        return pd.DataFrame(model_scores) 
