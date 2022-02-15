from abc import ABC, abstractmethod
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize

from reject_option import LvqRejectOption, LvqRejectProbabilistic


default_solver = cp.SCS
#default_solver = cp.MOSEK


class LvqRejectOptionCounterfactual():
    def __init__(self, reject_option_model, **kwds):
        if not isinstance(reject_option_model, LvqRejectOption):
            raise TypeError(f"'reject_option_model' must be an instance of 'LvqRejectOption' not of {type(reject_option_model)}")
        
        self.reject_option_model = reject_option_model
        self.lvq_model = reject_option_model.lvq_model
        self.reject_threshold = reject_option_model.threshold
        
        super().__init__(**kwds)


    @abstractmethod
    def compute_counterfactual_explanation(self, x_orig):
        raise NotImplementedError()


class LvqRejectOptionCounterfactualBlackBoxSolver(LvqRejectOptionCounterfactual):
    def __init__(self, **kwds):
        self.epsilon = 1e-2
        self.C = 1e3
        self.solver = "Nelder-Mead"

        super().__init__(**kwds)
    
    def cost_function(self, x):
        return min(self.reject_option_model.criterion(x) - self.reject_threshold - self.epsilon, 0)    # Assumin r(x) >= 0 forall x

    def compute_counterfactual_explanation(self, x_orig):
        # Build objective
        objective = lambda x: self.C * -1. * self.cost_function(x) + np.linalg.norm(x - x_orig, ord=1)

        # Minimize objective -- i.e. computing a counterfactual explanation
        res = minimize(objective, x_orig, method=self.solver)
        x_cf = res["x"]

        # Check "feasibility" -- i.e. is it a valid counterfactual explanation
        success = not self.reject_option_model(x_cf)
        if success is False:
            return None     # Return None if counterfactual is not valid!

        return x_cf


class LvqRejectOptionConvexProgram(ABC):
    def __init__(self, **kwds):
        self.epsilon = 1e-1
        self.solver = default_solver
        self.solver_verbosity = False

        super().__init__(**kwds)
    
    @abstractmethod
    def _build_constraints(self, var_x, other_prototypes, target_prototype, target_label):
        raise NotImplementedError()

    def _solve(self, prob):
        prob.solve(solver=self.solver, verbose=self.solver_verbosity)
    
    def _build_and_solve_opt(self, x_orig, other_prototypes, target_prototype, target_label, mad, objective_upper_bound=None, features_whitelist=None):
        dim = x_orig.shape[0]

        # Variables
        x = cp.Variable(dim)
        beta = cp.Variable(dim)

        # Build constraints
        constraints = self._build_constraints(x, other_prototypes, target_prototype, target_label)
        
        # If requested, freeze some features
        if features_whitelist is not None:
            A = []
            a = []

            for j in range(dim):
                if j not in features_whitelist:
                    t = np.zeros(dim)
                    t[j] = 1.
                    A.append(t)
                    a.append(x_orig[j])
            
            if len(A) != 0:
                A = np.array(A)
                a = np.array(a)

                constraints += [A @ x == a]

        # Build weight matrix for the weighted Manhattan distance
        Upsilon = np.diag(1. / mad)

        # Build final program
        f = cp.Minimize(cp.sum(beta))    # Minimize (weighted) Manhattan distance
        constraints += [Upsilon @ (x - x_orig) <= beta, (-1. * Upsilon) @ (x - x_orig) <= beta, beta >= 0]
        
        if objective_upper_bound:   # If given, put an upper bound on the objective -- can be used as some kind of early stopping by making the program infeasible if it clear that the solution can not be better than an already known solution
            constraints += [cp.sum(beta) <= objective_upper_bound]

        prob = cp.Problem(f, constraints)

        # Solve it
        self._solve(prob)

        delta = x_orig - x.value if x.value is not None else None
        return x.value, delta, f.value

    def _compute_counterfactual_explanation(self, x_orig, prototypes, prototypes_labels, mad=None, features_whitelist=None):
        xcf = None
        xcf_dist = float("inf")

        if mad is None:
            mad = np.ones(x_orig.shape[0])

        # Iterate over all possible (target) prototypes
        for prototype_idx in range(len(prototypes)):
            target_prototype = prototypes[prototype_idx]
            target_label = prototypes_labels[prototype_idx]
            other_prototypes = [prototypes[p_idx] for p_idx in filter(lambda i: prototypes_labels[i] != target_label, range(len(prototypes)))]

            xcf_, _, xcf_dist_ = self._build_and_solve_opt(x_orig, other_prototypes, target_prototype, target_label, mad, None if xcf_dist == float("inf") else xcf_dist, features_whitelist)
            if xcf_ is not None:
                xcf = xcf_
                xcf_dist = xcf_dist_
        
        #if xcf is None:
        #    raise Exception("Did not find a counterfactual.")

        return xcf


class LvqRejectOptionRelSimCounterfactual(LvqRejectOptionCounterfactual, LvqRejectOptionConvexProgram):
    def __init__(self, **kwds):
        super().__init__(**kwds)

    def _build_constraints(self, var_x, other_prototypes, target_prototype, target_label):
        constraints = []
        
        for p_j in other_prototypes:
            q = ((-1. / self.reject_threshold) - 1.) * self.lvq_model.dist_mat @ target_prototype + ((1. / self.reject_threshold) - 1.) * self.lvq_model.dist_mat @ p_j
            
            pj = p_j.T @ self.lvq_model.dist_mat @ p_j
            pi = target_prototype.T @ self.lvq_model.dist_mat @ target_prototype
            c = 0.5 * ((1. - (1. / self.reject_threshold)) * pj + (1. + (1. / self.reject_threshold)) * pi)

            constraints.append(cp.quad_form(var_x, self.lvq_model.dist_mat)+ q.T @ var_x + c + self.epsilon <= 0)

        return constraints

    def compute_counterfactual_explanation(self, x_orig, mad=None, features_whitelist=None):
        return LvqRejectOptionConvexProgram._compute_counterfactual_explanation(self, x_orig, self.lvq_model.prototypes, self.lvq_model.prototypes_labels, mad, features_whitelist)


class LvqRejectOptionDistDecisionBoundaryCounterfactual(LvqRejectOptionCounterfactual, LvqRejectOptionConvexProgram):
    def __init__(self, **kwds):
        super().__init__(**kwds)

    def _build_constraints(self, var_x, other_prototypes, target_prototype, target_label):
        constraints = []
        
        for p_j in other_prototypes:
            d = target_prototype - p_j
            c = p_j.T @ self.lvq_model.dist_mat @ p_j - target_prototype.T @ self.lvq_model.dist_mat @ target_prototype - 2. * self.reject_threshold * d.T @ self.lvq_model.dist_mat @ d

            q = 2. * self.lvq_model.dist_mat @ target_prototype - 2. * self.lvq_model.dist_mat @ p_j

            constraints.append(q.T @ var_x + c - self.epsilon >= 0)

        return constraints

    def compute_counterfactual_explanation(self, x_orig, mad=None, features_whitelist=None):
        return LvqRejectOptionConvexProgram._compute_counterfactual_explanation(self, x_orig, self.lvq_model.prototypes, self.lvq_model.prototypes_labels, mad, features_whitelist)


class LvqRejectOptionProbaCounterfactualBlackBox(LvqRejectOptionCounterfactual):
    def __init__(self, reject_option_model, **kwds):
        if not isinstance(reject_option_model, LvqRejectProbabilistic):
            raise TypeError(f"'reject_option_model' must be an instance of 'LvqRejectProbabilistic' not of {type(reject_option_model)}")

        self.epsilon = 1e-2
        self.C = 1e3
        self.solver = "Nelder-Mead"

        super().__init__(reject_option_model=reject_option_model,**kwds)

    def _cost_function(self, x, i):
        cost = 0

        for j in range(self.reject_option_model.num_classes):
            if j == i:
                continue

            # Get corresponding pari-wise classifier
            lvq_model_ij_data = self.reject_option_model.pairwise_wrapped_lvq_models[i][j]
            lvq_model_ij, alpha, beta = lvq_model_ij_data["model"], lvq_model_ij_data["alpha"], lvq_model_ij_data["beta"]

            # Compute closest prototypes (of both classes)
            distances_to_prototypes = [lvq_model_ij.dist(lvq_model_ij.prototypes[i], x) for i in range(len(lvq_model_ij.prototypes))]

            pi_idx = np.argmin(distances_to_prototypes)
            dp = distances_to_prototypes[pi_idx]
            pi_label = lvq_model_ij.prototypes_labels[pi_idx]

            other_prototypes_idx = np.where(lvq_model_ij.prototypes_labels != pi_label)[0]
            pj_idx = np.argmin([distances_to_prototypes[idx] for idx in other_prototypes_idx])
            dm = distances_to_prototypes[other_prototypes_idx[pj_idx]]

            r_ij = (dm - dp) / (dm + dp)
            cost +=  np.exp(alpha * r_ij + beta)

        cost += 1 - (1. / self.reject_option_model.threshold)

        return np.max(cost + self.epsilon, 0)   # Constraint: cost <= 0

    def compute_counterfactual_explanation(self, x_orig):
        xcf = None
        xcf_dist = float("inf")

        # Iterate over all possible classes
        for i in range(self.reject_option_model.num_classes):
            # Build objective
            objective = lambda x: self.C * self._cost_function(x, i) + np.linalg.norm(x - x_orig, ord=1)    # Apply penalty method to get rid of the constraints

            # Minimize objective -- i.e. computing a counterfactual explanation
            res = minimize(objective, x_orig, method=self.solver)
            x_cf = res["x"]

            # Check "feasibility" -- i.e. is it a valid counterfactual explanation
            success = not self.reject_option_model(x_cf)
            if success is True:
                x_cf_dist = np.linalg.norm(x_cf - x_orig, ord=1)
                if x_cf_dist < xcf_dist:
                    xcf = x_cf
                    xcf_dist = x_cf_dist

        return xcf


class LvqRejectOptionProbaCounterfactual(LvqRejectOptionCounterfactual):
    def __init__(self, reject_option_model, **kwds):
        if not isinstance(reject_option_model, LvqRejectProbabilistic):
            raise TypeError(f"'reject_option_model' must be an instance of 'LvqRejectProbabilistic' not of {type(reject_option_model)}")

        self.epsilon = 1e-2
        self.threshold_epsilon = .1
        self.solver = default_solver
        self.solver_verbosity = False

        super().__init__(reject_option_model=reject_option_model,**kwds)
    
    def _solve(self, prob):
        prob.solve(solver=self.solver, verbose=self.solver_verbosity)

    def _build_constraints(self, var_x, other_prototypes, target_prototype, target_label, alpha, beta, dist_mat):
        constraints = []
        
        c_prime = 1. - (1. / (self.reject_threshold - self.threshold_epsilon))
        gamma = (np.log(max(-1. * c_prime, self.epsilon)) - np.log(self.reject_option_model.num_classes - 1) - beta) / alpha

        for p_j in other_prototypes:
            q_j_T = (-2. * (gamma - 1.) * p_j.T @ dist_mat - 2. * (1. + gamma) * target_prototype.T @ dist_mat) / (-2. * gamma)
            c_j = ((gamma - 1.) * p_j.T @ dist_mat @ p_j + (1. + gamma) * target_prototype.T @ dist_mat @ target_prototype) / (-2. * gamma)

            constraints.append(cp.quad_form(var_x, dist_mat) + q_j_T @ var_x + c_j + self.epsilon <= 0)

        return constraints

    def __build_and_solve_opt(self, x_orig, other_prototypes, target_prototype, target_label, alpha, beta, dist_mat, mad, objective_upper_bound, features_whitelist):
        dim = x_orig.shape[0]

        # Variables
        x = cp.Variable(dim)
        var_beta = cp.Variable(dim)

        # Build constraints
        constraints = self._build_constraints(x, other_prototypes, target_prototype, target_label, alpha, beta, dist_mat)
        
        # If requested, freeze some features
        if features_whitelist is not None:
            A = []
            a = []

            for j in range(dim):
                if j not in features_whitelist:
                    t = np.zeros(dim)
                    t[j] = 1.
                    A.append(t)
                    a.append(x_orig[j])
            
            if len(A) != 0:
                A = np.array(A)
                a = np.array(a)

                constraints += [A @ x == a]

        # Build weight matrix for the weighted Manhattan distance
        Upsilon = np.diag(1. / mad)

        # Build final program
        f = cp.Minimize(cp.sum(var_beta))    # Minimize (weighted) Manhattan distance
        constraints += [Upsilon @ (x - x_orig) <= var_beta, (-1. * Upsilon) @ (x - x_orig) <= var_beta, var_beta >= 0]
        
        if objective_upper_bound:   # If given, put an upper bound on the objective -- can be used as some kind of early stopping by making the program infeasible if it is clear that the solution can not be better than an already known solution
            constraints += [cp.sum(var_beta) + self.epsilon <= objective_upper_bound]

        prob = cp.Problem(f, constraints)

        # Solve it
        self._solve(prob)

        delta = x_orig - x.value if x.value is not None else None
        return x.value, delta, f.value

    def compute_counterfactual_explanation(self, x_orig, mad=None, features_whitelist=None):
        xcf = None
        xcf_dist = float("inf")

        if mad is None:
            mad = np.ones(x_orig.shape[0])

        # For each label, for each pairwise classifier in this label, for each possible target prototype
        for y_target_idx in range(self.reject_option_model.num_classes):
            for other_target_class in range(self.reject_option_model.num_classes):
                if other_target_class == y_target_idx:
                    continue

                # Consider classifier: y_target_idx vs. other_target_class 
                pairwise_model_data = self.reject_option_model.pairwise_wrapped_lvq_models[y_target_idx][other_target_class]
                pairwise_model, alpha, beta = pairwise_model_data["model"], pairwise_model_data["alpha"], pairwise_model_data["beta"]

                # Compute counterfactual explanation under this model
                prototypes = pairwise_model.prototypes
                prototypes_labels = pairwise_model.prototypes_labels
                dist_mat = pairwise_model.dist_mat

                for prototype_idx in range(len(prototypes)):  # Iterate over all possible (target) prototypes
                    target_prototype = prototypes[prototype_idx]
                    target_label = prototypes_labels[prototype_idx]
                    other_prototypes = [prototypes[p_idx] for p_idx in filter(lambda i: prototypes_labels[i] != target_label, range(len(prototypes)))]

                    xcf_, _, xcf_dist_ = self.__build_and_solve_opt(x_orig, other_prototypes, target_prototype, target_label, alpha, beta, dist_mat, mad, None if xcf_dist == float("inf") else xcf_dist, features_whitelist)
                    if xcf_ is not None:
                        xcf = xcf_
                        xcf_dist = xcf_dist_

        return xcf
