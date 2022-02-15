from math import exp, log
import numpy as np


def train_sigmoid(out, target, prior1, prior0=None):
    # calculate A and B for the sigmoid
    # p_i = 1 / (1 + exp(A * f_i + B))
    #
    # INPUT:
    # out = array of outputs (SVM) or mu for LVQ
    # target = array of booleans: is i-th example a positive example?
    # prior1 = number of positive example
    # prior0 = number of negative example
    #
    #
    # OUTPUT:
    #
    # A, B = parameters of sigmoid

    if prior0 is None:
        posClass = prior1
        prior1 = np.sum(target == posClass)
        prior0 = np.sum(target != posClass)
        target = target == posClass

    maxiter = 100  # Maximum number of iterations
    minstep = 1e-10  # Minimum step taken in line search
    sigma = 1e-3  # For numerically strict PD of Hessian
    eps = 1e-5

    # Construct Target Support
    hiTarget = (prior1 + 1) / (prior1 + 2)
    loTarget = 1 / (prior0 + 2)
    n = len(out)
    t = np.zeros(n)
    for i in range(n):
        if target[i]:
            t[i] = hiTarget
        else:
            t[i] = loTarget

    # Initial Point and Initial Fun Value
    A = eps
    B = log((prior0 + 1) / (prior1 + 1))
    fval = 0

    for i in range(n):
        fApB = out[i] * A + B
        if fApB >= 0:
            fval += t[i] * fApB + log(1 + exp(-fApB))
        else:
            fval += (t[i] - 1) * fApB + log(1 + exp(fApB))

    for it in range(maxiter):
        h11 = sigma
        h22 = sigma
        h21 = 0
        g1 = 0
        g2 = 0
        for i in range(n):
            fApB = out[i] * A + B
            if fApB >= 0:
                p = exp(-fApB) / (1 + exp(-fApB))
                q = 1 / (1 + exp(-fApB))
            else:
                p = 1 / (1 + exp(fApB))
                q = exp(fApB) / (1 + exp(fApB))

            d2 = p * q
            h11 = h11 + out[i] * out[i] * d2
            h22 = h22 + d2
            h21 = h21 + out[i]*d2
            d1 = t[i] - p
            g1 = g1 + out[i] * d1
            g2 = g2 + d1

        if abs(g1)<eps and abs(g2)<eps:
            break
        # Finding Newton direction: -inv(H') * g
        det = h11 * h22 - h21 * h21
        dA = -(h22 * g1 - h21 * g2) / det
        dB = -(-h21 * g1 + h11 * g2) / det
        gd = g1 * dA + g2 * dB

        stepsize = 1
        while stepsize >= minstep:
            newA = A + stepsize * dA
            newB = B + stepsize * dB

            newf = 0
            for i in range(n):
                fApB = out[i] * newA + newB
                if fApB >= 0:
                    newf += t[i] * fApB + log(1 + exp(-fApB))
                else:
                    newf += (t[i] - 1) * fApB + log(1 + exp(fApB))

            if newf < (fval + 0.0001 * stepsize * gd):
                A = newA
                B = newB
                fval = newf
                break
            else:
                stepsize = stepsize / 2

        if stepsize < minstep:
            print(f"line search fails {A} {B} {g1} {g2} {dA} {dB}")
            return

    if it >= maxiter:
        print(f"reaching maximal iteration {g1} {g2}")

    return [A, B]

