# Copyright 2017 the GPflow authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import six
import tensorflow as tf
from collections import namedtuple
from numpy.testing import assert_allclose

import gpflow
from gpflow.features import InducingPointsBase
from gpflow.likelihoods import (Likelihood, Bernoulli, Exponential, Gamma, Gaussian, MultiClass,
                                Ordinal, Poisson, Softmax, StudentT, RobustMax, Beta, GaussianMC,
                                SwitchedLikelihood)
from gpflow.expectations import variationals
from gpflow import settings
# from gpflow.test_util import:
from gpflow.quadrature import ndiagquad

tf.random.set_seed(99012)
#
# class LikelihoodSetup(object):
#     def __init__(self, likelihood, Y, tolerance):
#         self.likelihood, self.Y, self.tolerance = likelihood, Y, tolerance
#         self.is_analytic = six.get_unbound_function(likelihood.predict_density) is not \
#                            six.get_unbound_function(gpflow.likelihoods.Likelihood.predict_density)

def is_analytic(likelihood):
    method = likelihood.__class__.predict_density
    return method is Likelihood.predict_density

class Datum:
    tolerance = 1e-06
    Yshape = (10, 2)
    Y = tf.random.normal(Yshape, dtype=tf.float64)
    F = tf.random.normal(Yshape, dtype=tf.float64)
    Fmu = tf.random.normal(Yshape, dtype=tf.float64)
    Fvar = tf.random.normal(Yshape, dtype=tf.float64)
    Fvar_zero = tf.zeros(Yshape, dtype=tf.float64)

    def square(x: tf.Tensor) -> tf.Tensor:
        return tf.square(x)

# class LikelihoodSetup(object):
#     def __init__(self, likelihood, Y=Datum.Y, rtol=1e-06, atol=0.):
#         self.likelihood = likelihood
#         self.Y = Y
#         self.rtol = rtol
#         self.atol = atol



LikelihoodSetup = namedtuple('LikelihoodSetup', ('likelihood', 'Y', 'rtol', 'atol'))
LikelihoodSetup.__new__.__defaults__ =(None, Datum.Y, 1e-6, 0.)

likelihood_setups = [
    LikelihoodSetup(Gaussian()),
    LikelihoodSetup(StudentT()),
    LikelihoodSetup(Beta()),
    LikelihoodSetup(MultiClass(2), Y=tf.argmax(Datum.Y, 1).numpy().reshape(-1, 1),
                    rtol=1e-3, atol=1e-3),
    LikelihoodSetup(Ordinal(np.array([-1, 1])), Y=np.random.randint(0, 3, Datum.Yshape)),
    LikelihoodSetup(Poisson(invlink=Datum.square)),
    LikelihoodSetup(Exponential(invlink=Datum.square)),
    LikelihoodSetup(Gamma(invlink=Datum.square)),
    LikelihoodSetup(Bernoulli(invlink=tf.sigmoid)),
]

analytic_likelihood_setups = [l for l in likelihood_setups if is_analytic(l.likelihood)]


@pytest.mark.parametrize('ls', likelihood_setups)
@pytest.mark.parametrize('mu, var', [[Datum.Fmu, tf.zeros_like(Datum.Fmu)]])
def test_conditional_mean_and_variance(ls, mu, var):
    """
    Here we make sure that the conditional_mean and conditional_var functions
    give the same result as the predict_mean_and_var function if the prediction
    has no uncertainty.
    """
    mu1 = ls.likelihood.conditional_mean(mu)
    var1 = ls.likelihood.conditional_variance(mu)
    mu2, var2 = ls.likelihood.predict_mean_and_var(mu, var)
    assert_allclose(mu1, mu2, rtol=ls.rtol, atol=ls.atol)
    assert_allclose(var1, var2, rtol=ls.rtol, atol=ls.atol)

#
@pytest.mark.parametrize('ls', likelihood_setups)
def test_variational_expectations(ls):
    """
    Here we make sure that the variational_expectations gives the same result
    as log_prob if the latent function has no uncertainty.
    """
    likelihood = ls.likelihood
    F = Datum.F
    Y = ls.Y
    r1 = likelihood.logp(F, Y)
    r2 = likelihood.variational_expectations(F, tf.zeros_like(F), Y)
    assert_allclose(r1, r2, atol=ls.atol, rtol=ls.rtol)


@pytest.mark.parametrize('ls', analytic_likelihood_setups)
@pytest.mark.parametrize('mu, var', [[Datum.Fmu, 0.01 * (Datum.Fvar ** 2)]])
def test_quadrature_variational_expectation(ls, mu, var):
    """
    Where quadrature methods have been overwritten, make sure the new code
    does something close to the quadrature.
    """
    likelihood, y  = ls.likelihood, ls.Y
    F1 = likelihood.variational_expectations(mu, var, y)
    F2 = ndiagquad(likelihood.logp, likelihood.num_gauss_hermite_points, mu, var, Y=y)
    assert_allclose(F1, F2, rtol=ls.rtol, atol=ls.atol)


@pytest.mark.parametrize('ls', analytic_likelihood_setups)
@pytest.mark.parametrize('mu, var', [[Datum.Fmu, 0.01 * (Datum.Fvar ** 2)]])
def test_quadrature_predict_density(ls, mu, var):
    likelihood, y  = ls.likelihood, ls.Y
    F1 = likelihood.predict_density(mu, var, y)
    F2 = Likelihood.predict_density(likelihood, mu, var, y)
    assert_allclose(F1, F2, rtol=ls.rtol, atol=ls.atol)


@pytest.mark.parametrize('ls', analytic_likelihood_setups)
@pytest.mark.parametrize('mu, var', [[Datum.Fmu, 0.01 * (Datum.Fvar ** 2)]])
def test_quadrature_mean_and_var(ls, mu, var):
    likelihood = ls.likelihood
    F1m, F1v = likelihood.predict_mean_and_var(mu, var)
    F2m, F2v = Likelihood.predict_mean_and_var(likelihood, mu, var)
    assert_allclose(F1m, F2m, rtol=ls.rtol, atol=ls.atol)
    assert_allclose(F1v, F2v, rtol=ls.rtol, atol=ls.atol)


@pytest.mark.parametrize('lik_variance', [0.3, 0.5, 1])
@pytest.mark.parametrize('mu, var, y', [[tf.random.normal((3, 10), dtype=tf.float64)] * 3])
def test_montecarlo_predict_mean_and_var(lik_variance, mu, var, y):
    likelihood = GaussianMC(lik_variance)
    likelihood2 = Gaussian(lik_variance)
    var = 0.01 * (var ** 2)
    likelihood.num_monte_carlo_points = 1000000
    F1 = likelihood.variational_expectations(mu, var, y)
    F2 = likelihood2.variational_expectations(mu, var, y)
    assert_allclose(F1, F2, rtol=5e-4, atol=1e-4)


@pytest.mark.parametrize('lik_variance', [0.3, 0.5, 1.])
@pytest.mark.parametrize('mu, var, y', [[tf.random.normal((3, 10), dtype=tf.float64)] * 3])
def test_montecarlo_predict_mean_and_var(lik_variance, mu, var, y):
    likelihood = GaussianMC(lik_variance)
    likelihood2 = Gaussian(lik_variance)
    var = 0.01 * (var ** 2)
    likelihood.num_monte_carlo_points = 1000000
    F1 = likelihood.predict_density(mu, var, y)
    F2 = likelihood2.predict_density(mu, var, y)
    assert_allclose(F1, F2, rtol=5e-4, atol=1e-4)


@pytest.mark.parametrize('lik_variance', [0.3, 0.5, 1.])
@pytest.mark.parametrize('mu, var, y', [[tf.random.normal((3, 10), dtype=tf.float64)] * 3])
def test_montecarlo_predict_mean_and_var(lik_variance, mu, var, y):
    likelihood = GaussianMC(lik_variance)
    likelihood2 = Gaussian(lik_variance)
    var = 0.01 * (var ** 2)
    likelihood.num_monte_carlo_points = 1000000
    F1mu, F1var = likelihood.predict_mean_and_var(mu, var)
    F2mu, F2var = likelihood2.predict_mean_and_var(mu, var)
    assert_allclose(F1mu, F2mu, rtol=5e-4, atol=1e-4)
    assert_allclose(F1var, F2var, rtol=5e-4, atol=1e-4)

@pytest.mark.parametrize('num, dimF', [[10, 5], [3, 2]])
@pytest.mark.parametrize('dimY', [10, 2, 1])
def test_softmax_y_shape_assert(num, dimF, dimY):
    """
    SoftMax assumes the class is given as a label (not, e.g., one-hot
    encoded), and hence just uses the first column of Y. To prevent
    silent errors, there is a tf assertion that ensures Y only has one
    dimension. This test checks that this assert works as intended.
    """
    F = tf.random.normal((num, dimF))
    dY = np.vstack((np.random.randn(num - 3, dimY), np.ones((3, dimY)))) > 0
    Y = tf.convert_to_tensor(dY, dtype=settings.int_type)
    likelihood = Softmax(dimF)
    try:
        likelihood.logp(F, Y)
    except tf.errors.InvalidArgumentError as e:
        assert "Condition x == y did not hold." in e.message


@pytest.mark.parametrize('num', [10, 3])
@pytest.mark.parametrize('dimF, dimY', [[2, 1]])
def test_softmax_bernoulli_equivalence(num, dimF, dimY):
    dF = np.vstack((np.random.randn(num - 3, dimF), np.array([[-3., 0.], [3, 0.], [0., 0.]])))
    dY = np.vstack((np.random.randn(num - 3, dimY), np.ones((3, dimY)))) > 0
    F = tf.convert_to_tensor(dF, dtype=settings.float_type)
    Fvar = tf.exp(tf.stack([F[:, 1], -10.0 + tf.zeros(F.shape[0], dtype=F.dtype)], axis=1))
    F = tf.stack([F[:, 0], tf.zeros(F.shape[0], dtype=F.dtype)], axis=1)
    Y = tf.convert_to_tensor(dY, dtype=settings.int_type)
    Ylabel = 1 - Y

    ls = Softmax(dimF)
    lb = Bernoulli(invlink=tf.sigmoid)
    ls.num_monte_carlo_points = 10000000
    lb.num_gauss_hermite_points = 50

    ls_cm = ls.conditional_mean(F)[:, :1]
    lb_cm = lb.conditional_mean(F[:, :1])
    ls_cv = ls.conditional_variance(F)[:, :1]
    lb_cv = lb.conditional_variance(F[:, :1])
    ls_lp = ls.logp(F, Ylabel)
    lb_lp = lb.logp(F[:, :1], Y.numpy())

    assert_allclose(ls_cm, lb_cm)
    assert_allclose(ls_cv, lb_cv)
    assert_allclose(ls_lp, lb_lp)

    ls_pm, ls_pv = ls.predict_mean_and_var(F, Fvar)
    lb_pm, lb_pv = lb.predict_mean_and_var(F[:, :1], Fvar[:, :1])

    assert_allclose(ls_pm[:, 0, None], lb_pm, rtol=1e-3)
    assert_allclose(ls_pv[:, 0, None], lb_pv, rtol=1e-3)

    ls_ve = ls.variational_expectations(F, Fvar, Ylabel)
    lb_ve = lb.variational_expectations(F[:, :1], Fvar[:, :1], Y.numpy())
    assert_allclose(ls_ve[:, 0, None], lb_ve, rtol=1e-3)


@pytest.mark.parametrize('num_classes, num_points', [[10, 3]])
@pytest.mark.parametrize('tol, epsilon', [[1e-4, 1e-3]])
def test_robust_max_multiclass_symmetric(num_classes, num_points, tol, epsilon):
    """
    This test is based on the observation that for
    symmetric inputs the class predictions must have equal probability.
    """
    rng = np.random.RandomState(1)
    p = 1. / num_classes
    F = tf.ones((num_points, num_classes), dtype=settings.float_type)
    Y = tf.convert_to_tensor(rng.randint(num_classes, size=(num_points, 1)),
                             dtype=settings.float_type)

    likelihood = MultiClass(num_classes)
    likelihood.invlink.epsilon = tf.convert_to_tensor(epsilon, dtype=settings.float_type)

    mu, _ = likelihood.predict_mean_and_var(F, F)
    pred = likelihood.predict_density(F, F, Y)
    variational_expectations = likelihood.variational_expectations(F, F, Y)

    expected_mu = (p * (1. - epsilon) +
                   (1. - p) * epsilon / (num_classes - 1)) * np.ones((num_points, 1))
    expected_log_density = np.log(expected_mu)

    # assert_allclose() would complain about shape mismatch
    assert (np.allclose(mu, expected_mu, tol, tol))
    assert (np.allclose(pred, expected_log_density, 1e-3, 1e-3))
    validation_variational_expectation = (p * np.log(1. - epsilon) +
                                         (1. - p) * np.log(epsilon / (num_classes - 1)))
    assert_allclose(variational_expectations,
                    np.ones((num_points, 1)) * validation_variational_expectation,
                    tol, tol)


@pytest.mark.parametrize('num_classes, num_points', [[5, 100]])
@pytest.mark.parametrize('mock_prob, expected_prediction, tol, epsilon',[
    [0.73, -0.5499780059, 1e-4, 0.231]
    # expected prediction evaluated on calculator:
    # log((1-\epsilon) * 0.73 + (1-0.73) * \epsilon/(num_classes -1))
])
def test_robust_max_multiclass_predict_density(
        num_classes, num_points, mock_prob, expected_prediction, tol, epsilon):
    class MockRobustMax(gpflow.likelihoods.RobustMax):
        def prob_is_largest(self, Y, Fmu, Fvar, gh_x, gh_w):
            return tf.ones((num_points, 1), dtype=settings.float_type) * mock_prob

    likelihood = MultiClass(num_classes, invlink=MockRobustMax(num_classes, epsilon))
    F = tf.ones((num_points, num_classes))
    rng = np.random.RandomState(1)
    Y = tf.convert_to_tensor(rng.randint(num_classes, size=(num_points, 1)),
                             dtype=settings.int_type)
    pred = likelihood.predict_density(F, F, Y)

    assert_allclose(pred, expected_prediction, tol, tol)


@pytest.mark.parametrize('num_classes', [5, 100])
@pytest.mark.parametrize('initial_epsilon, new_epsilon', [[1e-3, 0.412]])
def test_robust_max_multiclass_eps_K1_changes(num_classes, initial_epsilon, new_epsilon):
    """
    Checks that eps K1 changes when epsilon changes. This used to not happen and had to be
    manually changed.
    """
    likelihood = RobustMax(num_classes, initial_epsilon)
    expectedeps_K1 = initial_epsilon / (num_classes - 1.)
    actualeps_K1 = likelihood.eps_K1
    assert_allclose(expectedeps_K1, actualeps_K1)

    likelihood.epsilon = tf.convert_to_tensor(new_epsilon, dtype=settings.float_type)
    expected_eps_k2 = new_epsilon / (num_classes - 1.)
    actual_eps_k2 = likelihood.eps_K1
    assert_allclose(expected_eps_k2, actual_eps_k2)


@pytest.mark.parametrize('Y_list', [[tf.random.normal((i, 2)) for i in range(3,6)]])
@pytest.mark.parametrize('F_list', [[tf.random.normal((i, 2)) for i in range(3,6)]])
@pytest.mark.parametrize('Fvar_list', [[tf.exp(tf.random.normal((i, 2))) for i in range(3,6)]])
@pytest.mark.parametrize('Y_label', [[tf.ones((i, 2)) * (i - 3.)  for i in range(3,6)]])
def test_switched_likelihood_log_prob(Y_list, F_list, Fvar_list, Y_label):
    """
    SwitchedLikelihood is separately tested here.
    Here, we make sure the partition-stitch works fine.
    """
    Y_perm = list(range(3 + 4 + 5))
    np.random.shuffle(Y_perm)
    # shuffle the original data
    Y_sw = np.hstack([np.concatenate(Y_list), np.concatenate(Y_label)])[Y_perm, :3]
    F_sw = np.concatenate(F_list)[Y_perm, :]
    likelihoods = [Gaussian()] * 3
    for lik in likelihoods:
        lik.variance = np.exp(np.random.randn(1)).squeeze().astype(np.float32)
    switched_likelihood = SwitchedLikelihood(likelihoods)

    switched_results = switched_likelihood.logp(F_sw, Y_sw)
    results = [lik.logp(f, y) for lik, y, f in zip(likelihoods, Y_list, F_list)]

    assert_allclose(switched_results, np.concatenate(results)[Y_perm, :])


@pytest.mark.parametrize('Y_list', [[tf.random.normal((i, 2)) for i in range(3, 6)]])
@pytest.mark.parametrize('F_list', [[tf.random.normal((i, 2)) for i in range(3, 6)]])
@pytest.mark.parametrize('Fvar_list', [[tf.exp(tf.random.normal((i, 2))) for i in range(3, 6)]])
@pytest.mark.parametrize('Y_label', [[tf.ones((i, 2)) * (i - 3.) for i in range(3, 6)]])
def test_switched_likelihood_predict_density(Y_list, F_list, Fvar_list, Y_label):
    Y_perm = list(range(3 + 4 + 5))
    np.random.shuffle(Y_perm)
    # shuffle the original data
    Y_sw = np.hstack([np.concatenate(Y_list), np.concatenate(Y_label)])[Y_perm, :3]
    F_sw = np.concatenate(F_list)[Y_perm, :]
    Fvar_sw = np.concatenate(Fvar_list)[Y_perm, :]

    likelihoods = [Gaussian()] * 3
    for lik in likelihoods:
        lik.variance = np.exp(np.random.randn(1)).squeeze().astype(np.float32)
    switched_likelihood = SwitchedLikelihood(likelihoods)

    switched_results = switched_likelihood.predict_density(F_sw, Fvar_sw, Y_sw)
    # likelihood
    results = [lik.predict_density(f, fvar, y) for lik, y, f, fvar in
               zip(likelihoods, Y_list, F_list, Fvar_list)]
    assert_allclose(switched_results, np.concatenate(results)[Y_perm, :])


@pytest.mark.parametrize('Y_list', [[tf.random.normal((i, 2)) for i in range(3, 6)]])
@pytest.mark.parametrize('F_list', [[tf.random.normal((i, 2)) for i in range(3, 6)]])
@pytest.mark.parametrize('Fvar_list', [[tf.exp(tf.random.normal((i, 2))) for i in range(3, 6)]])
@pytest.mark.parametrize('Y_label', [[tf.ones((i, 2)) * (i - 3.) for i in range(3, 6)]])
def test__switched_likelihood_variational_expectations(Y_list, F_list, Fvar_list, Y_label):
    Y_perm = list(range(3 + 4 + 5))
    np.random.shuffle(Y_perm)
    # shuffle the original data
    Y_sw = np.hstack([np.concatenate(Y_list), np.concatenate(Y_label)])[Y_perm, :3]
    F_sw = np.concatenate(F_list)[Y_perm, :]
    Fvar_sw = np.concatenate(Fvar_list)[Y_perm, :]

    likelihoods = [Gaussian()] * 3
    for lik in likelihoods:
        lik.variance = np.exp(np.random.randn(1)).squeeze().astype(np.float32)
    switched_likelihood = SwitchedLikelihood(likelihoods)

    switched_results = switched_likelihood.variational_expectations(
        F_sw, Fvar_sw, Y_sw)
    results = [lik.variational_expectations(f, fvar, y) for lik, y, f, fvar in
               zip(likelihoods, Y_list, F_list, Fvar_list)]
    assert_allclose(switched_results, np.concatenate(results)[Y_perm, :])


@pytest.mark.parametrize('num_latent', [1, 2])
@pytest.mark.parametrize('X, Y', [[
    np.random.randn(100, 1), np.hstack((np.random.randn(100, 1), np.random.randint(0, 3, (100, 1))))
]])
def test_switched_likelihood_regression_valid_num_latent(X, Y, num_latent):
    """
    A Regression test when using Switched likelihood: the number of latent
    functions in a GP model must be equal to the number of columns in Y minus
    one. The final column of Y is used to index the switch. If the number of
    latent functions does not match, an exception will be raised.
    """

    Z = InducingPointsBase(Z=np.random.randn(num_latent, 1))
    likelihoods = [StudentT()] * 3
    switched_likelihood = SwitchedLikelihood(likelihoods)
    m = gpflow.models.SVGP(kernel=gpflow.kernels.Matern12(1),
                           feature=Z,
                           likelihood=switched_likelihood,
                           num_latent=num_latent)
    if num_latent == 1:
        m.log_likelihood(X, Y)
    else:
        with pytest.raises(ValueError):
            m.log_likelihood(X, Y)

if __name__ == "__main__":
    tf.test.main()
