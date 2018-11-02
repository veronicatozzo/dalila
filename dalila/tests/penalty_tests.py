from nose.tools import *
import numpy as np

from dalila.dl.penalty import *


@raises(ValueError)
def negative_parameter_l1_test():
    p = L1Penalty(-1)
    x = np.random.rand(100,100)
    p.apply_prox_operator(x, 0.1)


@raises(ValueError)
def negative_parameter_l2_test():
    p = L2Penalty(-1)
    x = np.random.rand(100,100)
    p.apply_prox_operator(x, 0.1)


@raises(ValueError)
def negative_parameter_elasticnet_test():
    p = ElasticNetPenalty(-1, -1, .5)
    x = np.random.rand(100,100)
    p.apply_prox_operator(x, 0.1)


@raises(ValueError)
def negative_parameter_l0_test():
    p = L0Penalty(-1)
    x = np.random.rand(100,100)
    p.apply_prox_operator(x, 0.1)


@raises(ValueError)
def negative_parameter_grouplasso_test():
    p = GroupLassoPenalty([np.arange(0, 50), np.arange(50, 100)], -1)
    x = np.random.rand(100,100)
    p.apply_prox_operator(x, 0.1)


@raises(ValueError)
def negative_parameter_linf_test():
    p = LInfPenalty(-1)
    x = np.random.rand(100,100)
    p.apply_prox_operator(x, 0.1)


@raises(ValueError)
def wrong_alpha1_elasticnet_test():
    p = ElasticNetPenalty(1, 1, -.5)
    x = np.random.rand(100,100)
    p.apply_prox_operator(x, 0.1)


@raises(ValueError)
def wrong_alpha2_elasticnet_test():
    p = ElasticNetPenalty(1, 1, 2)
    x = np.random.rand(100,100)
    p.apply_prox_operator(x, 0.1)


@raises(ValueError)
def too_high_s__l0_test():
    p = L0Penalty(10)
    x = np.random.rand(5,5)
    p.apply_prox_operator(x, 0.1)


@raises(ValueError)
def overlapping_groups_grouplasso_test():
    p = GroupLassoPenalty([np.arange(0, 55), np.arange(50, 100)], 1)
    x = np.random.rand(100,100)
    p.apply_prox_operator(x, 0.1)


@raises(ValueError)
def non_coverage_groups_grouplasso_test():
    p = GroupLassoPenalty([np.arange(0, 30), np.arange(50, 100)], 1)
    x = np.random.rand(100,100)
    p.apply_prox_operator(x, 0.1)
