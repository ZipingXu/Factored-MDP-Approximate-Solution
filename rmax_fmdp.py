"""
Author  :    Ziping Xu
Email   :    zipingxu@umich.edu
Date    :    Dec 26, 2019
Record  :    f-Rmax
"""


import itertools
import math
from copy import deepcopy
import numpy as np
from fmdp import *
import random

def rmax_mdp(mdp, m, s):
  As = list(range(mdp.n_action))
  random.shuffle(As)
  for a in As:
    for f in range(mdp.n_factor_s):
      if mdp.total_visitations[f, mdp.state_index.find_idx(s, mdp.G[f][0]), mdp.action_index.find_idx(a, mdp.G[f][1])] <= m[f]:
        return [False, a]
  p_new, r_new = mdp.empirical_estimate()
  n_factor_s = mdp.n_factor_s
  n_factor_a = mdp.n_factor_a
  n_state = mdp.n_state
  n_action = mdp.n_action
  G = mdp.G
  mdp_k = fmdp(n_factor_s, n_factor_a, n_state, n_action, p_new, r_new, G)
  return [True, mdp_k]


def rmax(mdp, m = None, initial_state=None, show = 10000, use_fix = False, O = None, accurate = False):
  '''
  UCRL2 algorithm
  See _Near-optimal Regret Bounds for Reinforcement Learning_. Jaksch, Ortner, Auer. 2010.
  '''
  if m == None:
    m = np.array([100]*mdp.n_factor_s)
  n_factor_s, n_factor_a, n_state, n_action, G = mdp.n_factor_s, mdp.n_factor_a, mdp.n_state, mdp.n_action, mdp.G
  t = 1
  # Initial state
  st = mdp.reset()
  complete = False
  h_k = 0
  for k in itertools.count():
    # print(k)
    # Initialize episode k
    h_k += 1
    t_k = t
    #p_hat, r_hat = mdp.empirical_estimate()
    #mdp_k = po_extended_mdp(mdp,p_hat,r_hat,t_k)
    if not complete:
      ret = rmax_mdp(mdp,m,st)
      if ret[0]:
        complete = True
        mdp_k = ret[1]
        if O == None:
          O =  [(0, i) for i in range(mdp_k.n_factor_s)] + [(1, 0)]
        if accurate:
          mdp_k.FactoredALP(O, basis = "accurate")
        else:
          mdp_k.FactoredALP(O)
        mdp.opt.lp_problem.solve()
        ac = mdp_k.policy(st, extended = False)
      else:
        ac = ret[1]
    else:
      # Execute policy
      ac = mdp_k.policy(st, extended = False)

    st, ac, next_st, reward = mdp.step(np.array([ac]))
    yield t, st, ac, next_st, reward
    # Update statistics
    mdp.record(st, ac, next_st, reward)
    t += 1
    st = next_st
if __name__ == '__main__':
  mdp = generate()
  transitions = rmax(mdp, initial_state=0)
  tr = []
  for _ in range(10):
      (t, st, ac, next_st, r) = transitions.__next__()
      tr.append((t, st, ac, next_st, r))
