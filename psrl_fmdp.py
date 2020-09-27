"""
Author  :    Ziping Xu
Email   :    zipingxu@umich.edu
Date    :    Dec 18, 2019
Record  :    A posterior sampling implementation for factored MDP
"""

import itertools
import math
from copy import deepcopy
import numpy as np
from fmdp import *

def ps_mdp(mdp, p_hat, r_hat, scale):
  p_new = []
  r_new = []
  n_factor_s = mdp.n_factor_s
  n_factor_a = mdp.n_factor_a
  n_state = mdp.n_state
  n_action = mdp.n_action
  G = mdp.G
  ns = max([len(G[i][0]) for i in range(n_factor_s)])
  na = max([len(G[i][1]) for i in range(n_factor_s)])
  p_new = np.zeros((n_factor_s, n_state**ns, n_action**na, n_state))
  r_new = np.zeros((n_factor_s, n_state**ns, n_action**na))
  a_idx = index_machine(n_factor_a, n_action)

  # mdp.total_visitations = mdp.total_visitations * 100

  for ft in range(n_factor_s):
    S = mdp.state_index.enumerate(G[ft][0])
    A = mdp.action_index.enumerate(G[ft][1])
    for xt in range(len(S)):
      for at in range(len(A)):
        p_new[ft, xt, at, :] = np.random.dirichlet((mdp.total_transitions[ft, xt, at, :] + 1.0)*scale)
        r_new[ft, xt, at] = r_hat[ft, xt, at] + np.random.normal(0, 1.0/(mdp.total_visitations[ft, xt, at] + 1)/scale)
  return fmdp(n_factor_s, n_factor_a, n_state, n_action, p_new, r_new, G, init_dist = None)

def psrl(mdp, scale = 1, initial_state=None, show = 10000, use_fix = False, O = None, accurate = False):
  '''
  UCRL2 algorithm
  See _Near-optimal Regret Bounds for Reinforcement Learning_. Jaksch, Ortner, Auer. 2010.
  '''
  n_factor_s, n_factor_a, n_state, n_action, G = mdp.n_factor_s, mdp.n_factor_a, mdp.n_state, mdp.n_action, mdp.G
  t = 1
  # Initial state
  st = mdp.reset()

  h_k = 0
  for k in itertools.count():
    # print(k)
    # Initialize episode k
    h_k += 1
    t_k = t
    p_hat, r_hat = mdp.empirical_estimate()
    mdp_k = ps_mdp(mdp,p_hat,r_hat, scale)
    flatten = lambda l: list([item for sublist in l for item in sublist])
    if O == None:
      O = flatten([[(0, i), (1, i)] for i in range(n_factor_s)]) + [(1, n_factor_s)]
    if accurate:
      mdp_k.FactoredALP(O, basis = "accurate")
    else:
      mdp_k.FactoredALP(O)
    mdp_k.opt.lp_problem.solve()
    # print(mdp_k.opt.obj.value())
    # print([tmp.value() for tmp in mdp_k.opt.w])
    # print(pi_k, mdp_k)

    # Execute policy
    ac = mdp_k.policy(st, extended = False)
    # End episode when we visit one of the state-action pairs "often enough"
    vi = deepcopy(mdp.total_visitations)
    while True:

      st, ac, next_st, reward = mdp.step(np.array([ac]))
      yield t, st, ac, next_st, reward
      # Update statistics
      mdp.record(st, ac, next_st, reward)

      end = False
      for i in range(mdp.n_factor_s):
        id_s = mdp.state_index.find_idx(st, mdp.G[i][0])
        id_a = mdp.action_index.find_idx(ac, mdp.G[i][1])
        if mdp.total_visitations[i, id_s, id_a] > 2 * vi[i, id_s, id_a]:
          end = True
          break
      if use_fix == False:
        if end or t_k + h_k < t:
          break
      else:
        if t_k + h_k < t:
          break
      t += 1
      st = next_st
      ac = mdp_k.policy(st, extended = False)
if __name__ == '__main__':
  mdp = generate()
  transitions = psrl(mdp, initial_state=0)
  tr = []
  for _ in range(10):
      (t, st, ac, next_st, r) = transitions.__next__()
      tr.append((t, st, ac, next_st, r))
