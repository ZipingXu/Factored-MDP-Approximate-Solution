"""
Author  :    Ziping Xu
Email   :    zipingxu@umich.edu
Date    :    Dec 18, 2019
Record  :    Implementation for DORL
"""

import itertools
import math
from copy import deepcopy
import numpy as np
from fmdp import *

def do_extended_mdp(mdp, p_hat, r_hat, rho, t_k, alpha = 18):
  p_new = []
  r_new = []
  n_factor_s = mdp.n_factor_s
  n_factor_a = mdp.n_factor_a
  n_state = mdp.n_state
  n_action = mdp.n_action
  G = mdp.G

  ns = max([len(G[i][0]) for i in range(n_factor_s)])
  na = max([len(G[i][1]) for i in range(n_factor_s)])
  new_n_factor_a = n_factor_a + n_factor_s
  new_n_action = max(n_action, n_state)
  new_G = [[G[i][0], [i, new_n_factor_a - 1]] for i in range(n_factor_s)]
  p_new = np.zeros((n_factor_s, n_state**ns, new_n_action**len(new_G[0][1]), n_state))
  r_new = np.zeros((n_factor_s, n_state**ns, new_n_action**len(new_G[0][1])))
  a_idx = index_machine(new_n_factor_a, new_n_action)

  # mdp.total_visitations = mdp.total_visitations * 100

  for ft in range(n_factor_s):
    S = mdp.state_index.enumerate(G[ft][0])
    A = mdp.action_index.enumerate(G[ft][1])
    for xt in range(len(S)):
      for at in range(len(A)):
        delta1 = np.sqrt(alpha * p_hat[ft, xt, at, :] * np.log(6 * n_factor_s * n_state * n_state**len(G[ft][0]) * t_k / rho) / np.max([mdp.total_visitations[ft, xt, at], 1]))
        delta2 = alpha * np.log(6 * n_factor_s * n_state * n_state**len(G[ft][0]) * t_k / rho) / np.max([mdp.total_visitations[ft, xt, at], 1])
        delta = delta1 + delta2
        delta = np.min(np.concatenate((delta.reshape(-1, 1), p_hat[ft, xt, at, :].reshape(-1, 1)), 1), 1)
        sum_delta = np.sum(delta)

        r_delta = np.sqrt(alpha * np.log(6 * n_factor_s * n_state * n_state**len(G[ft]) * t_k / rho) / np.max([mdp.total_visitations[ft, xt, at], 1]))
        r_plus = r_hat[ft, xt, at] + r_delta
        r_plus = np.clip(r_plus, 0, 1)
        for stt in range(n_state):
          p_plus = p_hat[ft, xt, at, :] - delta
          p_plus[stt] += sum_delta
          cur_ac = np.zeros(new_n_factor_a, dtype = 'int')
          cur_ac[-1] = at
          cur_ac[ft] = stt
          p_new[ft, xt, a_idx.find_idx(cur_ac, [ft, new_n_factor_a - 1]), :] = p_plus
          r_new[ft, xt, a_idx.find_idx(cur_ac, [ft, new_n_factor_a - 1])] = r_plus
  return fmdp(n_factor_s, new_n_factor_a, n_state, new_n_action, p_new, r_new, new_G, init_dist = None)


def dorl(mdp, delta, initial_state=None, show = 10000, O = None, use_fix = False, accurate = False, alpha = 18):
  '''
  UCRL2 algorithm
  See _Near-optimal Regret Bounds for Reinforcement Learning_. Jaksch, Ortner, Auer. 2010.
  '''
  n_factor_s, n_factor_a, n_state, n_action, G = mdp.n_factor_s, mdp.n_factor_a, mdp.n_state, mdp.n_action, mdp.G
  t = 1
  # Initial state
  st = mdp.reset(initial_state)
  h_k = 0
  for k in itertools.count():
    # print(k)
    # Initialize episode k
    h_k += 1
    t_k = t
    p_hat, r_hat = mdp.empirical_estimate()
    mdp_k = do_extended_mdp(mdp,p_hat,r_hat,delta,t_k, alpha = alpha)
    if O == None:
      O = order(mdp, extended = True)
    if accurate:
      mdp_k.FactoredALP(O, basis = "accurate")
    else:
      mdp_k.FactoredALP(O)
    mdp_k.opt.lp_problem.solve()

    # Execute policy
    ac = mdp_k.policy(st)
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
      ac = mdp_k.policy(st)
if __name__ == '__main__':
  mdp = generate()
  transitions = porl(mdp, delta=0.1, initial_state=0)
  tr = []
  for _ in range(10):
      (t, st, ac, next_st, r) = transitions.__next__()
      tr.append((t, st, ac, next_st, r))
