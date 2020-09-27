"""
Author  :    Ziping Xu
Email   :    zipingxu@umich.edu
Date    :    Dec 4, 2019
Record  :    Factored MDP implementation
"""

import numpy as np
import math
from functools import reduce
#from pulp import *
import pulp
from pulp import LpVariable
from copy import deepcopy

class optimizer:
  def __init__(self, n_basis):
    self.lp_problem = pulp.LpProblem("My LP Problem", pulp.LpMinimize)
    self.n_basis = n_basis
    self.w = [pulp.LpVariable('w_%d'%(i), cat='Continuous') for i in range(n_basis)]
    self.obj = pulp.LpVariable('lambda', lowBound=0, upBound=10, cat='Continuous')

    # add objective function
    self.lp_problem += self.obj

# Find index of some state or action
class index_machine:
  def __init__(self, factor, n):
    """[summary]

    Dealing with the conversion between the real state or action vectors and their indices in the transition matrix

    Arguments:
      factor {int} -- number of factors
      n {int or list} -- number of states for each factor
    """
    self.factor = factor
    if type(n) == list:
      self.n_list = n
      self.n = max(n)
    else:
      self.n = n
      self.n_list = [n] * self.factor
  def find_idx(self, x, scope):
    """ Find index of a state using its scope
    Arguments:
      x {np.array} -- a state or action vector of dim self.n
      scope {list} -- scope list

    Returns:
      int -- The index of the state or action

    Expample:
      factor = 2; n = 3
      x = [0, 1]; scope = [1]
      Then 1 is the second element in the scope [1]. Return 1
    """
    if type(x) == list:
      x = np.array(x)
    if type(x) == int:
      x = np.array([x])
    x = x[scope]
    tmp = np.array(self.n)**(x.shape[0] - np.array(range(x.shape[0])) - 1)
    return np.sum(tmp * x)
  def enumerate(self, scope):
    """

    Enumerate states or actions given a scope

    Arguments:
      scope {[list]} -- scope list ex. [0, 1]

    Returns:
      [list of arrays]
    Example:
      factor = 2; n = 3; scope = [1];
      Then returns [0, 0], [0, 1], [0, 2]
    """
    scope.sort()
    states = []
    def gen_states(current, target, iter):
      if iter == target:
        states.append(current)
        return
      for i in range(self.n):
        gen_states(current + [i], target, iter+1)
    gen_states([], target = len(scope), iter = 0)
    states = list(map(lambda x: self.extend(x, scope), states))
    if len(states) == 0:
      states = np.array([0] * self.n_factor)
    return states
  # Check whether a (x, scope) pair is valid
  def valid(self, x, scope):
    for i in scope:
      if x[i] >= self.n_list[i]:
        return False
    return True
  # Make extension for DORL
  def extend(self, sub_states, scope):
    new = np.array([0]*self.factor)
    new[scope] = sub_states
    return new

# main class
class fmdp:
  def __init__(self, n_factor_s, n_factor_a, n_state, n_action, p, r, G, init_dist = None):
    # We assume that number of factored = dimensions for states
    self.n_factor_s = n_factor_s  # number of factors
    self.n_factor_a = n_factor_a  # number of factors
    if type(n_state) == int:
      self.n_state = [n_state] * n_factor_s
    else:
      self.n_state = n_state    # number of states in each factor
    if type(n_action) == int:
      self.n_action = [n_action] * n_factor_a
    else:
      self.n_action = n_action  # number of actions in each factor

    self.state_index = index_machine(self.n_factor_s, self.n_state)
    self.action_index = index_machine(self.n_factor_a, self.n_action)

    self.n_action = max(self.n_action)
    self.n_state = max(self.n_state)

    self.p = p                # dim: target factor, all S in its scope, all A in its scope, S
    self.r = r                # dim: target factor, all S in its scope, all A in its scope
    self.G = G                # Depends on which factors
    self.generate_basis()     # generate the simple single plus basis function
    self.opt = None           # The optimization problem
    if init_dist == None:
      self.init_dist = (np.zeros(self.n_state) + 1) / self.n_state
    self.state = np.zeros(self.n_factor_s)

  # Record transitions
  def reset_recorder(self):
    ns = max([len(self.G[i][0]) for i in range(self.n_factor_s)])
    na = max([len(self.G[i][1]) for i in range(self.n_factor_s)])
    self.total_visitations = np.zeros((self.n_factor_s, self.n_state**ns, self.n_action**na), dtype = 'float')
    self.total_rewards = np.zeros((self.n_factor_s, self.n_state**ns, self.n_action**na), dtype = 'float')
    self.total_transitions = np.zeros((self.n_factor_s, self.n_state**ns, self.n_action**na, self.n_state), dtype = 'float')

  # Reset recorder
  def reset(self, init_state=None):
    if init_state is None:
      self.state = np.array([np.random.choice(self.n_state, p=self.init_dist) for i in range(self.n_factor_s)])
    else:
      self.state = init_state
    self.reset_recorder()
    return self.state

  # Store a transition
  def record(self, s, a, ns, r):
    for i in range(self.n_factor_s):
      idx_s = self.state_index.find_idx(s, self.G[i][0])
      idx_a = self.action_index.find_idx(a, self.G[i][1])
      self.total_visitations[i, idx_s, idx_a] += 1
      self.total_transitions[i, idx_s, idx_a, ns[i]] += 1
      self.total_rewards[i, idx_s, idx_a] += r[i]

  # Calculate empirical estimate
  def empirical_estimate(self):
    tmp = np.clip(self.total_visitations, 1, None)
    p_hat = self.total_transitions / np.expand_dims(tmp, 3)
    r_hat = self.total_rewards / tmp
    p_hat[:, :, :, -1] = 1 - np.sum(p_hat[:, :, :, 0:-1], 3)
    return p_hat, r_hat

  # Take a step using action
  def step(self, action):
    next_state = np.zeros(self.n_factor_s, dtype = "int")
    next_reward = []
    for i in range(self.n_factor_s):
      # idx_a = self.action_index.find_idx(a, self.G[i][1])
      # idx_s = self.state_index.find_idx(self.state, self.G[i][0])
      tran_func = self.transition([i])
      next_state[i] = np.random.choice(self.n_state, p=[tran_func(self.state, action, self.state_index.extend(y, [i])) for y in range(self.n_state)])
      rwd_func = self.rewards(i)[1]
      next_reward += [rwd_func(self.state, action)]
    previous_state = deepcopy(self.state)
    self.state = next_state
    return previous_state, action, next_state, next_reward

  # Find parent indices of a scope
  def parent(self, scope):
    flatten = lambda l: list(set([item for sublist in l for item in sublist]))
    return flatten([self.G[i][0] for i in scope]), flatten([self.G[i][1] for i in scope])

  # Return a function for a simple transition
  def single_transition(self, source_s, source_a, j):
    """[summary]

    [description]

    Arguments:
      source_s {list} -- scope for state
      source_a {list} -- scope for action
      j {int} -- target factor

    Returns:
      [function] -- single transition for target j, given current state, action and next state
    """
    # Input:
    ## s: current state; a: current action; y: next state
    return lambda s, a, y: self.p[j, self.state_index.find_idx(s, source_s), self.action_index.find_idx(a, source_a), y[j]]

  # function for multiple target factors
  def transition(self, target_factor):
    if type(target_factor) != list:
      target_factor = list(target_factor)
    p = [lambda s, a, y: 1]
    for j in target_factor:
      source_s = self.G[j][0]
      source_a = self.G[j][1]
      p.append(self.single_transition(source_s, source_a, j))
    return lambda s, a, y: np.array([tmp_p(s, a, y) for tmp_p in p]).cumprod()[-1]

  # reward function
  def rewards(self, target_factor):
    # generate b functions
    source_s = self.G[target_factor][0]
    source_a = self.G[target_factor][1]
    return ([source_s, source_a], lambda s, a: self.r[target_factor, self.state_index.find_idx(s, source_s), self.action_index.find_idx(a, source_a)])

  # functions for generating basis for approximate solution
  def generate_single(self, i):
  # for approximate solution using basis f_i(x) = x[i] i = 1, ..., n
    return lambda x: x[i]
  def generate_full(self, s):
  # basis for accurate solution
    return lambda x: int(sum((s-x)**2) == 0)
  def generate_basis(self, mode = "singlePlus"):
    if mode == "singlePlus":
      self.basis = []
      for i in range(self.n_factor_s):
        self.basis.append(([i], self.generate_single(i)))
    if mode == "accurate":
      self.basis = []
      source = list(range(self.n_factor_s))
      for s in self.state_index.enumerate(source):
        self.basis.append((source, self.generate_full(s)))

  # Generating Linear Programing Problem for Approximate Planner. See Figure 5 of Guestrin 2003 E±cient Solution Algorithms for Factored MDPs
  def FactoredLP(self, C, O):
    Funcs = []
    idx = -1
    for c in C:
      scope = c[0]
      S = self.state_index.enumerate(scope[0])
      A = self.action_index.enumerate(scope[1])
      LP_vars = []
      idx += 1
      for s in S:
        for a in A:
          if not (self.action_index.valid(a, scope[1]) and self.state_index.valid(s, scope[0])):
            continue
          # add variables
          LP_vars.append((LpVariable("fuc_%d_s_(%s)_a_(%s)_scp_(%s)"%(idx,str(s[scope[0]]), str(a[scope[1]]), str(scope)), cat='Continuous'), [s, a], scope))
          # add constraints
          self.opt.lp_problem.addConstraint(LP_vars[-1][0] == c[1](s, a))
      Funcs += LP_vars
    for i in range(len(O)):
      # The order is very important
      elim_type, l = O[i]
      Funcs_l = list(filter(lambda x: x[2][elim_type].count(l) > 0, Funcs))
      Funcs = list(filter(lambda x: x[2][elim_type].count(l) == 0, Funcs))
      if len(Funcs_l) == 0:
        continue
      new_scope = [set.union(*(list((map(lambda x: set(x[2][0]), Funcs_l))) + [set()])), set.union(*(list((map(lambda x: set(x[2][1]), Funcs_l))) + [set()]))]
      new_scope[elim_type].remove(l)
      # print(new_scope)
      new_scope[0] = list(new_scope[0])
      new_scope[1] = list(new_scope[1])
      S = self.state_index.enumerate(new_scope[0])
      A = self.action_index.enumerate(new_scope[1])
      LP_vars = []
      for s in S:
        for a in A:
          if not (self.action_index.valid(a, new_scope[1]) and self.state_index.valid(s, new_scope[0])):
            continue
          LP_vars.append((LpVariable("Elim_%d_s_(%s)_a_(%s)_scp_(%s)"%(i, str(s[new_scope[0]]), str(a[new_scope[1]]), new_scope), cat='Continuous'), [s, a], new_scope))
          z = [s, a]
          if elim_type == 0:
            max_over = self.n_state
          else:
            max_over = self.n_action
          for x_l in range(max_over):
            tmp_z = deepcopy(z)
            tmp_z[elim_type][l] = x_l
            lp_list = list(filter(lambda x: np.sum((x[1][0] != tmp_z[0])[x[2][0]]) + np.sum((x[1][1] != tmp_z[1])[x[2][1]]) == 0, Funcs_l))
            self.opt.lp_problem.addConstraint(LP_vars[-1][0] >= sum([tmp[0] for tmp in lp_list]))
      Funcs += LP_vars
    # add last constraint
    self.opt.lp_problem.addConstraint(self.opt.obj >= Funcs[0][0])

  # Given a target factor, find its scopes
  def back_proj(self, basis, i):
    # generate C functions
    tmp_scope, tmp_f = basis
    parent_scope = self.parent(tmp_scope)
    state_scope = self.state_index.enumerate(tmp_scope)
    g_tmp = lambda x, a: sum([tmp_f(s) * self.transition(tmp_scope)(x, a, s)  for s in state_scope])
    g_ret = lambda x, a: self.opt.w[i] * (g_tmp(x, a) - tmp_f(x))
    scope_ret = [list(set(tmp_scope + parent_scope[0])), parent_scope[1]]
    return scope_ret, g_ret

  # Direct solving the LP problem without variables elimination (Very slow. Only used for verifying correctness.)
  def FactoredDirect(self, basis = "singlePlus"):
    self.generate_basis(basis)
    self.opt = optimizer(len(self.basis))
    g_func = []
    for i in range(len(self.basis)):
      g_func.append(self.back_proj(self.basis[i], i))
    r_func = [self.rewards(i) for i in range(self.n_factor_s)]

    C = g_func + r_func
    S = self.state_index.enumerate(list(range(self.n_factor_s)))
    A = self.action_index.enumerate(list(range(self.n_factor_a)))
    for s in S:
      for a in A:
        if not (self.action_index.valid(a, list(range(self.n_factor_a))) and self.state_index.valid(s, list(range(self.n_factor_s)))):
          continue
        self.opt.lp_problem.addConstraint(self.opt.obj >= sum([tmp[1](s, a) for tmp in C]), name='%s_%s'%(str(s), str(a)))

  # Solving FMDP using Approxiamte solution
  def FactoredALP(self, O, basis = "singlePlus"):
    # Generate weigths for each basis
    self.generate_basis(basis)
    self.opt = optimizer(len(self.basis))
    g_func = []
    for i in range(len(self.basis)):
      g_func.append(self.back_proj(self.basis[i], i))
    r_func = [self.rewards(i) for i in range(self.n_factor_s)]
    self.FactoredLP(g_func + r_func, O)
    #status = self.opt.lp_problem.solve()
    #return status.real, [pulp.value(x) for x in self.opt.w]

  # Generate action given current state and optimized basis
  # Only works for the circle or threeleg environment
  def policy(self, s, extended = True):
    tmp_max = 0
    best_a = 0
    if extended:
      tmp_max = 0
      best_a = 0
      for a in range(self.n_action):
        aa = np.zeros(self.n_factor_a, dtype = 'int')
        aa[-1] = a
        tmp = sum([self.rewards(i)[1](s, aa) for i in range(self.n_factor_s)])
        w = [x.value() for x in self.opt.w]
        for i in range(len(self.basis)):
          tmp_scope, tmp_f = self.basis[i]
          parent_scope = self.parent(tmp_scope)
          state_scope = self.state_index.enumerate(tmp_scope)
          g_tmp = lambda x, a: sum([tmp_f(s) * self.transition(tmp_scope)(x, a, s)  for s in state_scope])
          g_ret = lambda x, a: w[i] * (g_tmp(x, a) - tmp_f(x))
          aaa = deepcopy(aa)
          aaa[i] = 1
          tmp += max(g_ret(s, aa), g_ret(s, aaa))
        if tmp >= tmp_max:
          best_a = a
          tmp_max = tmp
      return np.array([best_a])
    else:
      for a in range(self.n_action):
        aa = np.array([a])
        tmp = sum([self.rewards(i)[1](s, aa) for i in range(self.n_factor_s)])
        w = [x.value() for x in self.opt.w]
        for i in range(len(self.basis)):
          tmp_scope, tmp_f = self.basis[i]
          parent_scope = self.parent(tmp_scope)
          state_scope = self.state_index.enumerate(tmp_scope)
          g_tmp = lambda x, a: sum([tmp_f(s) * self.transition(tmp_scope)(x, a, s)  for s in state_scope])
          tmp += g_tmp(s, aa)
        if tmp >= tmp_max:
          best_a = a
          tmp_max = tmp
      return np.array([best_a])

# Generate Orders for variable elimination
def order(mdp, extended = False):
  if not extended:
    return [(0, i) for i in range(mdp.n_factor_s)] + [(1, 0)]
  else:
    flatten = lambda l: list([item for sublist in l for item in sublist])
    return flatten([[(0, mdp.n_factor_s - i - 1), (1, mdp.n_factor_s - i - 1)] for i in range(mdp.n_factor_s)]) + [(1, mdp.n_factor_s)]

# Generate a Circle environment
def generate(n_factor_s = 2, n_factor_a = 1, n_state = 2, n_action = 3, alpha1 = 0.1, alpha2 = 0.1):
  # n_factor_s, n_factor_a, n_state, n_action =
  s_idx = index_machine(n_factor_s, n_state)
  a_idx = index_machine(n_factor_a, n_action)

  p = []
  r = []
  G = []
  for fs in range(n_factor_s):
    G.append([[fs, fs - 1 if fs - 1 >= 0 else n_factor_s-1], [0]])

  for fs in range(n_factor_s):
    states = s_idx.enumerate(G[fs][0])
    actions = a_idx.enumerate(G[fs][1])
    for s in range(len(states)):
      for a in range(len(actions)):
        tmp_p = 0
        tmp_r = 0
        if states[s][fs] == 1:
          tmp_p = np.random.random() * alpha1
          tmp_r = 1
        else:
          tmp_p = max(np.random.random() * 1, 0.5)
        for ff in G[fs][0]:
          if states[s][ff] == 0:
            tmp_p += np.random.random() * alpha2
        if actions[a] == fs:
          tmp_p = 0
        if tmp_p > 1:
          tmp_p = 1
        r.append(tmp_r)
        p += [tmp_p, 1-tmp_p]

  p = np.ndarray((n_factor_s, n_state**2, n_action**1, n_state), buffer = np.array(p))
  r = np.ndarray((n_factor_s, n_state**2, n_action**1), buffer = np.array(r, dtype = "float"))
  mdp = fmdp(n_factor_s, n_factor_a, n_state, n_action, p, r, G)
  return mdp

# Generate a threeLeg environment
def generate_3leg(n_factor_s = 7, n_factor_a = 1, n_state = 2, n_action = 3, alpha1 = 0.1, alpha2 = 0.1):
  # n_factor_s, n_factor_a, n_state, n_action =
  s_idx = index_machine(n_factor_s, n_state)
  a_idx = index_machine(n_factor_a, n_action)

  p = []
  r = []
  G = []
  #for fs in range(n_factor_s):
  #  G.append([[fs, fs - 1 if fs - 1 >= 0 else n_factor_s-1], [0]])
  G.append([[0], [0]])

  for i in range(n_factor_s - 1):
    if i >= 3:
      G.append([[i+1, i - 2], [0]])
    else:
      G.append([[i+1, 0], [0]])
  """
  n = int((n_factor_s - 1) / 3)
  for i in range(n):
    if i == 0:
      G.append([[i * 3 + 1, 0], [0]])
      G.append([[i * 3 + 2, 0], [0]])
      G.append([[i * 3 + 3, 0], [0]])
    else:
      G.append([[i * 3 + 1, i * 3 -2], [0]])
      G.append([[i * 3 + 2, i * 3 -1], [0]])
      G.append([[i * 3 + 3, i * 3], [0]])
  """
  p = np.zeros((n_factor_s, n_state**2, n_action**1, n_state))
  r = np.zeros((n_factor_s, n_state**2, n_action**1))
  for fs in range(n_factor_s):
    states = s_idx.enumerate(G[fs][0])
    actions = a_idx.enumerate(G[fs][1])
    for s in range(len(states)):
      for a in range(len(actions)):
        tmp_p = 0
        tmp_r = 0
        if states[s][fs] == 1:
          tmp_p = np.random.random() * alpha1
          tmp_r = 1
        else:
          tmp_p = max(np.random.random() * 1, 0.5)
        for ff in G[fs][0]:
          if states[s][ff] == 0:
            tmp_p += np.random.random() * alpha2
        if actions[a] == fs:
          tmp_p = 0
        if tmp_p > 1:
          tmp_p = 1
        r[fs, s, a] = tmp_r
        p[fs, s, a, :] = np.array([tmp_p, 1-tmp_p])
  mdp = fmdp(n_factor_s, n_factor_a, n_state, n_action, p, r, G)
  return mdp
