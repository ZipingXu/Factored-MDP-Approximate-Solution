"""
Author  :    Ziping Xu
Email   :    zipingxu@umich.edu
Date    :    Jan 12, 2020
Record  :    New experiements codes
"""

import math
import numpy as np
from fmdp import *
from rmax_fmdp import rmax
from dorl_fmdp import dorl
from psrl_fmdp import psrl
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# include standard modules
import argparse

def evaluate(mdp, n = 10000):
  # lambda_star = mdp.value_iteration(0.0001)
  O = order(mdp)
  mdp.FactoredALP(O)
  mdp.opt.lp_problem.solve()
  st = mdp.reset()
  ac = mdp.policy(st, extended = False)
  r = []
  while n > 0:
    st, ac, next_st, reward = mdp.step(np.array([ac]))
    r.append(reward)
    # Update statistics
    mdp.record(st, ac, next_st, reward)
    n -= 1
    st = next_st
    ac = mdp.policy(st, extended = False)
  return np.mean(r) * mdp.n_factor_s

def do_experiment(mdp, T = 10000, delta = 1, lambda_star = None,
                  initial_state=None, show = 1000, use_fix = True,
                  alg = "psrl", O = None, rmax_m = 100, scale = 10,
                  accurate = False):
  # lambda_star = mdp.value_iteration(0.0001)
  mdp.reset_recorder()

  if alg == "dorl":
  # porl
    O = order(mdp, extended = True)
    transitions = dorl(mdp, delta=delta, initial_state=initial_state, show = show, use_fix = use_fix, alpha = scale, O = O)
  # psrl
  if alg == "psrl":
    transitions = psrl(mdp, scale = scale, initial_state=initial_state, show = show, use_fix = use_fix, O = O)
  # rmax
  if alg == "rmax":
    transitions = rmax(mdp, m = [rmax_m]*mdp.n_factor_s, initial_state=initial_state, show = show, use_fix = use_fix, O = O)
  tr = []
  for _ in range(T):
      (t, st, ac, next_st, r) = transitions.__next__()
      tr.append((t, st, ac, next_st, r))
  reward = np.cumsum(list(map(lambda x: np.sum(x[-1]), tr)))
  regret = lambda_star * (np.array(range(T))+1) - reward
  return regret

def plot(regret, title = "Random-6-2-dynamic-seed_1", alg = "PSRL"):
  regret = np.array(regret)
  regret_mean = np.mean(regret, axis = 0)
  regret_sd = np.std(regret, axis = 0)
  regret_max = regret_mean + regret_sd
  regret_min = regret_mean - regret_sd

  plt.figure(figsize=(8, 5))

  x = list(range(len(regret_mean)))
  plt.plot(x, regret_mean, label=alg, color='#1B2ACC')
  plt.fill_between(x, regret_min, regret_max, facecolor='#089FFF', alpha = 0.5, edgecolor='#1B2ACC')
  plt.ylabel('Regret')
  plt.legend()
  plt.title(title)
  # plt.show()
  plt.savefig(title)

# initiate the parser
parser = argparse.ArgumentParser()

parser.add_argument("--alg", help="Algorithms to use: psrl, dorl, rmax", default = "psrl")
parser.add_argument("-p", help="Type of MDP to solve: circle or threeLeg", default = "circle")
parser.add_argument("-s", help="Number of state factors", type = int, default = 4)

parser.add_argument("-n", help="Number of replicate", type = int, default = 20)
parser.add_argument("-t", help="Numer of steps for each replication", type=int, default = 100)

# Fixed horizon uses the stopping criterion in DORL paper. If it is set False, we use the criterion in Ouyang 2017.
parser.add_argument("-f", help="Whether to use fixed horizon", type = bool, default = False)
parser.add_argument("--acc", help="Whether to use accurate planner to calculate the true regret. If it is set False, use approximate planner", type = bool, default = False)

parser.add_argument("--show", help="Show results", type = int, default = 100)
parser.add_argument("--seed", help="Random seed", type = int, default = 1)
parser.add_argument("-d", help="Directory to save", default = "./exp/")

parser.add_argument("-c", help="alpha for DORL or PSRL", type = float, default = 0.1)
parser.add_argument("-m", help="m for Rmax", type = int, default = 100)

# Two parameters for environment generating.
parser.add_argument("-a", help="alpha1", type = float, default = 0.1)
parser.add_argument("-b", help="alpha2", type = float, default = 0.1)



# read arguments from the command line
args = parser.parse_args()

print("Use %s to solve %s with size %d and T %d; fixed horizon: %d; parameters %f, %f, %d, %f; accurate planner %d"%(args.alg, args.p, args.s, args.t, int(args.f), args.a, args.b, args.m, args.c, int(args.acc)))

# circle
if args.p == "circle":
  np.random.seed(args.seed)
  n_factor_s = args.s; n_factor_a = 1; n_state = 2; n_action = n_factor_s + 1
  mdp = generate(n_factor_s, n_factor_a, n_state, n_action, alpha1 = args.a, alpha2 = args.b)
  O = order(mdp)
  T = args.t; delta = 1; initial_state=np.array([0]*n_factor_s); show = args.show; use_fix = args.f; mdp = mdp


# three legs
if args.p == "threeLeg":
  np.random.seed(args.seed)
  n_factor_s = args.s; n_factor_a = 1; n_state = 2; n_action = n_factor_s + 1
  mdp = generate_3leg(n_factor_s, n_factor_a, n_state, n_action, alpha1 = args.a, alpha2 = args.b)
  O = order(mdp)
  O.reverse()
  O = O[1:] + [(1, 0)]
  T = args.t; delta = 1; initial_state=np.array([0]*n_factor_s); show = args.show; use_fix = args.f; mdp = mdp

regret = []
if args.acc:
  print("Start to evaluate the true optimal average reward")
  mdp.FactoredALP(O, basis = "accurate")
  mdp.opt.lp_problem.solve()
  lambda_star = mdp.opt.obj.value()
  print("True optimal average reward: %f" % (lambda_star))
else:
  print("Start to evaluate the approximate optimal average reward")
  lambda_star = evaluate(mdp)
  print("Approximate optimal average reward: %f" % (lambda_star))

for i in range(args.n):
  print("Replication %d" % (i))
  tmp_regret = do_experiment(
    mdp, T = T, initial_state = initial_state, lambda_star = lambda_star, use_fix = args.f, alg = args.alg, O = O, rmax_m = args.m, scale = args.c, accurate = args.acc, show = show)
  regret.append(tmp_regret)

file = "%s_%s_%d_%.2f_%.2f_%d_%.2f" %(args.alg, args.p, n_factor_s, args.a, args.b, args.m, args.c)
if args.f:
  file += "_fixed"
else:
  file += "_early"
if args.acc:
  file += "_acc"
else:
  file += "_apx"
plot(regret, title = "%s%s.png"%(args.d, file))
pickle.dump(regret, open( "%s%s.pkl"%(args.d, file), "wb" ))
