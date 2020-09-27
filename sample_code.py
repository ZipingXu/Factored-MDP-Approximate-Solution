"""
Author  :    Ziping Xu
Email   :    zipingxu@umich.edu
Date    :    Sep 27, 2020
Record  :    A sample code
"""

from fmdp import *

# generate a threeleg environment

# two parameters for the difficulties of each env
alpha1 = 0.1
alpha2 = 0.1

n_factor_s = 4
n_factor_a = 1; n_state = 2; n_action = n_factor_s + 1

mdp = generate_3leg(n_factor_s, n_factor_a, n_state, n_action, alpha1 = alpha1, alpha2 = alpha2)

# view transition and reward function
mdp.p
mdp.r

# Solving FMDP
O = order(mdp) # generate orders for variables elimination

# Accurate solution
mdp.FactoredALP(O, basis = "accurate")
mdp.opt.lp_problem.solve()
lambda_star = mdp.opt.obj.value()

print(lambda_star)

# Approximate solution (fast)
mdp.FactoredALP(O)
mdp.opt.lp_problem.solve()
lambda_app1 = mdp.opt.obj.value()
print(lambda_app1)

# Approximate solution without variable elimination (slow)
mdp.FactoredDirect()
mdp.opt.lp_problem.solve()
lambda_app2 = mdp.opt.obj.value()
print(lambda_app2)
print(lambda_app1 == lambda_app2)
