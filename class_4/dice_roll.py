# Copyright 2025 Ankur Mohan
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np

# We have two unbiased dice with 4 faces. Random variable x represents the outcome of rolling the first dice,
# y represents rolling the second dice. x and y are discreet random variables, with 4 possible outcomes (1,2,3,4)
# each with probability 1/4. The joint distribution of x,y p(x=i,y=j) = p(i)p(j) = 1/16, for i and j in {1,2,3,4)
# Let's calculate the mean and variance of x, x^2, y and y^2
# E_x, E_y
E_x = 0
E_x_sq = 0
Var_x = 0
p_x = 1/4
for x in range(1,5):
    E_x += x * p_x
    E_x_sq += x * x * p_x

for x in range(1,5):
    Var_x += p_x * ((x - E_x) ** 2)
E_y = E_x # The two dice are identical, so x and y are independent and identically distributed
E_y_sq = E_x_sq
Var_y = Var_x

# Now let's consider a random variable z = x + y. Let's calculate the mean and variance of z manually, and confirm
# it matches the formula
####################
# z = x + y
# z can take values 2 to 8. Let's manually calculate the number of ways each outcome can occur
# 2 (1,1)
# 3 (2,1), (1,2)
# 4 (1,3), (2,2), (3,1)
# 5 (1,4), (4,1), (2,3), (3,2)
# 6 (2,4), (3,3), (4,2)
# 7 (3,4), (4,3)
# 8 (4,4)
z = np.array([2,3,4,5,6,7,8])
p_z = np.array([1,2,3,4,3,2,1])/16
# Notice that the distribution of z is shaped like a triangle and is more spread out than that of x or y
# so we expect its mean and variance to be larger than that of x and y..
E_z = 0
Var_z = 0
E_z_sq = 0
for idx in range(0, len(p_z)):
    z_ = z[idx]
    E_z += z_ * p_z[idx]
    E_z_sq += z_ * z_ * p_z[idx]

for idx in range(0, len(p_z)):
    z_ = z[idx]
    Var_z += p_z[idx] * ((z_ - E_z) ** 2)

# Verify var_z = var_x + var_y (because x and y are IID)
# verify E_z = E_x + E_y (always true, regardless of whether x and y are IID or not)
# verify Var_z = E_z_sq - E_z*E_z
# What does it E_z mean? It means if we take lots of random samples of z, the average should be what we calculated
# above
num_trials = 1000
z_hist = []
for i in range(0, num_trials):
    x = np.random.randint(low=1, high=5)
    y = np.random.randint(low=1, high=5)
    z = x + y
    z_hist.append(z)

E_z_sim = np.mean(z_hist)
Var_z_sim = np.var(z_hist)
# verify E_z_sim is close to E_z
# verify Var_z_sim is close to Var_z
# plot histogram of z_hist and compare with the distribution calculated above

# Now suppose z=xy. z can take values from 1-16.
# 1: 1,1
# 2: (1,2), (2,1)
# 3: (1,3), (3,1)
# 4: (1,4), (4,1), (2,2)
# 5: 0
# 6: (2,3), (3,2)
# 7: 0
# 8: (2,4), (4,2)
# 9: (3,3)
# 10: 0
# 11: 0
# 12: (3,4), (4,3)
# 13: 0
# 14: 0
# 15: 0
# 16: (4,4)
p_z = np.array([1,2,2,3,0,2,0,2,1,0,0,2,0,0,0,1])/16
# verify p_xy is a probability distribution.. i.e., np.sum(p_xy) == 1

# z = xy
E_z = 0
for z in range(1,17):
    E_z += z * p_z[z-1]
Var_z = 0
for z in range(1,17):
    Var_z += p_z[z-1] * ((z - E_z) ** 2)
# Note Var(z=xy) is higher than Var(z=x+y).. because z=xy is more "spread out" than z = x+y
test = E_x_sq * E_y_sq - E_z ** 2
# verify E_xy = E_x * E_y
# verify Var_xy = E_x_sq * E_y_sq - (E_xy)^2

# Now let's consider non-independent (but identically distributed) x and y. Suppose our dice are "quantum entangled",
# so that when
# one dice rolls 1, the other also rolls 1.
# The joint distribution is now more complicated..
# when (i,j) in (1,1), p(x=i,y=j) = 1/4. (1,2) (1,3) etc are not possible
# For (i,j) in {2,3,4), p(x=i,y=j) =  3/4 (prob of picking element from the subset {2,3,4} from {1,2,3,4}) * 1/3 *
# 1/3 (probability of picking an element from {2,3,4} = 1/12

# As before, let's consider z = x + y
# 2: (1,1) p = 1/4.. because if one dice rolls 1, the other must also!
# 3: 0.. because (1,2) and (2,1) combos are not possible!
# 4: (2,2) p = 1/12.. (3,1) (1,3) not possible
# 5: (2,3), (3,2) p=2*1/12
# 6: (2,4), (3,3), (4,2) p=3*1/12
# 7: (3,4), (4,3) p=2*1/12
# 8: (4,4) p=1*1/12

p_z = np.array([1/4, 0, 1/12, 1/6, 1/4, 1/6, 1/12])
z = np.array([2,3,4,5,6,7,8])
# verify np.sum(p_z) == 1
s = np.sum(p_z)
# Now let's calculate E_z and Var_z
E_z = 0
Var_z = 0
E_z_sq = 0
for idx in range(0, len(p_z)):
    z_ = z[idx]
    E_z += z_ * p_z[idx]
    E_z_sq += z_ * z_ * p_z

for idx in range(0, len(p_z)):
    z_ = z[idx]
    Var_z += p_z[idx] * ((z_ - E_z) ** 2)

# Verify E_z = E_x + E_y
# Var_z != Var_x + Var_y! Because x and y are correlated
# Var_z = Var_x + Var_y + 2Cov(x,y)
# Cov(x,y) = E(xy) - E(x)E(y)
# Let's calculate E_xy
# xy can have 16 possible values..let's calculate the probability of each
# 1: 1,1: p=1/4
# 2: 0. No way for xy = 2
# 3: 0
# 4: 2,2: p=1/12
# 5: 0
# 6: (2,3), (3,2): p = 2/12
# 7: 0
# 8: (2,4), (4,2): p = 2/12
# 9: (3,3): p = 1/12
# 10: 0
# 11: 0
# 12: (3,4), (4,3): p = 2/12
# 13: 0
# 14: 0
# 15: 0
# 16: (4,4) p = 1/12

p_xy = np.array([1/4, 0, 0, 1/12, 0,  1/6, 0, 1/6, 1/12, 0, 0, 1/6, 0, 0, 0, 1/12])
# verify p_xy is a prob distribution
s = p_xy.sum()
E_xy = 0
for idx in range(1,17):
    E_xy += idx * p_xy[idx-1]
# We calculated E(xy) earlier
cov_xy = E_xy - E_x * E_y
# Verify Var_z = Var_x + Var_y + 2Cov(x,y)

# Now let's calculate KL distance between z1 = x + y when x and y are IID, and z2 = x + y when x and y are entangled.
# Def: KL_z2z1= sum p(z2) * (log(p(z2)) - log(p(z1)))
p_z1 = np.array([1,2,3,4,3,2,1])/16
p_z2 = np.array([1/4, 0, 1/12, 1/6, 1/4, 1/6, 1/12])
KL_z2z1 = 0
for i in range(0,len(p_z1)):
    KL_z2z1 += p_z2[i] * (np.log(p_z2[i]) - np.log(p_z1[i]))
# Note KL_z1z2 can't be calculated.. because p(z2) is 0 in some places!
print('done')