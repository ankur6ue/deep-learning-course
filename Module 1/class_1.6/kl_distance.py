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
import matplotlib.pyplot as plt
# Calculate KL distance between two distributions
# Def: KL_z2z1= sum p(z2) * (log(p(z2)) - log(p(z1)))
p_z1 = np.array([1,1,1,1,1,1])/6
p_z2 = np.array([1/8, 1/8, 3/16, 3/16, 3/16, 3/16])
KL_z2z1 = 0
KL_z1z2 = 0
for i in range(0,len(p_z1)):
    KL_z2z1 += p_z2[i] * (np.log(p_z2[i]) - np.log(p_z1[i]))
    KL_z1z2 += p_z1[i] * (np.log(p_z1[i]) - np.log(p_z2[i]))
# Note KL_z1z2 != KL_z2z1

# p_X is the prob distribution of a fair coin
p_X = np.array([1,1])/2
N = 100
# We are going to vary the heads probability of a biased coin (Y), and compute KL and MSE distance between X and Y
ph = np.linspace(0.001, 0.99, N)
KL_xy = [] # KL
d_xy = []  # MSE
for h_ in ph:
    KL_xy.append(p_X[0] * (np.log(p_X[0]) - np.log(h_)) + (1 - p_X[0]) * (np.log(1-p_X[0]) - np.log(1-h_)))
    d_xy.append(0.5*(p_X[0] - h_)**2 + 0.5*(1-p_X[0] - (1-h_))**2)

plt.plot(ph, KL_xy, label='KL(x,y)', color='blue')
plt.plot(ph, d_xy, label='d(x,y)', color='red')
plt.title('KL distance and Mean Square Error between \n RVs X (unbiased coin), Y (biased coin)')
plt.xlabel('p(Y=Heads)')
plt.legend()
plt.show()
KL_xy = []
plt.xlabel('current training step')
plt.ylabel('learning rate')
plt.legend()
plt.show()

