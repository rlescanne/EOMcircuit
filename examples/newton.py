# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:03:56 2021

@author: dione
"""

from scipy.optimize import minimize, newton, root
import numpy as np

def f(x):
    return (x-1j)*(x-1)

res = newton(f, np.array([0.5j-0.5]))
print(res)