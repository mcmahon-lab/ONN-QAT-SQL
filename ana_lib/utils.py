"""
Random util functions for data analysis
"""
import numpy as np
from scipy.interpolate import UnivariateSpline

def FWHM(X,Y):
    spline = UnivariateSpline(X, Y-np.max(Y)/2, s=0)
    ans = spline.roots() # find the roots
    return ans[-1] - ans[0]