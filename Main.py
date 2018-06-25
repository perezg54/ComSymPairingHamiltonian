# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:09:14 2018

@author: Gray
"""
#This is the Main file! Hopefully this is kept very simple and small!
import numpy as np
from numpy import linspace, array, logspace
from numpy.linalg import eigvals
from odeintw import odeintw
from Ploting import linear_ploter, log_ploter
from Hamiltonians import H_Pairing_n4_p3, H_Pairing_n4_p4
from BaseFunctions import HtoY0, derivative

def main():
    flowparams=np.linspace(0,100,2)
    gspace=linspace(-100,100,101)
    alpha=1.0
    g=1.0
    delta=1.0
    while alpha <=1.0:
        solution_set=[]
        eigenvalue_set=[]
        for h in gspace:
            #initial pairing matrix
            H0=H_Pairing_n4_p4(delta,h,alpha)
            #eigenvalues y standard diagonalization
            eigenvalue_set.append(eigvals(H0))
            # just reshapes the initial matrix into an array, honestly having a
            # function for this is over kill
            y0=HtoY0(H0)
            # generates an array of srg evoled matrix elements 
            # while I did make another version that didn't use odeintw, I ended
            # up using it again as effective shorthand since it seemed consistent
            # with results of my other version.
            solution_set.append(odeintw(derivative, y0, flowparams, args=(H0.shape[0],1.0,0.0,1.0,1.0))[-1])
        linear_ploter(array(solution_set).real,array(eigenvalue_set).real,gspace,True,alpha)
        linear_ploter(array(solution_set).imag,array(eigenvalue_set).imag,gspace,False,alpha)
        alpha=alpha+1.0
    print 'end'
    
if __name__ == "__main__": 
  main()