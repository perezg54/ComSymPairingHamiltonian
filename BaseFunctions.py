# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:00:00 2018

@author: Gray
"""
#This file will hold my basic functions!
from numpy import dot, diag, reshape

# commutator of matrices
def commutator(a,b):
  return dot(a,b) - dot(b,a)

# Hamiltonian to initial vector
def HtoY0(H):
    y0=reshape(H,-1)
    return y0

# derivative / right-hand side of the flow equation
def derivative(y, t, dim, a, b, c, d):

  # reshape the solution vector into a dim x dim matrix
  H = reshape(y, (dim, dim))

  # extract diagonal Hamiltonian...
  Hd  = diag(diag(H))

  # ... and construct off-diagonal the Hamiltonian
  Hod = H-Hd
  
  # calculate the generator
  eta = a*c*commutator(Hd.real, Hod.real)-b*d*commutator(Hd.imag, Hod.imag)+1.0j*(a*d*commutator(Hd.real,Hod.imag)+b*c*commutator(Hd.imag,Hod.real))

  # dH is the derivative in matrix form 
  dH  = commutator(eta.real, H.real)-commutator(eta.imag, H.imag)+1.0j*(commutator(eta.real,H.imag)+commutator(eta.imag, H.real))

  # convert dH into a linear array for the ODE solver
  dydt = reshape(dH, -1)
    
  return dydt