# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:57:17 2018

@author: Gray
"""
#This doucment will hold Hamiltonians!
from numpy import array
def H_Pairing_n4_p3(delta,g,alpha):
    H=array([[2*delta-g , -0.5*g , -0.5*g],
             [-0.5*g , 4*delta-g-2j*alpha, -0.5*g],
             [-0.5*g , -0.5*g , 6*delta-g-2j*alpha]]
        )
    return H

def H_Pairing_n4_p4(delta, g, alpha):
    H = array(
        [[2*delta-g,    -0.5*g,               -0.5*g,    -0.5*g,                -0.5*g,                    0.],
         [   -0.5*g, 4*delta-g,               -0.5*g,    -0.5*g,                    0.,                -0.5*g],
         [   -0.5*g,    -0.5*g, 6*delta-g-2.0j*alpha,        0.,                -0.5*g,                -0.5*g],
         [   -0.5*g,    -0.5*g,                    0, 6*delta-g,                -0.5*g,                -0.5*g],
         [   -0.5*g,         0,               -0.5*g,    -0.5*g,  8*delta-g-2.0j*alpha,                -0.5*g],
         [        0,    -0.5*g,               -0.5*g,    -0.5*g,                -0.5*g, 10*delta-g-2.0j*alpha]]
      )
    
    return H