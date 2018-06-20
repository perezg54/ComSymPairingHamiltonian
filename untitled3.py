# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 07:00:15 2018

@author: Gray
"""
import numpy as np
import matplotlib.pyplot as plt

def Plot_Evolution(g,data, eigenvals):
    plt.errorbar(g,data.real,yerr=data.imag,ecolor='blue',mfc='blue')
    plt.errorbar(g,eigenvals.real,yerr=eigenvals.imag,ecolor='red',mfc='red')