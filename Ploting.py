# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:37:21 2018

@author: Gray
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:47:48 2018

@author: Gray
"""
#This will hold the ploting functions!
import matplotlib.pyplot as plt
import numpy as np

def linear_ploter(sol,eigs,t,real,alpha):
    plt.figure(1)
    plt.clf()
    colors = ['red','blue','green','black','orange','purple']
    markersizesrg=0
    markersizeeig=1

    plt.plot(t, sol[:, 0], color=colors[0], label='Hs[0,0]',marker='1',ms=markersizesrg)
    plt.plot(t, sol[:, 7], color=colors[1], label='Hs[1,1]',marker='1',ms=markersizesrg)
    plt.plot(t, sol[:, 14], color=colors[2], label='Hs[2,2]',marker='1',ms=markersizesrg)
    plt.plot(t, sol[:, 21], color=colors[3], label='Hs[3,3]',marker='1',ms=markersizesrg)
    plt.plot(t, sol[:, 28], color=colors[4], label='Hs[4,4]',marker='1',ms=markersizesrg)
    plt.plot(t, sol[:, 35], color=colors[5], label='Hs[5,5]',marker='1',ms=markersizesrg)

    plt.plot(t, eigs[:,0],'--', color=colors[0],linestyle='None', marker='2',ms=markersizeeig)
    plt.plot(t, eigs[:,1],'--', color=colors[1],linestyle='None', marker='2',ms=markersizeeig)
    plt.plot(t, eigs[:,2],'--', color=colors[2],linestyle='None', marker='2',ms=markersizeeig)
    plt.plot(t, eigs[:,3],'--', color=colors[3],linestyle='None', marker='2',ms=markersizeeig)
    plt.plot(t, eigs[:,4],'--', color=colors[4],linestyle='None', marker='2',ms=markersizeeig)
    plt.plot(t, eigs[:,5],'--', color=colors[5],linestyle='None', marker='2',ms=markersizeeig)
    if real ==True:
        plt.title('Real_alpha=%2.1f'%(alpha))
#        plt.savefig("srg_pairing_Real_alpha%2.1f.png"%(int(alpha*10)), bbox_inches="tight", pad_inches=0.05)
#        plt.savefig("srg_pairing_Real_alpha%2.1f.png"%(int(g*10)), bbox_inches="tight", pad_inches=0.05)
    elif real ==False:
        plt.title('Imag_alpha=%2.1f'%(alpha))
#        plt.savefig("srg_pairing_imag_alpha%2.1f.png"%(int(alpha*10)), bbox_inches="tight", pad_inches=0.05)
    plt.legend(loc='best')
    plt.grid(True)
#    plt.xscale('symlog')
    plt.xlabel('g/delta')
    plt.ylabel('Energy')
    plt.show()

def log_ploter(sol,eigs,t,real):
    plt.figure(1)
    plt.clf()
    colors = ['red','blue','green','black','orange','purple']
    markersizesrg=0
    markersizeeig=2    

    plt.semilogy(t, sol[:, 0], color=colors[0], label='Hs[0,0]',marker='1',ms=markersizesrg)
    plt.semilogy(t, sol[:, 7], color=colors[1], label='Hs[1,1]',marker='1',ms=markersizesrg)
    plt.semilogy(t, sol[:, 14], color=colors[2], label='Hs[2,2]',marker='1',ms=markersizesrg)
    plt.semilogy(t, sol[:, 21], color=colors[3], label='Hs[3,3]',marker='1',ms=markersizesrg)
    plt.semilogy(t, sol[:, 28], color=colors[4], label='Hs[4,4]',marker='1',ms=markersizesrg)
    plt.semilogy(t, sol[:, 35], color=colors[5], label='Hs[5,5]',marker='1',ms=markersizesrg)

    plt.semilogy(t, eigs[:,0],'--', color=colors[0],linestyle='None', label='E0',marker='2',ms=markersizeeig)
    plt.semilogy(t, eigs[:,1],'--', color=colors[1],linestyle='None', label='E1',marker='2',ms=markersizeeig)
    plt.semilogy(t, eigs[:,2],'--', color=colors[2],linestyle='None', label='E2',marker='2',ms=markersizeeig)
    plt.semilogy(t, eigs[:,3],'--', color=colors[3],linestyle='None', label='E3',marker='2',ms=markersizeeig)
    plt.semilogy(t, eigs[:,4],'--', color=colors[4],linestyle='None', label='E4',marker='2',ms=markersizeeig)
    plt.semilogy(t, eigs[:,5],'--', color=colors[5],linestyle='None', label='E5',marker='2',ms=markersizeeig)
    if real ==True:
        plt.title('Real')
    elif real ==False:
        plt.title('Imag')
#    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    