#------------------------------------------------------------------------------
# srg_pairing.py
#
# author:   H. Hergert 
# version:  1.1.0
# date:     Nov 18, 2016
# 
# tested with Python v2.7
# 
# Solves the pairing model for four particles in a basis of four doubly 
# degenerate states by means of a Similarity Renormalization Group (SRG)
# flow.
#
#------------------------------------------------------------------------------


import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import SymLogNorm, Normalize
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable

import numpy as np
from numpy import array, dot, diag, reshape, arange
from scipy.linalg import eigvals
from scipy.integrate import odeint

#------------------------------------------------------------------------------
# plot helpers
#------------------------------------------------------------------------------
def myLabels(x, pos):
    '''format tick labels using LaTeX-like math fonts'''
    return '$%s$'%x

def myPlotSettings(ax, formatter):
    '''save these settings for use in other plots'''
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis='both',which='major',width=1.5,length=8)
    ax.tick_params(axis='both',which='minor',width=1.5,length=5)
    ax.tick_params(axis='both',width=2,length=10,labelsize=20)
    for s in ['left', 'right', 'top', 'bottom']:
        ax.spines[s].set_linewidth(2)
    ax.set_xlim([0.0007,13])  
    return

#------------------------------------------------------------------------------
# plot routines
#------------------------------------------------------------------------------
def plot_diagonals(data, eigenvalues, flowparams, delta, g):
    '''plot eigenvalues and diagonals'''
    dim       = len(data)
    formatter = FuncFormatter(myLabels)
    markers   = ['o' for i in range(dim)]
    cols      = ['blue', 'red', 'purple', 'green', 'orange', 'deepskyblue']

    # diagonals vs. eigenvalues on absolute scale
    fig, ax = plt.subplots()
    for i in range(dim):
        plt.semilogx(flowparams, [eigenvalues[i] for e in range(flowparams.shape[0])], color=cols[i], linestyle='solid')
        plt.semilogx(flowparams, data[i], color=cols[i], linestyle='dashed', marker=markers[i], markersize=10)

    myPlotSettings(ax, formatter)

    plt.savefig("srg_pairing_diag_delta%2.1f_g%2.1f.pdf"%(delta, g), bbox_inches="tight", pad_inches=0.05)
    plt.show()

    # difference between diagonals and eigenvalues
    fig, ax = plt.subplots()
    for i in range(dim):
        plot_diff = plt.semilogx(flowparams, data[i]-eigenvalues[i], color=cols[i], linestyle='solid', marker=markers[i], markersize=10)

    myPlotSettings(ax, formatter)

    plt.savefig("srg_pairing_diag-eval_delta%2.1f_g%2.1f.pdf"%(delta, g), bbox_inches="tight", pad_inches=0.05)
    plt.show()
    return

#------------------------------------------------------------------------------
# plot matrix snapshots
#------------------------------------------------------------------------------
def plot_snapshots(Hs, flowparams, delta, g):
    fig  = plt.figure(1, (10., 5.))
    grid = AxesGrid(fig, 111,                       # similar to subplot(111)
                     nrows_ncols=(2, Hs.shape[0]/2),  # creates grid of axes
                     axes_pad=0.25,                 # pad between axes in inch.
                     label_mode='L',                # put labels on left, bottom
                     cbar_mode='single',            # one color bar (default: right of last image in grid)
                     cbar_pad=0.20,                 # insert space between plots and color bar
                     cbar_size='10%'                # size of colorbar relative to last image
                     )

    # create individual snapshots - figures are still addressed by single index,
    # despite multi-row grid
    for s in range(Hs.shape[0]):
        img = grid[s].imshow(Hs[s], 
                            cmap=plt.get_cmap('RdBu_r'),                                  # choose color map
                            interpolation='nearest',       
                            norm=SymLogNorm(linthresh=1e-10,vmin=-0.5*g,vmax=10*delta),   # normalize 
                            vmin=-0.5*g,                                                  # min/max values for data
                            vmax=10*delta
                            )

        # tune plots: switch off tick marks, ensure that plots retain aspect ratio
        grid[s].set_title('$s=%s$'%flowparams[s])
        grid[s].tick_params(      
  
        bottom='off',      
        top='off',
        left='off',      
        right='off'
        )
  
        grid[s].set_xticks([0,1])
        grid[s].set_yticks([0,1])
        grid[s].set_xticklabels(['$0$','$1$'])
        grid[s].set_yticklabels(['$0$','$1$'])

        cbar = grid.cbar_axes[0]
        plt.colorbar(img, cax=cbar, 
          ticks=[ -1.0e-1, -1.0e-3, -1.0e-5, -1.0e-7, -1.09e-9 , 0., 
                 1.0e-9, 1.0e-7, 1.0e-5, 1.0e-3, 0.1, 10.0]
        )

        cbar.axes.set_yticklabels(['$-10^{-1}$', '$-10^{-3}$', '$-10^{-5}$', '$-10^{-7}$', 
                             '$-10^{-9}$', '$0.0$', '$10^{-9}$', '$10^{-7}$', '$10^{-5}$', 
                             '$10^{-3}$', '$10^{-1}$', '$10$'])
        cbar.set_ylabel('$\mathrm{[a. u.]}$') 


        plt.savefig("srg_pairing_delta%2.1f_g%2.1f.pdf"%(delta, g), bbox_inches="tight", pad_inches=0.05)
    plt.show()

    return

#------------------------------------------------------------------------------
# SRG 
#------------------------------------------------------------------------------

# Hamiltonian for the pairing model
def Hamiltonian(delta,g):

  H = array(
      [[2*delta-g,    -0.5*g,     -0.5*g,     -0.5*g,    -0.5*g,          0.],
       [   -0.5*g, 4*delta-g,     -0.5*g,     -0.5*g,        0.,     -0.5*g ], 
       [   -0.5*g,    -0.5*g,  6*delta-g,         0.,    -0.5*g,     -0.5*g ], 
       [   -0.5*g,    -0.5*g,         0.,  6*delta-g,    -0.5*g,     -0.5*g ], 
       [   -0.5*g,        0.,     -0.5*g,     -0.5*g, 8*delta-g,     -0.5*g ], 
       [       0.,    -0.5*g,     -0.5*g,     -0.5*g,    -0.5*g, 10*delta-g ]]
    )

  return H

# commutator of matrices
def commutator(a,b):
  return dot(a,b) - dot(b,a)
a=0
b=0
c=0
d=0
# derivative / right-hand side of the flow equation
def derivative(yreal, t, real, yimag, dim):
  
  # reshape the solution vector into a dim x dim matrix
  if real == True:
      Hr  = reshape(yreal, (dim, dim))

      Hi  = reshape(yimag, (dim, dim))
  elif real ==False:
      Hr  = reshape(yimag, (dim, dim))

      Hi  = reshape(yreal, (dim, dim))
  
  H  = Hr+1.0j*Hi
    
  # extract diagonal Hamiltonian...
  Hd  = diag(diag(H))

  # ... and construct off-diagonal the Hamiltonian
  Hod = H-Hd
  
  # calculate the generator
  if real == True:
      eta = a*c*commutator(Hd.real, Hod.real)-b*d*commutator(Hd.imag,Hod.imag)
      # dH is the derivative in matrix form 
      dH  = commutator(eta, Hr)
  elif real == False:
      eta= a*d*commutator(Hd.real,Hod.imag)+b*c*commutator(Hd.imag,Hod.real)
      # note zero if any arguement is zero
      # dH is the derivative in matrix form 
      dH  = commutator(eta, Hi)
      # convert dH into a linear array for the ODE solver
  dydt = reshape(dH, -1)
  
  return dydt

def check_eigenvalues(H0, Hsr, Hsi, epsilon):
    status=[]
    i=0
    eigenvalue_differences=[]
    while i<len(Hsr):
        eigenvalues=abs(eigvals(H0)-(eigvals(Hsr[-1])+1j*eigvals(Hsi[-1])))
        k=0
        current_status=[]
        while k<len(eigenvalues):
            if eigenvalues[k].real<epsilon and eigenvalues[k].imag<epsilon:
                current_status.append(True)
            else:
                current_status.append(False)
            k=k+1
        status.append(current_status)
        eigenvalue_differences.append(eigenvalues)
        i=i+1
    print status

def test_generators():
    scores=[]
    for a in arange(0,1,0.1):
        for b in arange(0,1,0.1):
            for c in arange(0,1,0.1):
                for d in arange(0,1,0.1):
                    scores.append(calc())
    reals=[]
    imags=[]
    reals.append(x[0] for x in scores)
    imags.append(x[-1] for x in scores)
    print([min(list(reals)), reals.index(min(list(reals)))])
    print([min(list(imags)),imags.index(min(list(imags)))])
    
def calc():
  g     = 0.5
  delta = 1

  H0    = array([[1, 1+1j],[1+1j,2+1j]])
  dim   = H0.shape[0]
  
  # calculate exact eigenvalues
  eigenvalues = eigvals(H0)
#  print(eigenvalues)
  
  # turn initial Hamiltonian into a linear array
  y0  = reshape(H0, -1)                 
  
  # flow parameters for snapshot images
  flowparams = array([0.,0.001,0.01,0.05,0.1, 1., 5., 10.])
  
  # integrate flow equations - odeint returns an array of solutions,
  # which are 1d arrays themselves
  ysreal  = odeint(derivative,  y0.real,flowparams, args=(True,y0.imag,dim,))
  
  ysimag  = odeint(derivative, y0.imag, flowparams, args=(False,y0.real,dim,))

  # reshape individual solution vectors into dim x dim Hamiltonian
  # matrices
  Hsr  = reshape(ysreal, (-1, dim,dim))
  Hsi  = reshape(ysimag, (-1, dim,dim))
  Hs= Hsr+1j*Hsi
  #print(Hs)

#  print diag(Hs[-1])
  
  return([diag(Hs[-1]).real-eigenvalues.real,diag(Hs[-1]).imag-eigenvalues.imag])
  
  #print (eigvals(Hsr[-1])+1j*eigvals(Hsi[-1]))
  # note that the above gives a result consistent with the initial hamiltonian
  # whereas the below does not
  # print eigvals(Hs[-1])

  data = []
  for h in Hsr:
    data.append(diag(h).real)
  data = zip(*data)
  
#  plot_diagonals(data, eigenvalues.real, flowparams, delta, g)
#  plot_snapshots(Hsr, flowparams, delta, g)
  
  data = []
  for h in Hsi:
    data.append(diag(h))
  data = zip(*data)
  
#  plot_diagonals(data, eigenvalues.imag, flowparams, delta, g)
#  plot_snapshots(Hsi, flowparams, delta, g)
  
  # stricter test on consistency of eigenvalues during SRG 
  # gives True if both the real and imaginary part agree with the eigvals of
  # H0 within epsilon
  # check_eigenvalues(H0,Hsr,Hsi,0.1)
#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------
def main():
    test_generators


#------------------------------------------------------------------------------
# make executable
#------------------------------------------------------------------------------
if __name__ == "__main__": 
  main()