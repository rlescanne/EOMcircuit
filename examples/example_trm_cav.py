# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 18:18:46 2019

@author: Zaki
"""

from EOMcircuit.functions import Dipole, Node, Hole, Circuit, Representation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

plt.close('all')

import scipy.constants
h = scipy.constants.h
hbar = h/2/np.pi
phi0=hbar/(2*scipy.constants.e)
sci = hbar/phi0**2

W = Dipole('W')

_ = None

N = Node('') # expect this for dummy nodes
G = Node('G')
E = Node('E')

F = Hole('F', 0)

# storage params
wa = 2*np.pi*4
Za = 50

# LC cav
Ca = Dipole('Ca', 1/wa/Za)
La = Dipole('La', Za/wa)
Ta = Dipole('Ta', (np.pi/wa, Za), ground = G)

# trm params
wb = 2*np.pi*6
Zb = 50

# TRM dipoles (asymm)
Cb = Dipole('Cb', 1/wb/Zb)
Jb = Dipole('Jb', Zb/wb)

# capa coup
Cc = Dipole('Cc', Cb.val/10)

# capa input
Cg = Dipole('Cg', Cb.val/10)

R0 = Dipole('R0', 5000)


circuit = [[ N, W, N,Cc, N, W, N,Cg, N],
           [Jb, _,Cb, _,La, _,Ca, _,R0],
           [ G, W, N, _, G, W, N, _, G]]

circuit = [[ N, W, N,Cc, N,Ta, N,Cg, N],
           [Jb, _,Cb, _, _, _, _, _,R0],
           [ G, W, N, _, _, _, _, _, G]]

circuit = [[ G,Cc, N,Ta, N, R0, G],
           [ _, _, _, _, _, _, _],
           [ _, _, _, _, _, _, _]]

# circuit = [[ G,Cc, N, W, N, R0, G],
#            [ _, _,La, _,Ca, _, _],
#            [ _, _, G, W, N, _, _]]


fig_circuit, ax_circuit = plt.subplots()
c = Circuit(circuit) # parse circuit and much more is done
c.plot(ax_circuit)
print('### Circuit plotted ###')

fig = plt.figure(figsize=(9,6))
gs = gridspec.GridSpec(ncols=2, nrows = 2)
gs.update(left=0.10, right=0.95, wspace=0.0, hspace=0.1, top=0.95, bottom =0.05)
ax_eom = fig.add_subplot(gs[0, :])

omegas = np.linspace(2*2*np.pi, 10*2*np.pi, 201)
kappas = np.linspace(1e-6*2*np.pi, 1e-1*2*np.pi, 201)

# omega_center = 3.9169965323536404*2*np.pi
# kappa_center = 0.024938670630199132*2*np.pi
# omegas = np.linspace(omega_center-0.04, omega_center+0.04, 501)
# kappas = np.linspace(kappa_center-0.04, kappa_center+0.04, 501)

guesses = [4*2*np.pi]
c.rep_AC.display_eom(ax_eom, omegas, kappas=kappas, guesses=guesses, log_kappa=True)#, kappas=kappas, guesses=guesses)#kappas=kappas
eig_omegas, eig_phizpfs = c.rep_AC.solve_EIG(guesses)

#ax_eom.vlines(eig_omegas, np.amin(to_plot), np.amax(to_plot), lw=1, color='r')

### plot modes

ax0 = fig.add_subplot(gs[1, :])
c.plot(ax0)
c.rep_AC.plot_phi(ax0, 10*np.real(eig_phizpfs[0]), offset=0.3, color='C0') # 4* -> magnification for plot only
# c.rep_AC.plot_phi(ax0, 10*np.real(eig_phizpfs[1]), offset=0.5, color='C1') # 4* -> magnification for plot only
# c.rep_AC.plot_phi(ax0, 10*np.real(eig_phizpfs[1]), offset=0.7, color='C2') # 4* -> magnification for plot only

#ax1 = fig.add_subplot(gs[1, 1])
#c.plot(ax1)
#c.rep_AC.plot_phi(ax0, 40*eig_phizpfs[1], offset=0.5, color='C1')


### sweep
# phiext = np.linspace(0, 2*np.pi, 201)
# list_eig_omegas, list_eig_phizpfs = c.sweep(F, phiext, guesses = guesses)

# fig, ax = plt.subplots(2)
# ax[0].plot(phiext, np.real(list_eig_omegas[0])/2/np.pi)
# ax[0].plot(phiext, np.real(list_eig_omegas[1])/2/np.pi)
# ax[1].plot(phiext, np.real(list_eig_omegas[0])/(2*np.imag(list_eig_omegas[0])))
# ax[1].plot(phiext, np.real(list_eig_omegas[1])/(2*np.imag(list_eig_omegas[1])))
# ax[1].set_xlabel(r'$\varphi_{ext}$')
# ax[0].set_ylabel(r'freq (GHz)')
# ax[1].set_ylabel(r'Q')
#ax.plot(phiext, np.real(list_eig_omegas[1])/2/np.pi)

#fig, ax = plt.subplots()
#ax.plot(phiext, np.imag(list_eig_omegas[0])/2/np.pi)
#ax.plot(phiext, np.imag(list_eig_omegas[1])/2/np.pi)
#


