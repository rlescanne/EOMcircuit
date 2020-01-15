# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 18:18:46 2019

@author: Zaki
"""

from functions import Dipole, Node, Hole, Circuit, Representation
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import os
import fig_file as f

save_dir='/Users/Raphael/Documents/These/Chapters/THEO/fig'

plt.close('all')

ccav = np.array([0/255,185/255,255/255])
cbuf = np.array([255/255,100/255,100/255])

aspect_fig = 1/3
fig = f.create_fig(aspect_fig)

gs = GridSpec(ncols=1, nrows = 1)#, width_ratios=[1.1,1])
gs.update(left=-0.05, right=1.05, wspace=0.2, hspace=0, top=1.13, bottom =-0.15)

ax1 = fig.add_subplot(gs[0,0])
#ax2 = fig.add_subplot(gs[0,1])

I1 = Dipole('I1', 0.1, text=r'$I_\mathrm{DC}$')
AC = Dipole('AC', 0.1, text=r'$I_\mathrm{AC}$')

Lb = Dipole('Lb', 1, phi=r'$\varphi_{L_b}$', color=cbuf)
J1 = Dipole('J1', 1, phi=r'$\varphi_{J_1}$', color=cbuf)
J2 = Dipole('J2', 1, phi=r'$\varphi_{J_2}$', color=cbuf)
Lm = Dipole('Lm', 1, phi=r'$\varphi_{M}$', color=cbuf)
Cb = Dipole('Cb', 1, phi=r'$\varphi_{C_b}$', color=cbuf)

Cc = Dipole('Cc', 1, phi=r'$\varphi_{C_c}$', color=cbuf, colorbis=ccav)
Cg = Dipole('Cg', 1, phi=r'$\varphi_{C_g}$', color=ccav, colorbis='k')

R0 = Dipole('R0', 1, phi=r'$\varphi_{Z_0}$')

La = Dipole('La', 1, phi=r'$\varphi_{L_a}$', color=ccav)
Ca = Dipole('Ca', 1, phi=r'$\varphi_{C_a}$', color=ccav)

W = Dipole('W')
Wa = Dipole('Wa', color=ccav)
Wb = Dipole('Wb', color=cbuf)

_ = None

N = Node('') # expect this for dummy nodes
G = Node('G')
Ga = Node('Ga', color=ccav)
Gb = Node('Gb', color=cbuf)

M = Hole('Ma', 1, text=r'$M$')#, text=r'$D$')

circuit = [[ N,I1, N, _, _, _, _, _, _, _, _, _, _],
           [ W, _, W, _, _, _, _, _, _, _, _, _, _],
           [ N,Lm, N, Wb, N,Cc, N, Wa, N,Cg, N, W, N],
           [J1, _,J2, _,Cb, _,Ca, _,La, _,R0, _,AC],
           [ N, Wb,Gb, Wb, N, _, N, Wa,Ga, _, G, W, N]]
#           [ _, _,  _,  _,  _, _, _, _,  _, _,  _],
#           [ N, M1, N,  M2,N, _, _, _,  _, _,  _],
#           [ W, _,  W,  _,  W, _, _, _,  _, _,  _],
#           [ N, AC1, G,AC2, N, _, _, _,  _, _,  _]]


c = Circuit(circuit) # parse circuit and much more is done
c.plot(ax1, debug=False, lw_scale=0.8)





#fig.text(0.005, 0.97, '(a)', transform=fig.transFigure, va='top', ha = 'left', weight='bold')
#fig.text(0.505, 0.97, '(b)', transform=fig.transFigure, va='top', ha = 'left', weight='bold')

fig.savefig(os.path.join(save_dir, 'fig_example_debug.pdf'), dpi=300, type='pdf')
