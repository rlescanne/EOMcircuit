#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 10:43:17 2020

@author: julescraquelin
"""

from EOMcircuit.functions import Dipole, Node, Hole, Circuit, Representation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize, newton, root

plt.close('all')

import scipy.constants
h = scipy.constants.h
pi=np.pi
hbar = h/2/np.pi
phi0=hbar/(2*scipy.constants.e)
sci = hbar/phi0**2

#connecting and wires
W = Dipole('W')
_ = None
N = Node('') # expect this for dummy nodes
G = Node('G')




if 1 : 
    desired_output = np.array([6, 5, 0.2, 0.003, 0.003]) #what do we aim
    #input_param0 = np.array([2*np.pi*6, 2*np.pi*5, 100, 1/(2*np.pi*6)/100/10, 1/(2*np.pi*6)/100/10])
    #input_param0 = np.array([2*np.pi*7, 2*np.pi*6, 200, 1/(2*np.pi*6)/200/10, 1/(2*np.pi*6)/200/10])
    #input_param0 = np.array([2*np.pi*(6.12), 2*np.pi*(5.48), 355, 1/(2*np.pi*(6.27))/465/10, 1/(2*np.pi*(6.27))/541/10])
    #input_param0 = np.array([6.05688463e+00, 5.22853426e+00, 6.09325351e+01, 1.14777012e-06, 7.59989362e-07])*2*pi
    #input_param0 = np.array([6.09837119e+00, 5.22934561e+00, 6.09420074e+01, 1.15169586e-06, 1.91192618e-06])*2*np.pi
    input_param0 = np.array([3.83171962e+01, 3.28569475e+01, 3.82909925e+02, 7.23631853e-06, 1.20129865e-05])

    wa = input_param0[0]
    Za = 50
    wb = input_param0[1]
    Zb = input_param0[2]
    Z0 = 50
    
    
    
    Cb = Dipole('Cb', 1/wb/Zb)
    Jb = Dipole('Jb', Zb/wb)
    Cc = Dipole('Cc', input_param0[3])
    Cg = Dipole('Cg', input_param0[4])
    R0 = Dipole('R0', Z0)
    Ca = Dipole('Ca', 1/wa/Za)
    La = Dipole('La', Za/wa)


if 0 : 
    desired_output = np.array([6, 5, 4, 0.2, 0.003, 0.003, 0.0003]) #what do we aim
    input_param0 = np.array([3.83171962e+01, 3.28569475e+01, 2.73569475e+01, 151.09925, 0.000163265306122449, 0.0001142857142857143, 8e-05])
    
    
   # input_param0 = np.array([2*np.pi*(6.2), 2*np.pi*(5.2), 2*np.pi*(4.2), 110, 1/(2*np.pi*6)/100/10, 1/(2*np.pi*6)/100/10, 1/(2*np.pi*4)/50/10])
    #[wa, wb, wc, Zb, Cc, Cg, Cc2]


    wa = input_param0[0]
    Za = 50
    wb = input_param0[1]
    Zb = input_param0[3]
    Z0 = 50
    
    wm = input_param0[2]
    Zm = 50
    
    
    Cb = Dipole('Cb', 1/wb/Zb)
    Jb = Dipole('Jb', Zb/wb)
    Cc = Dipole('Cc', input_param0[4])
    Cg = Dipole('Cg', input_param0[5])
    R0 = Dipole('R0', Z0)
    Ca = Dipole('Ca', 1/wa/Za)
    La = Dipole('La', Za/wa)
 
    Cm = Dipole('Cm', 1/wm/Zm)
    Lm = Dipole('Lm', Zm/wm)
    Cc2 = Dipole('Cc2', input_param0[6])


EJ=phi0**2*wb/Zb/h*2*np.pi #junction's energy in GHz

#Circuit's geometry
#circuit=[[N,W,N,Cg,N],
#         [Jb,_,Cb,_,R0],
 #        [G,W,N,_,G]]
if 1 : 
    circuit = [[ N, W, N,Cc, N, W, N,Cg, N],
           [Jb, _,Cb, _,La, _,Ca, _,R0],
           [ G, W, N, _, G, W, N, _, G]]
    dipoles = np.array([Cb, Jb, Ca, La, Cc, Cg, R0])

if 0 : 
    circuit = [[ N, W, N, Cc2, N, W, N,Cc, N, W, N,Cg, N],
               [Lm, _, Cm, _, Jb, _,Cb, _,La, _,Ca, _,R0],
               [G, W, N, _, G, W, N, _, G, W, N, _, G]]
    
    
    dipoles = np.array([Cb, Jb, Ca, La, Cc, Cg, R0, Cm, Lm, Cc2])





#fig_circuit, ax_circuit = plt.subplots()
c = Circuit(circuit, verbose = False) # parse circuit and much more is done
    
#c.plot(ax_circuit)
#print('### Circuit plotted ###')

#fig = plt.figure(figsize=(9,6))
#gs = gridspec.GridSpec(ncols=2, nrows = 2)
#gs.update(left=0.10, right=0.95, wspace=0.0, hspace=0.1, top=0.95, bottom =0.05)
#ax_eom = fig.add_subplot(gs[0, :])

#omegas = np.linspace(2*2*np.pi, 8*2*np.pi, 501)
#kappas = np.linspace(1e-6*2*np.pi, 1e-1*2*np.pi, 51)

if 1 : 
    guesses = [wa, wb]
    #c.rep_AC.display_eom(ax_eom, omegas, kappas=kappas, guesses=guesses)#, kappas=kappas, guesses=guesses)#kappas=kappas
    eig_omegas, eig_phizpfs = c.rep_AC.solve_EIG(guesses)
if 0 :  
    guesses = [wa, wb, wm] 
    eig_omegas, eig_phizpfs = c.rep_AC.solve_EIG(guesses)

  

#ax_eom.vlines(eig_omegas, np.amin(to_plot), np.amax(to_plot), lw=1, color='r')

### plot modes

#ax0 = fig.add_subplot(gs[1, :])
#c.plot(ax0)
#c.rep_AC.plot_phi(ax0, 10*np.real(eig_phizpfs[0]), offset=0.3, color='C0') # 4* -> magnification for plot only
#c.rep_AC.plot_phi(ax0, 10*np.real(eig_phizpfs[1]), offset=0.5, color='C1') # 4* -> magnification for plot only


#Ker, Cross-Ker frequencies and Charging energy
#Ka = EJ/2*np.cos(eig_phizpfs[0][2])*eig_phizpfs[0][2]**4 # ker frequency (mode a) in GHz
#Kb = EJ/2*np.cos(eig_phizpfs[1][2])*eig_phizpfs[1][2]**4 # ker frequency (mode b) in GHz

#Ec=phi0**2/2/Zb*wb*eig_phizpfs[0][1]**2/h # capa energy in GHz

#Ctotb = Cb.val + Cg.val +Cc.val
#Ctota = Ca.val + Cg.val +Cc.val
#ECb = scipy.constants.e**2/Ctotb/2/h # charging energy in mode b
#ECa = scipy.constants.e**2/Ctota/2/h # charging energy in mode a

#Khi=np.sqrt(Ka*Kb*2)



def circuit_outputs (input_param, circuit, dipoles, plot = True) :
    #input_param should be a ndarray like [wa, wb, Zb, Cc, Cg]
    #it returns an array [wcav, wtr, kappacav, kertr, khi
    if 1 : 
        if type(input_param) != np.ndarray or len(input_param) != 5:
            print('### circuit_outputs : ERROR ###')
            print('input_param must be a ndarray with 5 elements')
            return 

    if 1 : 
        wa = input_param[0]
        wb = input_param[1]
        Zb = input_param[2]   
        #print ('#### INPUT_PARAM')
        #print (input_param/2/np.pi)
        #print('#### INPUT_PARAM')
        
        Za = 50
        EJ = phi0**2*wb/Zb/h*2*np.pi #junction's energy in GHz
    
    
        dipoles[0].val = 1/wb/Zb
        dipoles[1].val = Zb/wb
        dipoles[2].val = 1/Za/wa
        dipoles[3].val = Za/wa
        dipoles[4].val = input_param[3]
        dipoles[5].val = input_param[4]

        guesses = [wa, wb]
        eig_omegas, eig_phizpfs = c.rep_AC.solve_EIG(guesses, verbose = False)
        
    if 0 : 
        wa = input_param[0]
        wb = input_param[1]
        wm = input_param[2] 
        
        Zb = input_param[3]
        Za = 50
        Zm = 50
        
        EJ = phi0**2*wb/Zb/h*2*np.pi #junction's energy in GHz
        
        dipoles[0].val = 1/wb/Zb # Cb
        dipoles[1].val = Zb/wb # Jb
        dipoles[2].val = 1/Za/wa # Ca
        dipoles[3].val = Za/wa # La
        dipoles[4].val = input_param[4] #Cc
        dipoles[5].val = input_param[5]#Cg
        dipoles[7].val = 1/wm/Zm # Cm
        dipoles[8].val = Zm/wm # Lm
        dipoles[9].val = input_param[6]#Cc2
        
        guesses = [wa, wb, wm]
        eig_omegas, eig_phizpfs = c.rep_AC.solve_EIG(guesses, verbose = False)

        
        
        





    if plot : 
        omegas = np.linspace(2*2*np.pi, 8*2*np.pi, 501)
        kappas = np.linspace(1e-6*2*np.pi, 1e-1*2*np.pi, 51)
        fig = plt.figure(figsize=(9,6))
        gs = gridspec.GridSpec(ncols=2, nrows = 2)
        gs.update(left=0.10, right=0.95, wspace=0.0, hspace=0.1, top=0.95, bottom =0.05)
        ax_eom = fig.add_subplot(gs[0, :])
    
        c.rep_AC.display_eom(ax_eom, omegas, kappas=kappas, guesses=guesses)#, kappas=kappas, guesses=guesses)#kappas=kappas
    

        ax0 = fig.add_subplot(gs[1, :])
        c.plot(ax0)
        c.rep_AC.plot_phi(ax0, 10*np.real(eig_phizpfs[0]), offset=0.3, color='C0') # 4* -> magnification for plot only
        c.rep_AC.plot_phi(ax0, 10*np.real(eig_phizpfs[1]), offset=0.5, color='C1') # 4* -> magnification for plot only
    
    #Ker, Cross-Ker frequencies and Charging energy
    if 1 : 
        Kcav = EJ/2*np.cos(eig_phizpfs[0][2])*eig_phizpfs[0][2]**4 # ker frequency (mode a) in GHz
        Ktr = EJ/2*np.cos(eig_phizpfs[1][2])*eig_phizpfs[1][2]**4 # ker frequency (mode b) in GHz
        Khi_ab = np.sqrt(2*Ktr*Kcav) # Cross ker frequency in GHz??????? le 2 devrait etre hors de la sqrt

    if 0 : 
        Kcav = EJ/2*np.cos(eig_phizpfs[0][2])*eig_phizpfs[0][2]**4 # ker frequency (mode a) in GHz
        Ktr = EJ/2*np.cos(eig_phizpfs[1][2])*eig_phizpfs[1][2]**4 # ker frequency (mode b) in GHz
        Khi_ab = np.sqrt(2*Ktr*Kcav) # Cross ker frequency in GHz??????? le 2 devrait etre hors de la sqrt
        Km = EJ/2*np.cos(eig_phizpfs[2][2])*eig_phizpfs[2][2]**4 # ker frequency (mode c = m) in GHz
        Khi_bc = np.sqrt(2*Ktr*Km)
    


    
    if 1 : 
        return np.array([np.real(eig_omegas[0]), np.real(eig_omegas[1]), Ktr, Khi_ab, 2*np.imag(eig_omegas[0])])/2/np.pi
    
 
    if 0 : 
        return np.array([np.real(eig_omegas[0]), np.real(eig_omegas[1]), np.real(eig_omegas[2]), Ktr, Khi_ab, 2*np.imag(eig_omegas[0]), Khi_bc])/2/np.pi


def optimize_components (desired_output, input_param0):
    # If several conditions satisfy the desired outputs, this function may return several lists
    # desired_output should be a list like with 5 elements [wcav, wtr, kappacav, kertr, khi]
    # input_param0 should be like [wa, wb Zb Cc Cg]

    if 1 : 
        if type(desired_output) != np.ndarray or len(desired_output) != 5 or type(input_param0) != np.ndarray or len(input_param0) != 5 :
            print('### optimize_components : ERROR ###')
            print('desired_outputs must be a ndarray with 5 elements')
            print('input_param0 must be a ndarray with 5 elements')
            return 
  
    to_minimize = lambda input_param, circuit, dipoles : np.abs(\
        desired_output - circuit_outputs(input_param, circuit, dipoles, plot =False))
    #to_minimize = lambda input_param : np.abs(\
      #   [ a[0] - a[1] for a in zip(desired_output, circuit_outputs(input_param)) ])#if we used lists...
            #to_minimize is a function that take for argument input_param
            #it gives the difference between desired_out and circuit_outputs(input_param)
            
    return root(to_minimize, input_param0, args=(circuit, dipoles)).x #root finds zeros of a vectorial function 
    

#params = optimize_components(desired_output, input_param0)

result = circuit_outputs(input_param0, circuit, dipoles, plot = False)
print(result)
