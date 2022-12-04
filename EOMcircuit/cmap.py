#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 14:19:21 2020

@author: Raphael
"""
import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')

def color(Z, power=0.5):
    color = cmo.phase((np.angle(Z)+np.pi)/2/np.pi)
    color = color[:,:,:3]
    mag = np.abs(Z)
    mag = mag/np.amax(mag)
    color = np.moveaxis(color, -1, 0)
    color =  (1-(1-color*mag**power)*(1-mag)**power)
    color = np.moveaxis(color, 0, -1)
    return color

if __name__=='__main__':

    x = np.linspace(-10, 10, 11)
    y = np.linspace(-10, 10, 11)

    X, Y = np.meshgrid(x, y)

    angle = np.angle(X+1j*Y)
    mag = np.abs(X+1j*Y)

    fig, ax = plt.subplots(2)



    ax[0].imshow(color(X+1j*Y), origin='lower', extent=[(3*x[0]-x[1])/2, (3*x[-1]-x[-2])/2, (3*y[0]-y[1])/2, (3*y[-1]-y[-2])/2])
    ax[0].set_aspect('equal')
    # ax[1].pcolor(X, Y, mag)
    # ax[1].set_aspect('equal')

    import numpy as np
    import matplotlib.pyplot as plt

    #make some sample data
    r, g = np.meshgrid(np.linspace(0,255,100),np.linspace(0,255,100))
    b=255-r

    #this is now an RGB array, 100x100x3 that I want to display
    rgb = np.array([r,g,b]).T

    color_tuple = rgb.transpose((1,0,2)).reshape((rgb.shape[0]*rgb.shape[1],rgb.shape[2]))/255.0

    fig, ax = plt.subplots()
    m = ax.pcolormesh(r, color=color_tuple, linewidth=0)
    # m.set_array(None)
    plt.show()
