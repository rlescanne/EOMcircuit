#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:44:46 2019

@author: Raphael
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as nl
from scipy.optimize import minimize, newton, root
import scipy.constants
from math import factorial
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle
from sympy.parsing import sympy_parser
from scipy.linalg import inv

DIRECTIONS = [(0, 1), (1,0), (0,-1), (-1, 0)]

from .cmap import color

e = scipy.constants.e
hbar = scipy.constants.hbar
conv_L = 2*np.pi*hbar/(4*e*np.pi)**2 # conversion factor from LJ nH to EJ GHz
                                     # LJ = conv_L/EJ
conv_C = e**2/2/(hbar*2*np.pi) # conversion factor from C nF to EC GHz
                               # C = conv_c/EC
                               
plt.close('all')


C = 'cap'
R = 'res'
L = 'ind'
J = 'jct'
T = 'trl'
W = 'wir'
A = 'AC'
V = 'vol'
I = 'amp'

E = 'equ'
G = 'grd'
N = 'nod'

F = 'flu'
M = 'mut'
#
#L1 = 'L1'
#C1 = 'C1'
#Cc = 'Cc'
#Z0 = 'Z0'
#R1 = 'R1'
#R2 = 'R2'
#R3 = 'R3'

def simplify_arith_expr(expr):
    try:
        out = repr(sympy_parser.parse_expr(str(expr)))
        return out
    except Exception:
        print("Couldn't parse", expr)
        raise

def name_elements(elts): # gives relevant name to repeating dipoles
    parents = [elt.parent for elt in elts]
    parents_number = {}
    for parent in set(parents):
        count = parents.count(parent)
        if count==1:
            parents_number[parent] = -1
        else:
            parents_number[parent] = count-1

    for elt in elts[::-1]:
        parent = elt.parent
        if parents_number[parent]==-1:
            pass
        else:
            elt.name = elt.name+'%d'%(parents_number[parent])
            parents_number[parent] -= 1

def find_ker(m): #implements Gauss elimination to find the loops in the circuit
    indices = []
    for ii, line in enumerate(m): # nodes that are connected to nothing, prevent from working
        if not(np.any(line)):
            indices.append(ii)
    m = np.delete(np.copy(m), indices, axis=0)

    diag = np.eye(len(m[0]))
    M = np.concatenate((m, diag)).T
    rr=0
    for jj in range(len(m.T)):### modified CB & RL for jj in range(len(m)):
    #        print('')
        kk = np.argmax(np.abs(M[rr:,jj]))+rr
#        print(kk)
        if M[kk, jj]!=0:
            M[kk] = M[kk]/M[kk,jj]
            line = np.copy(M[kk])
#            print(line)
            M[kk] = np.copy(M[rr])
            M[rr] = np.copy(line)
#            print(M)
            for ii in range(rr+1, len(M)):
                M[ii] = M[ii]-M[ii, jj]*line
            rr+=1

    m_trig = (M.T)[:len(m)]
    M_trig = (M.T)[len(m):]

    ker = []
    for jj, col in enumerate(m_trig.T):
        if np.sum(col**2)==0:
            ker.append((M_trig.T)[jj])

    return np.array(ker)

def equal_float(float1, float2, margin=1e-8):
    if float1!=0 and float2!=0:
        rel_diff = abs((float1-float2)/float1)
    elif float1==0:
        rel_diff = float2
    elif float2==0:
        rel_diff = float1
    else:
        rel_diff=0
    if rel_diff<margin:
        return True
    else:
        return False

def crossing_edge_along_y(coor_point, edges_list):
    #take the coordinate of the point and the list of edges and returns boolean if the point is on one of the edge (we just check the crossing along the x)
    for coor_center in edges_list: #edge is a tuple of coordinates
        if all(coor_center==coor_point):
            return True
    else:
        return False


#def point_in_a_loop(coor_point, edges_loop):
#    #take the coordinate of the point and the list of the edges of the loop and returns boolean
#    #premier pas de 1
##        print("########## Point in a loop #######")
#        inside_circuit=True
#        counter=0
#        actual_coor= coor_point
##        print('x dim', Coor.x_dim)
##        print('y dim', Coor.y_dim)
##        print('coor', actual_coor)
#        actual_coor+=(0,1) #first step of 1 to go on a edge
##        print('coor', actual_coor)
#        while inside_circuit :
#            if crossing_edge_along_y(actual_coor, edges_loop):
#                counter+=1
##            print('count',counter)
##            print('coor', actual_coor)
#            try:
#                actual_coor+=(0,2) #we shift of 2 along x at each step to go from edge to edge
#            except:
#                inside_circuit=False
#        return(counter%2==1)

def hole_in_loop(hole, dipole_loop):
    inside_circuit=True
    counter=0
    actual_coor= np.copy(hole.coor)
    edges_loop = [dipole.center for dipole in dipole_loop if not dipole.horiz]
    actual_coor+=np.array([-1, 0]) #first step of 1 to go on a edge
    while inside_circuit:
        if crossing_edge_along_y(actual_coor, edges_loop):
            counter+=1
        inside_circuit = actual_coor[0]>0 # when x<=0 outside circuit
        actual_coor+=np.array([-1, 0])
    return (counter%2==1)



def clockwise(edges_loop, orientation_edges): # only means something if the loop is closed
                           # returns whatever when loop in not closed
                           # to determine orientation we look at most right vertical edge
    edge_max = edges_loop[0]
    print(edges_loop)
    y_max = edge_max[0][1] #whatever
    for ii, edge in enumerate(edges_loop):
        coord_node0=edge[0]
        coord_node1=edge[1]
        if coord_node0[1]==coord_node1[1]: #along x, constant y
            y =  coord_node0[1]
            if y>=y_max:
                y_max=y
                print(ii)
                print(orientation_edges[ii])
                if orientation_edges[ii]==1:
                    edge_max = edge
                else:
                    edge_max = (edge[1], edge[0])

    if edge_max[0][0]>edge_max[1][0]:
        print('trigo')
        return 1
    else:
        print('antitrigo')
        return -1

#    #runs ok if the edges are along a (O,x,y) grid
#    #function to determine if a the loop is clockwise or anticlockwise. We take the convention to put +phi ext to clockwise loop and -phiext to anticlockwise loop.
#    #edges_loop is a list of tuples (=edges). Each edge is a tuple of tuples (coordinates of nodes).
#    #find the edges that have the maximum y coordinate and that are just along y
#    print(edges_loop)
#    orientation=-1
#    edges_max_along_y=[]
#    y_max=0
#    for edge in edges_loop:
#        coord_node0=edge[0]
#        coord_node1=edge[1]
#        if (coord_node0[1]==coord_node1[1]) :
#            if coord_node0[1]>=y_max:
#                y_max=coord_node0[1]
#                edges_max_along_y.append(edge)
#    #the only case when this list is empty is when the loop has no edge parallel to the y axis (possible in the logical case)
#    if len(edges_max_along_y)==0:
#        edges_max_along_x=[]
#        x_max=0
#        for edge in edges_loop:
#            coord_node0=edge[0]
#            coord_node1=edge[1]
#            if (coord_node0[0]==coord_node1[0]) :
#                if coord_node0[0]>=x_max:
#                    x_max=coord_node0[0]
#                    edges_max_along_x.append(edge)
#     #if the ymax edges go from small x to large x : clockwise
#    #we can take any edge in edges_max_along_y
#    if len(edges_max_along_y)!=0:
#        edge=edges_max_along_y[0]
#        coord_node0=edge[0]
#        coord_node1=edge[1]
#        if (coord_node1[0]-coord_node0[0])<0 :
#            orientation=+1
#    elif len(edges_max_along_x)!=0:
#        edge=edges_max_along_x[0]
#        coord_node0=edge[0]
#        coord_node1=edge[1]
#        if (coord_node0[1]-coord_node1[0])<0 :
#            orientation=+1
#    else:
#        raise ValueError('This is not a loop aligned on the (x,y) grid')
#
#    return(orientation)
#orientation =+1 if clockwise
#orientation =-1 if anticlockwise


def contact_edges(edge1, edge2): #edge is a tuple of coordinates (tuples)
    edge1[0]==edge2[0] or edge1[0]==edge2[1] or edge1[1]==edge
    return(res)


def tuple_list(n):
    _tuple_list = [tuple(range(n))]
    for ii in range(n-1):
        temp_list = list(range(n))
        temp_list[n-ii-2]=n-1
        temp_list[n-1]=n-ii-2
        _tuple_list.append(tuple(temp_list))
    return _tuple_list


def get_factor(which):
    factor = 1
    for elt in set(which):
        factor = factor*factorial(which.count(elt))
    return factor

def line_is_zero(M, ind):
    #functions that returns True if the line ind of array M False otherwise
    res_bool=True
    for ind_col in range(M.shape[1]):
        res_bool=res_bool and (M[ind, ind_col]==0)
    return(res_bool)

def complete_lines_with_zeros(A_T):
    n_nodes = A_T[-1].shape[0]
    for ii, A_T_line in enumerate(A_T):
        shape_diff = n_nodes-A_T_line.shape[0]
        if shape_diff>0:
            A_T[ii]= np.concatenate((A_T_line, np.zeros(shape_diff)))
    return np.array(A_T)

def remove_wires(D_vec, A_T, nodes_type, nodes):
    count = 0
    for ii, D_val in enumerate(D_vec):
        ii=ii-count
        if D_val=='wire':
            D_vec = np.delete(D_vec, ii)
            A_line = A_T[ii]
#                print("Pline",P_line)
            ends = np.where(A_line)[0] #return the indices of the ends of the wire
#                print('equiv indices', equiv_indices)
#                print('nodes_type', nodes_type,'\n')
            if nodes_type[ends[1]]=='excitation':
                # by default delete the second one
                # except if it's the excitation port
                if nodes_type[ends[0]]=='ground': # only the first one can be ground by construction
                    raise ValueError('You shorted excitation to ground')
                ends = ends[::-1]
            nodes.pop(ends[1])
            nodes_type.pop(ends[1])
            A_T[:,ends[0]] = A_T[:,ends[0]]+A_T[:,ends[1]]
            A_T = np.delete(A_T, ii, axis=0)
            A_T = np.delete(A_T, ends[1], axis=1)
            count += 1
    return D_vec, A_T, nodes_type, nodes

def add_dipole_for_trl(D_vec, A_T):

    count = 0
    for ii, D_val in enumerate(D_vec):
        ii=ii+count
        if D_val[0]=='T':
            D_vec[ii] = D_val+'_m'
            D_vec = np.insert(D_vec, ii+1, D_val+'_p')
            A_line = A_T[ii]
            ends = np.where(A_line)[0] #return the indices of the ends of the trl

            if ends[0]==0: # or if nodes_type[ends[0]]=='ground'
                # trl is shorted to ground
                raise NotImplementedError('You shorted a transmission line to ground \
                                 this is not implemented yet')
            else:
                A_line_m = np.zeros(np.shape(A_line))
                A_line_m[0] = -1
                A_line_m[ends[0]] = 1
                A_line_p = np.zeros(np.shape(A_line))
                A_line_p[0] = -1
                A_line_p[ends[1]] = 1
            A_T = np.delete(A_T, ii, axis=0)
            A_T = np.insert(A_T, ii, A_line_p, axis=0)
            A_T = np.insert(A_T, ii, A_line_m, axis=0)
            count += 1
    return D_vec, A_T


class Coor(tuple): #this class allows to avoid problems of looping indexes : it raises error if we have an index outside the circuit (instead of returning to the begining)
    # should be set to the size of circuit
    x_dim = 1
    y_dim = 1
    def __new__(cls, *args):
        if len(args)==1:
            return super(Coor, cls).__new__(cls, args[0])
        else:
            args = (arg for arg in args)
            return super(Coor, cls).__new__(cls, args)

    def __add__(self, other):
        sum = []
        for ii, jj in zip(self, other):
            sum.append(ii+jj)
        if sum[0]<0 or sum[0]>=self.y_dim:
            raise ValueError('y coor out of circuit')
        if sum[1]<0 or sum[1]>=self.x_dim:
            raise ValueError('x coor out of circuit')
        return(Coor(sum))

    def __floordiv__(self, other):
        div = []
        for ii in self:
            div.append(ii//other)
        return(Coor(div))

    def __truediv__(self, other):
        div = []
        for ii in self:
            div.append(ii/other)
        return(Coor(div))

def array_to_plot(coor):
    jj, ii = coor
    return np.array([ii, -jj])

def plot_to_array(coor):
    x, y = coor
    return (int(y), -x)

def pt_to_xy(points):
    x = []
    y = []
    for point in points:
        x.append(point[0])
        y.append(point[1])
    return x, y

def rem_digits(string):
    new_string = ''
    for char in string:
        if char.isdigit():
            break
        else:
            new_string+=char
    return new_string

def complete_with_identity(mat):
    shape = mat.shape
    maygo = np.logical_not(np.logical_not(mat)) #line should so that diagonal is non zero
    ret_mat = np.eye(shape[1])
    ret_mat_c = np.eye(shape[1])
    choices = choice(maygo)
    for ii, elt in enumerate(choices):
        ret_mat[elt] = mat[ii]

    ret_mat_c = np.delete(ret_mat_c, choices, axis=1)
    if nl.matrix_rank(ret_mat)!=shape[1]:
        raise ValueError('Assigning the constrains did not produce a invertible matrix')
    return ret_mat, ret_mat_c, choices

def choice(maygo):
    shape = maygo.shape
    choices = [0 for ii in range(shape[0])]
    for ii in range(len(maygo)):
        # choose first the one with the least choices
        list_where_to_choose = [elt if elt>0 else np.inf for elt in np.sum(maygo, axis=0)]
        where_to_choose = np.argmin(list_where_to_choose)
        # choose
        which = np.argmax(maygo[:, where_to_choose])
        choices[which] = where_to_choose
        # remove what is done
        maygo[which] = np.array([False for ii in range(shape[1])])
        maygo[:, where_to_choose] = np.array([False for ii in range(shape[0])])
    return choices

def associated_string(arglist,type_string):
    print(arglist)
    if arglist==[]:
        to_return='(0)'
    else:
        if type_string=='flux':
            to_return = '('
            for elt in arglist:
                to_return+=str(elt)+'+'
            to_return=to_return[:-1]+')'
        elif type_string=='I_DC':
            to_return = '('
            for elt in arglist:
                to_return+= elt[1]+str(elt[0]) #elt[1] is a string with sign and elt[0] is the source
            to_return+=')'
    print('res')
    print(type_string)
    print(to_return)
    return to_return

#def associated_string(F_list):
#    print('associated string')
#    print(F_list)
#    if F_list==[]:
#        to_return='(0)'
#    else:
#        to_return = '('
#        for F in F_list:
#            to_return+=str(F)+'+'
#        to_return=to_return[:-1]+')'
#    print('res')
#    print(to_return)
#    return to_return

def matrix_product_str(mat, vec):
    # mat is float
    # vec is string
    to_return = []
    for line in mat:
        to_str = ''
        for ii, elt in enumerate(line):
            to_str+=str(elt)+'*'+vec[ii]+'+'
        to_return.append(simplify_arith_expr(to_str[:-1]))
    return to_return

class Dipole():

    dipole_list = []
    dipole_list_DC = []
    dipole_list_AC = []
    dipole_val = {}
    dipoles_nodes = {}
    n_wire = 0

    def __init__(self, name, val=None, text=None, color='k', phi=None, colorbis=None, size=1, parent=None, ground=None, plotted=True, circuit=None):
        # kind should be in 'C', 'L', 'R', 'J', 'T'
        self.name = name
        self.assign_kind()
        self.ground=ground
        if (self.kind==T) and (ground is None):
            raise Exception('You should specify a ground for transmission lines')

        self.parent = parent
        self.children = []
        self.circuit=circuit
        self.val = val # for kind 'T', val should be a tuple with (t, Z0), for kind 'J', val should be a tuple with (L, fp)
#        if not(val is None):
#            if self.kind!=T:
#                Dipole.dipole_val[self.name]=val
#            else:
#                Dipole.dipole_val[self.name+'_t']=val[0]
#                Dipole.dipole_val[self.name+'_Z']=val[1]
#                Dipole.dipole_val[self.name+'_L']=val[0]*val[1]
        if text is None:
            self.text=name
        else:
            self.text=text
        self.color=color
        self.colorbis=colorbis
        self.size = size
        self.phi=phi
        if phi is not None:
            self.arrow = True
            self.phi_text = phi
        else:
            self.arrow = False
            self.phi_text = ''
        self.start = None
        self.end = None
        self.horiz = None
        self.center = None
        self.phi_DC = 0 # should change with solve_DC
        self.plotted = plotted

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    def val(self):
        return self.__val

    @val.setter
    def val(self, val):
        # Dipole.dipole_val[self.name]=val
        for dipole in self.children:
            dipole.val = val
        self.__val = val
        if self.circuit is not None:
            self.circuit.dipoles_val[self.name] = val

    def assign_kind(self):
        name = self.name
        if name[0]=='C':
            self.kind = C
        if name[0]=='R':
            self.kind = R
        if name[0]=='L':
            self.kind = L
        if name[0]=='J':
            self.kind = J
        if name[0]=='T':
            self.kind = T
        if name[0]=='W':
            self.kind = W
        if name[0]=='A':
            self.kind = A
        if name[0]=='V':
            self.kind = V
        if name[0]=='I':
            self.kind = I

    def copy_dipole(self, suff='', plotted=True, circuit=None):
        child = Dipole(self.name+suff, val=self.val, text=self.text, color=self.color, phi=self.phi, colorbis=self.colorbis, parent=self, ground=self.ground, plotted=plotted, circuit=circuit)
        self.children.append(child)
        return child

    def assign_coor(self, circuit, start, end): # used at parsing, jump when only need to create fictious dipole for AC/DC Repr
        # this should create a dipole with the val linked to the one that
        # was used for creation

        dipole = self.copy_dipole(circuit=circuit)
        circuit.dipoles.append(dipole)

        dipole.start = start
        dipole.end = end
        dipole.center = (start+end)/2
        dipole.horiz = start[1]==end[1]

        return dipole

    def fill_dipoles(self):

        if self.kind in [L, J, W, I, V]:
            self.circuit.dipoles_DC.append(self)

        if self.kind in [T]:
            dipole_s = self.copy_dipole(suff='s', plotted=False, circuit=self.circuit)
            dipole_e = self.copy_dipole(suff='e', plotted=False, circuit=self.circuit)
            ground = self.ground
            if self.horiz:
                ground_s_coor = self.start+np.array([0.1, -0.5])
                ground_e_coor = self.end+np.array([-0.1, -0.5])
            else:
                ground_s_coor = self.start+np.array([-0.5, 0.1])
                ground_e_coor = self.end+np.array([-0.5, -0.1])
            ground.assign_coor(self.circuit, ground_s_coor, plotted=False) # won't be plotted
            ground.assign_coor(self.circuit, ground_e_coor, plotted=False) # won't be plotted

            dipole_s.start = ground_s_coor
            dipole_s.end = self.start
            dipole_s.center = (ground_s_coor+self.start)/2
            dipole_e.start = ground_e_coor
            dipole_e.end = self.end
            dipole_e.center = (ground_e_coor+self.end)/2
            dipole_s.horiz = not self.horiz
            dipole_e.horiz = not self.horiz

            self.circuit.dipoles.append(dipole_s)
            self.circuit.dipoles.append(dipole_e)
            self.circuit.dipoles_AC.append(dipole_s)
            self.circuit.dipoles_AC.append(dipole_e)

        if self.kind in [C, R, L, J, A, I]:
            self.circuit.dipoles_AC.append(self)

    @classmethod
    def empty_dipole_list(cls):
        Dipole.dipole_list = []
        Dipole.dipole_list_DC = []
        Dipole.dipole_list_AC = []
        Dipole.dipoles_nodes = {}
        Dipole.n_wire = 0


    def plot(self, ax, lw_scale):
        if self.plotted:
            plt.rc('lines', color=self.color, lw=2*lw_scale, solid_capstyle='round', dash_capstyle='round')

            if self.horiz:
                ha = 'center'
                va = 'top'
            else:
                ha = 'right'
                va = 'center'

            x_c, y_c = self.center
            name = self.name
            text = self.text
            if self.kind==W:
    #            print(self.name)
    #            print('color')
    #            print(self.color)
                ax.plot(*pt_to_xy([self.start, self.end]), color=self.color)
            else:
                if not self.arrow or (name != text):
                    if self.horiz:
                        if self.kind==C:
                            ax.text(x_c, y_c-0.3, text, va=va, ha=ha)
                        elif self.kind in [A, V, I]:
                            ax.text(x_c, y_c-0.4, text, va=va, ha=ha)
                        else:
                            ax.text(x_c, y_c-0.25, text, va=va, ha=ha)
                    else:
                        if self.kind==C:
                            ax.text(x_c-0.3, y_c, text, va=va, ha=ha)
                        elif self.kind in [A, V, I]:
                            ax.text(x_c-0.4, y_c, text, va=va, ha=ha)
                        else:
                            ax.text(x_c-0.25, y_c, text, va=va, ha=ha)
                if self.arrow:
                    self.plot_arrow(ax, text=self.phi_text)
                if self.kind==L:
                    self.draw_ind(ax, lw_scale=lw_scale)
                if self.kind==C:
                    self.draw_capa(ax, lw_scale=lw_scale)
                if self.kind==R:
                    self.draw_res(ax, lw_scale=lw_scale)
                if self.kind==J:
                    self.draw_jct(ax, lw_scale=lw_scale)
                if self.kind==T:
                    self.draw_trl(ax, lw_scale=lw_scale)
                if self.kind==A:
                    self.draw_AC(ax, lw_scale=lw_scale)
                if self.kind==V:
                    self.draw_V(ax, lw_scale=lw_scale)
                if self.kind==I:
                    self.draw_I(ax, lw_scale=lw_scale)
#        self.plot_arrow(ax)

    def plot_arrow(self, ax, size=1, offset=0.3, color='k', text=''):
        if np.abs(size)>1e-4:
            nodes = self.circuit.dipoles_nodes[self]
            node_start = nodes[0].coor
            node_end = nodes[1].coor
            if self.horiz: # horizontal arrow
                if node_start[0]<node_end[0]: # right arrow
                    line = np.array([[-size/2, offset], [size/2, offset]])
                else: # left arrow
                    line = np.array([[size/2, offset], [-size/2, offset]]) # left arrow
                center_text = self.center + np.array([0,offset+np.sign(offset)*0.1])
            else: # vertical arrow
                if node_start[1]<node_end[1]: # top arrow
                    line = np.array([[offset, -size/2], [offset, size/2]])
                else: # down arrow
                    line = np.array([[offset, size/2], [offset, -size/2]])
                center_text = self.center + np.array([offset+np.sign(offset)*0.1,0])
            line = line+self.center

    #        line_xy = pt_to_xy(line)
    #        ax.plot(*line_xy, color='red', lw=4)
#            print(size)
            ax.annotate('', xy=line[1], xytext=line[0], arrowprops=dict(arrowstyle="->, head_length=%f,head_width=%f"%(np.abs(size/3), np.abs(size/3)), color=color, lw=2))
            if self.horiz:
                if offset>0:
                    va='bottom'
                else:
                    va='top'
                ax.text(*(center_text), text, va=va, ha='center')
            else:
                if offset>0:
                    ha='left'
                else:
                    ha='right'
                ax.text(*(center_text), text, va='center', ha=ha)
    def draw_res(self, ax, lw_scale=1):
        size = 0.15

        theta = np.linspace(-4, 4, 17)
        y_theta = np.sin(theta*np.pi)
        x_coil = np.array([theta/12])
        y_coil = np.array([y_theta*size])

        _line1 = np.array([[-1, 0], [x_coil[0,0], 0]])
        _line2 = np.array([[x_coil[0,-1], 0], [1, 0]])
        _res = np.concatenate((x_coil, y_coil)).T

        if not self.horiz:
            _res = _res[:, ::-1]
            _line1 = _line1[:, ::-1]
            _line2 = _line2[:, ::-1]

        _res = _res + self.center
        _line1 = _line1 + self.center
        _line2 = _line2 + self.center

        res = Line2D(*pt_to_xy(_res))
        line1 = Line2D(*pt_to_xy(_line1))
        line2 = Line2D(*pt_to_xy(_line2))

        artists = [res, line1, line2]
        for art in artists:
            ax.add_artist(art)

    def draw_ind(self, ax, lw_scale=1):
        size = 0.15

        theta = np.linspace(np.pi,5*2*np.pi, 101)
        x_coil = np.cos(theta)*size/2+size*theta/8
        y_coil = -np.array([np.sin(theta)*size])
        start = x_coil[0]
        end = x_coil[-1]
        x_coil = np.array([x_coil-(start+end)/2])

        _line1 = np.array([[-1, 0], [x_coil[0,0], 0]])
        _line2 = np.array([[x_coil[0,-1], 0], [1, 0]])
        _res = np.concatenate((x_coil, y_coil)).T

        if not self.horiz:
            _res = _res[:, ::-1]
            _line1 = _line1[:, ::-1]
            _line2 = _line2[:, ::-1]

        _res = _res + self.center
        _line1 = _line1 + self.center
        _line2 = _line2 + self.center

        res = Line2D(*pt_to_xy(_res))
        line1 = Line2D(*pt_to_xy(_line1))
        line2 = Line2D(*pt_to_xy(_line2))

        artists = [res, line1, line2]
        for art in artists:
            ax.add_artist(art)

    def draw_capa(self, ax, lw_scale=1):
        size = 0.2*self.size
    #    plt.rc('lines', color='k', lw=2)

        _plate1 = np.array([[-size/3, -size], [-size/3, size]])
        _plate2 = np.array([[size/3, -size], [size/3, size]])
        _line1 = np.array([[-1, 0], [-size/3, 0]])
        _line2 = np.array([[size/3, 0], [1, 0]])
        if not self.horiz:
            _plate1 = _plate1[:, ::-1]
            _plate2 = _plate2[:, ::-1]
            _line1 = _line1[:, ::-1]
            _line2 = _line2[:, ::-1]
        _plate1 = _plate1 + self.center
        _plate2 = _plate2 + self.center
        _line1 = _line1 + self.center
        _line2 = _line2 + self.center
        
        plate1 = Line2D(*pt_to_xy(_plate1), lw=4*lw_scale*self.size, solid_capstyle='butt')

        line1 = Line2D(*pt_to_xy(_line1))
        if self.colorbis is not None:
            plate2 = Line2D(*pt_to_xy(_plate2), lw=4*lw_scale*self.size, solid_capstyle='butt', color=self.colorbis)
            line2 = Line2D(*pt_to_xy(_line2), color=self.colorbis)
        else:
            plate2 = Line2D(*pt_to_xy(_plate2), lw=4*lw_scale*self.size, solid_capstyle='butt')
            line2 = Line2D(*pt_to_xy(_line2))

        artists = [line1, line2, plate1, plate2]
        for art in artists:
            ax.add_artist(art)

    def draw_AC(self, ax, lw_scale=1):
        size = 0.3
        width = 0.066
    #    plt.rc('lines', color='k', lw=2)
        x = np.linspace(-np.pi, np.pi, 21)
        sine =-np.sin(x)*size/4
        _line_sin = np.stack((x/np.pi*size/2, sine)).T

        _circle_center = np.array([0,0])
        _line1 = np.array([[-1, 0], [-size, 0]])
        _line2 = np.array([[size, 0], [1, 0]])
        _arrow_line = np.array([[size*2/3,0], [-size*2/3,0]])
        _arrow_head = np.array([[size*2/3-width,width], [size*2/3,0], [size*2/3-width,-width]])
        if not self.horiz:
            _arrow_line = _arrow_line[:, ::-1]
            _arrow_head = _arrow_head[:, ::-1]
            _line1 = _line1[:, ::-1]
            _line2 = _line2[:, ::-1]
        _arrow_line += self.center
        _arrow_head += self.center
        _line1 = _line1 + self.center
        _line2 = _line2 + self.center
        _circle_center = _circle_center +self.center
        _line_sin = _line_sin +  self.center

        arrow_line = Line2D(*pt_to_xy(_arrow_line))
        arrow_head = Line2D(*pt_to_xy(_arrow_head))
        line1 = Line2D(*pt_to_xy(_line1))
        line2 = Line2D(*pt_to_xy(_line2))
        line_sin = Line2D(*pt_to_xy(_line_sin))

        circle = Circle(_circle_center, radius=size, fc='none', ec = self.color, lw=2*lw_scale)
        artists = [line1, line2, line_sin, circle]#, arrow_line, arrow_head]
        for art in artists:
            ax.add_artist(art)

    def draw_V(self, ax, lw_scale=1):
        size = 0.3
        width = 0.066
    #    plt.rc('lines', color='k', lw=2)
#        x = np.linspace(-np.pi, np.pi, 21)
#        sine = np.sin(x)*size/4
#        _line_sin = np.stack((x/np.pi*size/2, sine)).T
#
        _circle_center = np.array([0,0])
        _line1 = np.array([[-1, 0], [-size, 0]])
        _line2 = np.array([[size, 0], [1, 0]])
        _plus1 = np.array([[size*1/2-width,0], [size*1/2+width,0]])
        _plus2 = np.array([[size*1/2,-width], [size*1/2,width]])
        _minus = np.array([[-size*1/2-width,0], [-size*1/2+width,0]])
        if not self.horiz:
            _plus1 = _plus1[:, ::-1]
            _plus2 = _plus2[:, ::-1]
            _line1 = _line1[:, ::-1]
            _line2 = _line2[:, ::-1]
            _minus = np.array([[-width,-size*1/2], [width,-size*1/2]])
        _plus1 += self.center
        _plus2 += self.center
        _minus += self.center
        _line1 = _line1 + self.center
        _line2 = _line2 + self.center
        _circle_center = _circle_center +self.center

        line1 = Line2D(*pt_to_xy(_line1))
        line2 = Line2D(*pt_to_xy(_line2))
        plus1 = Line2D(*pt_to_xy(_plus1), solid_capstyle='butt')
        plus2 = Line2D(*pt_to_xy(_plus2), solid_capstyle='butt')
        minus = Line2D(*pt_to_xy(_minus), solid_capstyle='butt')

        circle = Circle(_circle_center, radius=size, fc='none', ec = self.color, lw=2*lw_scale)
        artists = [line1, line2, circle, plus1, plus2, minus]
        for art in artists:
            ax.add_artist(art)

    def draw_I(self, ax, lw_scale=1):
        size = 0.3
        width = 0.066

    #    plt.rc('lines', color='k', lw=2)
#        x = np.linspace(-np.pi, np.pi, 21)
#        sine = np.sin(x)*size/4
#        _line_sin = np.stack((x/np.pi*size/2, sine)).T
#
        _circle_center = np.array([0,0])
        _line1 = np.array([[-1, 0], [-size, 0]])
        _line2 = np.array([[size, 0], [1, 0]])
        _arrow_line = np.array([[size*2/3,0], [-size*2/3,0]])
        _arrow_head = np.array([[size*2/3-width,width], [size*2/3,0], [size*2/3-width,-width]])
        if not self.horiz:
            _arrow_line = _arrow_line[:, ::-1]
            _arrow_head = _arrow_head[:, ::-1]
            _line1 = _line1[:, ::-1]
            _line2 = _line2[:, ::-1]
        _arrow_line += self.center
        _arrow_head += self.center
        _line1 += self.center
        _line2 += self.center
        _circle_center = _circle_center +self.center

        arrow_line = Line2D(*pt_to_xy(_arrow_line))
        arrow_head = Line2D(*pt_to_xy(_arrow_head))
        line1 = Line2D(*pt_to_xy(_line1))
        line2 = Line2D(*pt_to_xy(_line2))
        circle = Circle(_circle_center, radius=size, fc='none', ec = self.color, lw=2*lw_scale)
        artists = [line1, line2, arrow_line, arrow_head, circle]
        for art in artists:
            ax.add_artist(art)

    def draw_jct(self, ax, lw_scale=1):
        size = 0.15
        
        _linea = np.array([[-size, -size], [-size, size]])
        _lineb = np.array([[-size, -size], [size, -size]])
        _linec = np.array([[size, size], [size, -size]])
        _lined = np.array([[size, size], [-size, size]])
        _cross1 = np.array([[-size, -size], [size, size]])
        _cross2 = np.array([[-size, size], [size, -size]])
        _line1 = np.array([[-1, 0], [-size, 0]])
        _line2 = np.array([[size, 0], [1, 0]])
        if not self.horiz:
            _line1 = _line1[:, ::-1]
            _line2 = _line2[:, ::-1]
            
        _cross1 = _cross1 + self.center
        _cross2 = _cross2 + self.center
        _linea += self.center
        _lineb += self.center
        _linec += self.center
        _lined += self.center
        
        _line1 = _line1 + self.center
        _line2 = _line2 + self.center

        cross1 = Line2D(*pt_to_xy(_cross1), lw=4*lw_scale, solid_capstyle='butt')
        cross2 = Line2D(*pt_to_xy(_cross2), lw=4*lw_scale, solid_capstyle='butt')
        line1 = Line2D(*pt_to_xy(_line1))
        line2 = Line2D(*pt_to_xy(_line2))

        linea = Line2D(*pt_to_xy(_linea))
        lineb = Line2D(*pt_to_xy(_lineb))
        linec = Line2D(*pt_to_xy(_linec))
        lined = Line2D(*pt_to_xy(_lined))

        artists = [line1, line2, cross1, cross2, linea, lineb, linec, lined]
        for art in artists:
            ax.add_artist(art)

    def draw_trl(self, ax, lw_scale=1):
        size = 0.15
        _line0 = np.array([[-1, 0], [1, 0]])
        _line1 = np.array([[-1+2*size, size], [1-2*size, size]])
        _line2 = np.array([[-1+2*size, -size], [1-2*size, -size]])

        if not self.horiz:
            _line0 = _line0[:, ::-1]
            _line1 = _line1[:, ::-1]
            _line2 = _line2[:, ::-1]
        _line0 = _line0 + self.center
        _line1 = _line1 + self.center
        _line2 = _line2 + self.center

        line0 = Line2D(*pt_to_xy(_line0))
        line1 = Line2D(*pt_to_xy(_line1))
        line2 = Line2D(*pt_to_xy(_line2))

        artists = [line1, line2, line0]
        for art in artists:
            ax.add_artist(art)

    def plot_phi(self, ax):
        pass


class Node():

    node_list = []
    eq_node_list = []
    eq_node_dict = {}
    n_node = {}
    # node can have several coordinates

    def __init__(self, name, color='k', parent=None, plotted=True, circuit=None):
        self.name = name
        self.assign_kind()
        self.parent = parent
        self.children = []
        self.coor = None
        self.color= color
        self.plotted=plotted
        self.circuit=circuit

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def eq_node(self):
        return self.circuit.eq_nodes_dict[self]

    def copy_node(self, plotted=True, circuit=None):
        child = Node(self.name, color=self.color, parent=self, plotted=plotted, circuit=circuit)
        self.children.append(child)
        return child

    def assign_coor(self, circuit, coor, plotted=True):
        # create a node each time one assigns a coordinate, maybe not for excitations
        node = self.copy_node(plotted=plotted, circuit=circuit)
        circuit.nodes.append(node)
        circuit.eq_nodes_dict[node]=node
        node.coor = coor
        return node

    def assign_kind(self):
        name = self.name
        if name=='':
            self.kind = N
            self.text = None
        else:
            self.kind = E
            self.text = name

    @classmethod
    def print_node_list(cls):
        print('Nodes:')
        for node in cls.node_list:
            to_print = '%s - %s'%(node, node.coor)
            print(to_print)
        print('')


    @classmethod
    def empty_node_list(cls):
        cls.node_list = []
        cls.eq_node_list = []
        cls.eq_node_dict = {}
        cls.n_node = {}


    def plot(self, ax, lw_scale=1):
        coor = self.coor
        #ax.plot(*coor, '.', color='k')
        if self.plotted:
#            ax.text(*self.coor, self.text,
#                verticalalignment='center',
#                horizontalalignment='center',
#                bbox={'facecolor':'white'})
#            if self.kind == E:
#                size = 0.2
#                line = np.array([[0, 0], [size, size]])
#                line = line + coor
#                ax.plot(*pt_to_xy(line), color='k')
#                ax.text(*(line[1]), self.text, va = 'bottom', ha='left')
            if self.name[0] == 'G':
                size = 0.2
                size_small = 0.05
                line = np.array([[0, 0], [size, -size], [size+size_small, -size+size_small], [size-size_small, -size-size_small]])#, [size-size_small,  -size-size_small], [size, -size]])
#                line = np.array([[0, 0], [size, -size], [size+size_small, -size+size_small], [size+size_small, -size-size_small], [size-size_small,  -size-size_small], [size, -size]])
                dot = np.array([[size+size_small, -size-size_small]])
                line = line + coor
                dot = dot + coor
                ax.plot(*pt_to_xy(line), color=self.color)
                ax.plot(*pt_to_xy(dot), '.', color=self.color, markersize=2*lw_scale)
        ax.text(coor[0], coor[1], self.name, color='k', fontsize=6, va='bottom', ha='left')


class Hole():
    hole_list = []
    hole_val = {}
    def __init__(self, name, val, text=None, parent=None, color=None, circuit=None):
        print(Hole.hole_val)
        self.name = name
        self.parent = parent
        self.children = []
        self.circuit=circuit
        self.val = val
        self.color=color
        if text is None:
            self.text=name
        else:
            self.text=text
        self.assign_kind()
#        Hole.hole_val[name]=val
        print(Hole.hole_val)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    def val(self):
        return self.__val

    @val.setter
    def val(self, val):
        for hole in self.children:
            hole.val = val
        self.__val = val
        if self.circuit is not None:
            self.circuit.holes_val[self.name] = val

    def assign_kind(self):
        name = self.name
        if name[0]=='F':
            self.kind = F
        if name[0]=='M':
            self.kind = M

    def copy_hole(self, circuit=None):
        child = Hole(self.name, self.val, text=self.text, color=self.color, parent=self, circuit=circuit)
        self.children.append(child)
        return child

    def assign_coor(self, circuit, coor):
        # create a hole each time one assigns a coordinate
        hole = self.copy_hole(circuit=circuit)
        circuit.holes.append(hole)
        hole.coor = coor
        return hole

    def plot(self, ax, lw_scale=1):
        if self.kind==F:
            size = 0.2
            ax.plot(*self.coor, '.', color='k')
            ax.plot(*self.coor, 'o', color='k', markerfacecolor='none', markersize=10)
    #        TODO understand the bug below
    #        circle = Circle(self.coor, radius=size, fill=True, ec='k', lw=2)
    #        print('print'+str(circle))
    #        ax.add_artist(circle)
            coor_text = self.coor-np.array([0, size])
            if self.text is None:
                ax.text(*coor_text, (r'$\varphi_{ext,%s}$'%(self.name[1:])), verticalalignment='top', horizontalalignment='center')
            else:
                ax.text(*coor_text, self.text, verticalalignment='top', horizontalalignment='center')

        elif self.kind==M:
            width = 0.1
            length = 0.5
            line1 = np.array([[-width, -length], [-width, length]])+self.coor
            line2 = np.array([[width, -length], [width, length]])+self.coor

            ax.plot(*pt_to_xy(line1), color='k', lw=2*lw_scale)
            ax.plot(*pt_to_xy(line2), color='k', lw=2*lw_scale)

            coor_text = self.coor+np.array([0,-length-0.1])
            ax.text(*coor_text, self.text, color='k', va='top', ha='center')

    @classmethod
    def empty_hole_list(cls):
        Hole.hole_list = []

#class Source_DC():
#    source_DC_list = []
#    source_DC_val = {}
#    source_DC_nodes ={}
#    def __init__(self, name, val, text=None):
#        self.name = name
#        self.val = val
#        if text is None:
#            self.text=name
#        else:
#            self.text=text
#        self.start = None
#        self.end = None
#        self.center = None
#        Source_DC.source_DC_val[name]=val
#        print(Source_DC.source_DC_val)
#
#
#    def __str__(self):
#        return self.name
#
#    def __repr__(self):
#        return self.name
#
#    @property
#    def val(self):
#        return self.__val
#
#    @val.setter
#    def val(self, val):
#        Source_DC.source_DC_val[self.name]=val
#        self.__val = val
#
##    def assign_kind(self):
##        name = self.name
##        if name[0]=='F':
##            self.kind = F
##        if name[0]=='M':
##            self.kind = M
#   # later maybe if the class source contais DC sources and AC sources
#
#    def assign_coor(self, start, end):
#        source=self
#        source.start = start
#        source.end = end
#        source.center = (start+end)/2
#        Source_DC.source_DC_list.append(self)
#
#    def assign_kind(self):
#        name = self.name
#        if name[0]=='I':
#            self.kind = I
#
#
##for the moment, there is just a method for DC source I to plot
#    def plot(self, ax, lw_scale=1):
#        size = 0.3
#        width = 0.066
#
#    #    plt.rc('lines', color='k', lw=2)
##        x = np.linspace(-np.pi, np.pi, 21)
##        sine = np.sin(x)*size/4
##        _line_sin = np.stack((x/np.pi*size/2, sine)).T
##
#        _circle_center = np.array([0,0])
#        _line1 = np.array([[-1, 0], [-size, 0]])
#        _line2 = np.array([[size, 0], [1, 0]])
#        _arrow_line = np.array([[size*2/3,0], [-size*2/3,0]])
#        _arrow_head = np.array([[size*2/3-width,width], [size*2/3,0], [size*2/3-width,-width]])
#        if not self.horiz:
#            _arrow_line = _arrow_line[:, ::-1]
#            _arrow_head = _arrow_head[:, ::-1]
#            _line1 = _line1[:, ::-1]
#            _line2 = _line2[:, ::-1]
#        _arrow_line += self.center
#        _arrow_head += self.center
#        _line1 += self.center
#        _line2 += self.center
#        _circle_center = _circle_center +self.center
#
#        arrow_line = Line2D(*pt_to_xy(_arrow_line))
#        arrow_head = Line2D(*pt_to_xy(_arrow_head))
#        line1 = Line2D(*pt_to_xy(_line1))
#        line2 = Line2D(*pt_to_xy(_line2))
#        circle = Circle(_circle_center, radius=size, fc='none', ec = self.color, lw=2*lw_scale)
#        artists = [line1, line2, arrow_line, arrow_head, circle]
#        for art in artists:
#            ax.add_artist(art)
#
#    @classmethod
#    def build_sources_nodes(cls):
#        for source in cls.source_DC_list:
#            for node in Node.node_list:
#                if (node.coor==source.start).all():
#                    node0 = node
#                elif (node.coor==source.end).all():
#                    node1 = node
#            cls.source_DC_nodes[source] = [node0, node1] # this dict map a dipole to a pair of nodes


class Representation():
    # represents a subset of dipoles and nodes

    def __init__(self, kind, circuit, dipoles):
        self.dipoles = dipoles
        self.dipoles_names = [dipole.name for dipole in self.dipoles]
        self.circuit = circuit

        if kind=='raw':
            self.nodes = circuit.nodes
        elif kind=='equ':
            self.nodes = circuit.eq_nodes
        else:
            raise ValueError("For Representation(), 'kind' should be in ['raw', 'equ']")
        self.ker = None
#        self.I_DC_nodes=[]  #list that is the size of the node list (whether it is raw or eq list). For each node it gives the current (with sign) which is at the node
        if self.dipoles is not None:
            self.build_A()

    def convert_raw_to_eq(self): #just for DC circuit$
        print('ker')
        print(self.ker)
        if len(self.ker)!=0:
            dipole_in_ker = [elt>0 for elt in np.abs(self.ker).sum(axis=0)] # is a given dipole used in loops
            filtre = [(dipole_in_ker[ii] and dipole.kind!=W) for ii, dipole in enumerate(self.dipoles)]
            dipoles = [dipole for ii, dipole in enumerate(self.dipoles) if filtre[ii]]
            rep = Representation('equ', self.circuit, dipoles)
            ker_new = []
            for loop in self.ker:
                ker_new.append(loop[filtre])
            rep.ker = np.array(ker_new)
            rep.associated_phiext = self.associated_phiext
            # we do not care but
            # TODO should add the other loops that appear when making the nodes equivalents
            # basically when constructing ker should take in consideration preexisting one
            # in this case filtre must not be used

            # TODO: Add filtering of the nodes
        else:
            rep = Representation('equ', self.circuit, None)
        return rep

    def build_A(self):
        nN = len(self.nodes)
        columns=[]
        for dipole in self.dipoles:
            column = np.zeros(nN)
            node_start, node_end = self.circuit.dipoles_nodes[dipole]
            # print(self.nodes)
            index_start = self.nodes.index(node_start.eq_node())
            index_end = self.nodes.index(node_end.eq_node())
            column[index_start]=-1
            column[index_end]=1
            columns.append(column)
        self.A = np.array(columns).T
#        print(self.A)
#        print(self.nodes)
#        print(self.dipoles)

    def plot_arrows(self, ax):
        for dipole in self.dipoles:
            if dipole.kind!=W:
                dipole.plot_arrow(ax, size=1)

    def find_loops(self, debug=False): # returns loops in trigo way
        if len(self.A)!=0:
            if self.ker is None:
                ker = find_ker(self.A)
                # now we need to orient them
                for ii, loop in enumerate(ker):
                    if not self.trigo(loop):
                        ker[ii]=-loop
                self.ker = ker

            if debug:
                # printing
                for loop in self.ker:
                    self.print_loop(loop)
                print('')
            return True
        else:
            print('No DC loops')
            return False

    def trigo(self, loop): # check if loops rotate in trigo way
        indices = np.where(loop)[0]
        x_max=0
        index_max=0
        for index in indices: # all the dipoles which are in the loop
            dipole = self.dipoles[index]
            x_center = dipole.center[0]
            if x_max < x_center: # far most right dipole (should be a vertical dipole by construction)
                x_max = x_center
                index_max = index

        if loop[index_max]==1:
            return True
        else:
            return False

    def print_loop(self, loop):
        to_print = 'loop ['
        for ii, ori in enumerate(loop): # ori may be 0, 1, or -1
            if ori==1:
                to_print+= str(self.dipoles[ii])+', '
            if ori==-1:
                to_print+= '-'+str(self.dipoles[ii])+', '
        print(to_print[:-2]+']')

    def associate_loop_with_phiext(self):
        self.associated_phiext = [] # list (running on loops) of list of phiext
        for jj, loop in enumerate(self.ker):
            self.associated_phiext.append([])
            dipole_loop = [self.dipoles[ii] for ii in np.where(loop)[0]]
            for hole in self.circuit.holes:
                if hole_in_loop(hole, dipole_loop):
                    self.associated_phiext[jj].append(hole)

    def build_vec_I_DC(self):
        # print('build vec I DC')
        # print(self.dipoles)
        vec=[]
        for (ii, elt) in enumerate(self.dipoles):
            if elt.kind==I:
                for val in elt.val:
                    if val[1]==0:
                        vec.append(val[0])
                        break
                else:
                    print('Print no DC current in %s'%(elt.name))
                    vec.append(0)
            else:
                vec.append(0)
        self.vec_I_DC=vec

    def build_constrain(self, kind):
        # print('Build constrain '+kind)
        if self.dipoles is not None:
            n_phi = len(self.dipoles)
            # print('ker')
            # print(type(self.ker))
            if self.ker.shape != (0,):
                constrain_mat, reshape_mat, choices = complete_with_identity(self.ker)
            else:
                constrain_mat = np.eye(n_phi)
                reshape_mat = np.eye(n_phi)
                choices = np.array([])
            inv = nl.inv(constrain_mat)
            F = inv @ reshape_mat

            self.F = F
            # print(F)
            # print(self.dipoles)
            if kind=='DC':
                constrain_vec = ['0' for ii in range(n_phi)]
                for ii, choice in enumerate(choices):
                    associated_str = associated_string(self.associated_phiext[ii],type_string='flux')
                    constrain_vec[choice] = associated_str
                f = matrix_product_str(inv, constrain_vec)
                # print(f)
                self.f = f
            self.independant_dipoles = [dipole for ii, dipole in enumerate(self.dipoles) if not(ii in choices)]
            self.build_vec_I_DC()
            # print('Vec I DC')
            # print(self.vec_I_DC)
            # print('Independant Dipoles')
            # print(self.independant_dipoles)

    def eval_f(self):
        f_eval = []
        for elt in self.f:
            f_eval.append(eval(elt, self.circuit.holes_val))
        self.f_eval = np.array(f_eval)
#
#    def eval_I_DC_nodes(self):
#        I_DC_nodes_eval=[]
#        for elt in self.I_DC_nodes:
#            I_DC_nodes_eval.append(eval(elt,Source_DC.source_DC_val))
#        self.I_DC_nodes_eval=np.array(I_DC_nodes_eval)


    def oL_oJ_mat(self):
        n_phi = len(self.dipoles)
        oL_mat = np.zeros((n_phi, n_phi))
        oJ_mat = np.zeros((n_phi, n_phi))
        for ii, dipole in enumerate(self.dipoles):
            if dipole.kind==L:
                oL_mat[ii,ii]=1/dipole.val
            if dipole.kind==T:
                oL_mat[ii,ii]=1/dipole.val[0]/dipole.val[1]
            if dipole.kind==J:
                oJ_mat[ii,ii]=1/dipole.val[0]
        self.oL_mat = oL_mat
        self.oJ_mat = oJ_mat
#        print('oJ_oL_mat_defined')
#        print(oL_mat)

    def U(self, gamma_vec):
        phi_vec = self.F @ gamma_vec + self.f_eval
#        print(phi_vec)
#        print(self.oL_mat)
#        print(self.oJ_mat)
        return np.sum(1/2*self.oL_mat@phi_vec**2-self.oJ_mat@np.cos(phi_vec)) + phi_vec.T @ self.vec_I_DC

    def current(self, gamma_vec): # pendant of U for current conservation
        phi_vec = self.F @ gamma_vec + self.f_eval
        current = np.diag(self.oL_mat).T*phi_vec+np.diag(self.oJ_mat)*np.sin(phi_vec) + np.array(self.vec_I_DC) # branch current vec
        return np.delete(self.A @ current, 0, axis=0) # should be 0 vector when solved

    def solve_DC(self, guess=None, verbose=True, debug=False):

        if verbose:
            print('')
            print('################')
            print('### Solve DC ###')
            print('################')
            print('')

        if self.dipoles is not None:
            self.oL_oJ_mat()
            self.eval_f()
#            self.eval_I_DC_nodes()
            n_gamma = len(self.independant_dipoles)
            if debug:
                print(n_gamma)
                if n_gamma==1:
                    print(self.oL_mat)
                    print(self.oJ_mat)
                    gammas = np.linspace(-2*np.pi, 2*np.pi, 101)
                    Us = []
                    for gamma in gammas:
                        Us.append(self.U(np.array([gamma])))
                    gammas = np.array(gammas)
                    fig, ax = plt.subplots()
                    ax.plot(gammas, Us)
            if guess is None:
                gamma_vec_0 = np.zeros((n_gamma,))
            else:
                gamma_vec_0 = guess
#            gamma_vec_0 = np.array([np.pi/2+0.1])
            res = minimize(self.U, gamma_vec_0, method='Powell')
            if n_gamma==1:
                gamma_vec_sol = np.array([res.x])
            else:
                gamma_vec_sol = np.array(res.x)
#            print(gamma_vec_sol)
            if debug:
                if n_gamma==1:
                    ax.plot(gamma_vec_sol[0], self.U(gamma_vec_sol), '.')

            if verbose:
                print('')
                print('Res from minimization')
                print('Gamma vec sol')
                print(self.independant_dipoles)
                print(gamma_vec_sol)
                print('Node currents')
                print(self.current(gamma_vec_sol))

            print(self.dipoles)
            print(self.nodes)
            print(gamma_vec_sol)
            print(self.A)
            res = root(self.current, gamma_vec_sol)
            gamma_vec_sol = np.array(res.x)

            if verbose:
                print('')
                print('Res from current law')
                print('Gamma vec sol')
                print(self.independant_dipoles)
                print(gamma_vec_sol)
                print('Node currents')
                print(self.current(gamma_vec_sol))

            phi_vec_sol = self.F@gamma_vec_sol + self.f_eval
            for ii, dipole in enumerate(self.dipoles):
                dipole.phi_DC = phi_vec_sol[ii]
            if verbose:
                print(self.dipoles)
                print(phi_vec_sol)
                print('')
                print('Gamma sol')
                print(gamma_vec_sol)
            return gamma_vec_sol

    def build_current_mat(self, omega):
        n_phi = len(self.dipoles)
        current_mat = np.zeros((n_phi, n_phi), dtype=complex)
        counter_T = 0
        for ii, dipole in enumerate(self.dipoles):
            if dipole.kind==C:
                current_mat[ii, ii]=-dipole.val*omega**2
            if dipole.kind==R:
                current_mat[ii, ii]=1j*1/dipole.val*omega
            if dipole.kind==L:
                current_mat[ii, ii]=1/dipole.val
            if dipole.kind==J:
                capa = 1/(2*np.pi*dipole.val[1])**2/dipole.val[0]
                current_mat[ii, ii]=1/dipole.val[0]*np.cos(dipole.phi_DC)-capa*omega**2
            if dipole.kind==T:
                prefact = omega/dipole.val[1]/np.sin(omega*dipole.val[0])
#                if prefact>500:
#                    print(prefact)
                if counter_T%2==0:
                    current_mat[ii, ii]=prefact*np.cos(omega*dipole.val[0])
                    current_mat[ii, ii+1]=prefact*(-1)
                else:
                    current_mat[ii, ii-1]=prefact*(-1)
                    current_mat[ii, ii]=prefact*np.cos(omega*dipole.val[0])
                counter_T+=1
        return current_mat

    def build_capa_ind_mat(self):
        n_phi = len(self.dipoles)
        capa_m = np.zeros((n_phi, n_phi))
        ind_m = np.zeros((n_phi, n_phi))

        for ii, dipole in enumerate(self.dipoles):
            if dipole.kind==C:
                capa_m[ii, ii]=1/8/(conv_C/dipole.val)
            if dipole.kind==L:
                ind_m[ii, ii]=conv_L/dipole.val
            if dipole.kind==J:
                capa = 1/(2*np.pi*dipole.val[1])**2/dipole.val[0]
                capa_m[ii, ii]=1/8/(conv_C/capa)
                ind_m[ii, ii]=conv_L/dipole.val[0]*np.cos(dipole.phi_DC)
                
        return self.F.T@capa_m@self.F, self.F.T@ind_m@self.F

    def freq_zpf(self, verbose=False):
        capa_m, ind_m = self.build_capa_ind_mat()
        omegamat = inv(capa_m)@ind_m
        e, v = nl.eig(omegamat)
        idx = e.argsort()   
        e = e[idx]
        v = v[:,idx]

        U = np.diag(v.T @ ind_m @ v)/2

        factor = U / (e**0.5/4)

        v = v / factor**0.5

        return e**0.5, self.F@v # [dipole, mode]

    def update(self, dipoles, values):
        for (value, dipole) in zip(values, dipoles):
            if dipole.kind == 'J':
                dipole.val = (value, dipole.val[1]) # assume not dynamically changing plasma frequency
            else:
                dipole.val = value
    
    def optimize(self, constraints, dipoles, guesses, show_init=False):
        """
        constraints : is the a dictionnary of target parameters. It should be constructed as follows :
            keyword -> 'freq' is the target frequency in GHz
            keyword -> any dipole object is the target zpf of mode with freq 'freq' in dipole object
        
        dipoles : is the list of dipoles on which the optimization can modify the values
        guesses : is the list of guesses for these values

        /!\ when acting on junctions elements, only the inductance parameter is optimized on
        """

        mode_indices = []
        mode_target = []
        zpf_dipole_indices = []
        zpf_mode_indices = []
        zpf_target = []
        for ii, constraint in enumerate(constraints):
            mode_indices.append(ii)
            mode_target.append(constraint['f'])
            for key in constraint.keys():
                if key != 'f':
                    dipole_index = self.dipoles_names.index(key.name)
                    zpf_dipole_indices.append(dipole_index)
                    zpf_mode_indices.append(ii)
                    zpf_target.append(constraint[key])
        mode_target = np.array(mode_target)
        zpf_target = np.array(zpf_target)

        if show_init:
            f, v = self.freq_zpf()
            fs = f[mode_indices]
            vs = np.abs(v[zpf_dipole_indices, zpf_mode_indices])

            print("Result")
            for (v, x, y) in zip(mode_indices, mode_target, fs):
                print(v, ':', x, '->', y)
            for (u, v, x, y) in zip(zpf_dipole_indices, zpf_mode_indices, zpf_target, vs):
                print(self.dipoles[u], v, ':', x, '->', y)
            print('')
            print("Parameters")
            for dipole in dipoles:
                print(dipole, '=', dipole.val)

        def cost(ps):
            ps = np.abs(ps)
            self.update(dipoles, ps)

            f, v = self.freq_zpf()

            fs = f[mode_indices]
            vs = np.abs(v[zpf_dipole_indices, zpf_mode_indices])

            f_cost = np.sum(((fs-mode_target)/mode_target)**2)
            zpf_cost = np.sum(((vs-zpf_target)/zpf_target)**2)

            cost = np.real((f_cost + zpf_cost)*1e3)

            if False:
                print(ps)
                print(fs, mode_target)
                print(vs, zpf_target)
                print(cost)
                print('')

            return cost

        res = minimize(cost, guesses)
        self.update(dipoles, res.x)
        
        f, v = self.freq_zpf()
        fs = f[mode_indices]
        vs = np.abs(v[zpf_dipole_indices, zpf_mode_indices])

        print("Result")
        for (v, x, y) in zip(mode_indices, mode_target, fs):
            print(v, ':', x, '->', y)
        for (u, v, x, y) in zip(zpf_dipole_indices, zpf_mode_indices, zpf_target, vs):
            print(self.dipoles[u], v, ':', x, '->', y)
        print('')
        print("Parameters")
        for dipole in dipoles:
            print(dipole, '=', dipole.val)
        
        return f, v

    def zpf(self, vector, dipole, mode_index=None):
        """
        tool to retrieve easily the zpf accross a dipole
        vector : zpf vector
        dipole : dipole object
        mode_index : mode index
        """
        dipole_index = self.dipoles_names.index(dipole.name)
        if mode_index is not None:
            return vector[dipole_index, mode_index]
        else:
            return vector[dipole_index]

    def mag_energy(self, omega, phi):
        energy = 0
        counter_T = 0
        for ii, dipole in enumerate(self.dipoles):
            if dipole.kind==C:
                pass
            if dipole.kind==R:
                pass
            if dipole.kind==L:
                energy+=conv_L/dipole.val/2*np.abs(phi[ii])**2
            if dipole.kind==J:
                energy+=conv_L/dipole.val[0]/2*np.abs(phi[ii])**2*np.cos(dipole.phi_DC)
            if dipole.kind==T:
                if counter_T%2==0:
                    phi_se = np.array([phi[ii], phi[ii+1]])
                    bl = omega*dipole.val[0]
                    Ltot = dipole.val[0]*dipole.val[1]
                    det = 2*1j*np.sin(bl)
                    mat = 1/det*np.array([[np.exp(1j*bl/2), -np.exp(-1j*bl/2)],
                                          [-np.exp(-1j*bl/2), np.exp(1j*bl/2)]])
                    phi_pm = mat @ phi_se
#                    print('phi +/- = ', phi_pm)
                    angle = np.angle(phi_pm[0])
                    magni = np.abs(phi_pm[0])
                    e = magni**2/Ltot*bl*(bl+np.sin(bl)*np.cos(2*angle))
#                    print('energy_T =', e)
                    energy+= conv_L*e
                counter_T+=1
#        print('energy = ', energy)
        return energy

    def ele_energy(self, omega, phi):
        print('Electric energy computation does not work yet')
        print('This is not useful anyway')
        energy = 0
        counter_T = 0
        for ii, dipole in enumerate(self.dipoles):
            if dipole.kind==C:
                energy+=dipole.val/2*np.abs(phi[ii])**2*np.abs(omega)**2
            if dipole.kind==R:
                pass
            if dipole.kind==L:
                pass
            if dipole.kind==J:
                capa = 1/(2*np.pi*dipole.val[1])**2/dipole.val[0]
                energy+=capa/2*np.abs(phi[ii])**2*np.abs(omega)**2
            if dipole.kind==T:
                if counter_T%2==0:
                    phi_se = np.array([phi[ii], phi[ii+1]])
                    bl = omega*dipole.val[0]
                    Ctot = dipole.val[0]/dipole.val[1]
                    det = 2*1j*np.sin(bl)
                    mat = 1/det*np.array([[np.exp(1j*bl/2), -np.exp(-1j*bl/2)],
                                          [-np.exp(-1j*bl/2), np.exp(1j*bl/2)]])
                    phi_pm = mat @ phi_se
                    e_plus = Ctot*np.abs(phi_pm[0])**2/2*np.abs(omega)**2
                    e_minus = Ctot*np.abs(phi_pm[1])**2/2*np.abs(omega)**2
                    print('energy_plus =', e_plus)
                    print('energy_minus =', e_minus)
                    energy+= (e_plus + e_minus)
                counter_T+=1
        print('energy = ', energy)
        return energy

    def eom(self, omega):
        cur_mat = self.build_current_mat(omega)
        eom_mat = self.A @ cur_mat @ self.F
        eom_mat = np.delete(eom_mat, 0, axis=0) #delete ground equation of motion
        return eom_mat, cur_mat

    def det_eom(self, omega): # take a single omega or several
        if isinstance(omega, np.ndarray):
            if omega.ndim==1:
                dets = np.empty(omega.shape, dtype=complex)
                for ii, w in enumerate(omega):
                    dets[ii] = self.det_eom(w)
                return dets
            elif omega.ndim==2:
                dets = np.empty(omega.shape, dtype=complex)
                for jj, line in enumerate(omega):
                    for ii, w in enumerate(line):
                        dets[jj, ii] = self.det_eom(w)
                return dets
            else:
                raise ValueError('det_eom Cannot handle arrays with more than 2 dim')
        eom_mat, cur_mat = self.eom(omega)
        to_return = nl.det(eom_mat)
        counter_T = 0
        for ii, dipole in enumerate(self.dipoles):
            if dipole.kind==T:
                if counter_T%2==0:
                    to_return = to_return*np.sin(omega*dipole.val[0])
                counter_T+=1
#        print('det_eom_called')
#        print('omega val '+str(omega))
#        print('res '+str(to_return))
        return to_return

    def display_eom(self, ax, omegas, kappas=None, guesses=None, log_kappa=True):
        if guesses is not None:
            eig_omegas, eig_phizpfs = self.solve_EIG(guesses, verbose=False)
        if kappas is None:
            dets = self.det_eom(omegas)
            ax.plot(omegas/2/np.pi, np.abs(dets)/np.nanmax(np.abs(dets)))
            ax.hlines(0, omegas[0]/2/np.pi, omegas[-1]/2/np.pi, lw=0.5, color='k')
            if not(guesses is None):
                for ii, eig_omega in enumerate(eig_omegas):
                    ax.plot(np.real(eig_omega)/2/np.pi, 0, '.', color='C%d'%ii)
            ax.set_xlabel(r'$\omega/2\pi$')
            ax.set_ylabel(r'$\|Det(\omega)|$')
        else:
            n_omega = len(omegas)
            n_kappa = len(kappas)
            kappa_min = kappas[0]
            kappa_max = kappas[-1]
            if log_kappa:
                log_kappa_min = np.log(kappa_min)
                log_kappa_max = np.log(kappa_max)
                log_kappas = np.linspace(log_kappa_min, log_kappa_max, n_kappa)
                kappas = np.exp(log_kappas)
                domega = (omegas[1]-omegas[0])
                dkappa = (kappas[1]/kappas[0])
            else:
                domega = (omegas[1]-omegas[0])
                dkappa = (kappas[1]-kappas[0])
            _omegas, _kappas = np.meshgrid(omegas, kappas)
            Omegas = _omegas+1j*_kappas/2
            dets = self.det_eom(Omegas)

            if log_kappa:
                omegas = omegas-domega/2
                kappas = kappas/dkappa**0.5
                end_omega = omegas[-1]+domega
                end_kappa = kappas[-1]*dkappa
                omegas = np.append(omegas, end_omega)
                kappas = np.append(kappas, end_kappa)
                _omegas, _kappas = np.meshgrid(omegas, kappas)
                color_dets = color(dets, power=0.1)
                color_tuples = color_dets.reshape(n_omega*n_kappa, 3).astype(float)
                color_tuples = np.insert(color_tuples,3,1.0,axis=1)
                m = ax.pcolormesh(_omegas/2/np.pi, _kappas/2/np.pi, np.imag(dets), color=color_tuples, linewidth=0)
                m.set_array(None)
            else:
                ax.imshow(color(dets, power=0.1), aspect='auto', extent=[(omegas[0]-domega/2)/2/np.pi, (omegas[-1]+domega/2)/2/np.pi, (kappas[0]-dkappa/2)/2/np.pi, (kappas[-1]+dkappa/2)/2/np.pi], origin='lower')

            if not(guesses is None):
                for ii, eig_omega in enumerate(eig_omegas):
                    kappa = 2*np.imag(eig_omega)
                    omega = np.real(eig_omega)
                    if kappa>kappa_min and kappa<kappa_max:
                        ax.plot(omega/2/np.pi, kappa/2/np.pi, 'o', color='C%d'%ii, markeredgecolor='w')
                    elif kappa<kappa_min:
                        ax.plot(omega/2/np.pi, kappa_min/2/np.pi, 'v', color='C%d'%ii, markeredgecolor='w')
                    else:
                        ax.plot(omega/2/np.pi, kappa_max/2/np.pi, '^', color='C%d'%ii, markeredgecolor='w')
            ax.set_xlabel(r'$\omega/2\pi$')
            ax.set_ylabel(r'$\kappa/2\pi$')
            if log_kappa:
                ax.set_yscale('log')
        if guesses is not None: 
            return eig_omegas, eig_phizpfs
        else:
            return None, None

    def eig_phi(self, omega): # return the eig vector with smallest eigenvalue (expected 0)
                              # at a given frequency
        eom_mat, cur_mat = self.eom(omega)
        e, v = nl.eig(eom_mat)
        gamma = v.T[np.argmin(np.abs(e)**2)]
        phi = self.F @ gamma
        return phi#, gamma
    
    def solve_EIG(self, guesses, verbose=False):


        if verbose:
            print('')
            print('#################')
            print('### Solve EIG ###')
            print('#################')
            print('')

        eig_omegas = []
        eig_phizpfs = []
#        eig_gammazpfs = []
        for guess in guesses:
            if verbose:
                print('Det eom for Newton routine')
                print(self.det_eom(guess))
            eig_omega = newton(self.det_eom, guess) # find nearest root -> eig_omega
            if verbose:
                print('Mode omega/2pi:', np.real(eig_omega)/2/np.pi)
            if 2*np.imag(eig_omega)/np.real(eig_omega) > 1e-9:
                if verbose:
                    print('Mode kappa/2pi:', 2*np.imag(eig_omega)/2/np.pi)
                    print('Mode Q:', np.real(eig_omega)/2/np.imag(eig_omega))
                    print('Cannot handle dissipative modes precisely')
                    print('phizpf will be approximated to the no-dissipation case')
            eig_omegas.append(eig_omega)
            eig_phi = self.eig_phi(eig_omega) # find the corresponding phi configuration
            if verbose:
                print(eig_phi)
            mag = self.mag_energy(eig_omega, eig_phi) # find the magnetic energy of this configuration
            #prop = (4*mag/(sci*np.real(eig_omega)))**0.5 # factor to convert the phi configuration to a phi_zpf configuration
            prop = (4*mag*2*np.pi/np.real(eig_omega))**0.5 # factor to convert the phi configuration to a phi_zpf configuration
            phi_zpf = np.real(eig_phi)/prop
#            gamma_zpf = np.real(eig_gamma)/prop
            eig_phizpfs.append(phi_zpf)
#            eig_gammazpfs.append(gamma_zpf)
#            print('Mode impedance:', 1/mag/2*np.real(eig_omega)) # WRONG
            if verbose:
                print('phi_zpf:', np.real(phi_zpf))
                print(self.dipoles)
                print('')
        return np.array(eig_omegas), np.array(eig_phizpfs)#, np.array(eig_gammazpfs)

    def plot_phi(self, ax, phi, offset=0.3, color='k'):
        for ii, dipole in enumerate(self.dipoles):
            if dipole.kind!=T:
                dipole.plot_arrow(ax, phi[ii], offset=offset, color=color)

    def solve_AC(self, method='linear', verbose=True):

        if verbose:
            print('')
            print('################')
            print('### Solve AC ###')
            print('################')
            print('')

        if method=='linear':
            # spirit solve separately the different frequency components of
            # current sources:
            phis = [] # store each solution
            for dipole in self.dipoles:
                if dipole.kind == I:
                    if verbose:
                        print('')
                        print(str(dipole)+':')
                    for val in dipole.val: # val is : amp, freq, phase
                        if val[1]!=0:
                            if verbose:
                                print('val '+str(val))
                            vec_I_AC = np.array([val[0]/2*np.exp(1j*val[2]) if dipole==dpl else 0 for dpl in self.dipoles])
                            vec_Igamma_AC = vec_I_AC@self.F
                            eom_mat, cur_mat = self.eom(val[1])
                            sol_gamma = -nl.inv(eom_mat)@vec_Igamma_AC
#                            if verbose:
#                                print('vec_Igamma_AC')
#                                print(vec_Igamma_AC)
#                                print('eom_mat')
#                                print(eom_mat)
#                                print('sol_gamma')
#                                print(sol_gamma)
                            sol_phi = self.F@sol_gamma
                            sol_current_branch = cur_mat @ sol_phi
                            sol_current_node = self.A @ sol_current_branch
                            if verbose:
                                print(self.dipoles)
                                print(sol_current_branch)
                                print(self.nodes)
                                print(sol_current_node)
                                print('')
                            phis.append(self.F@sol_gamma)
            return np.sum(np.array(phis), axis=0)
#    def print_dipoles_nodes(cls):
#        for dipole, nodes in cls.dipoles_nodes.items():
#            print('%s - from %s to %s'%(dipole.name, nodes[0], nodes[1]))


#
#        self.A = A
#        self.D = D
#        self.nodes = nodes
#        self.nodes_type = nodes_type
#        shape = np.shape(A)
#        self.nD = shape[1]
#        self.nN = shape[0]
#
#    def find_loops(self):
#        ker = find_ker(self.A) # find ker but need to check the loop orientation
#        for ii, loop in enumerate(ker):
#            edges, orientations = self.get_edges_loop(loop)
#            print(self.translate_loop(loop))
#            print(loop)
#            ker[ii] = clockwise(edges, orientations)*loop
#            print(ker[ii])
#            print('')
#        return ker

    def get_edges_loop(self, loop, ax=None, display=False, offset=0):
        edges=[]
        orientations=[]
        which_dipole=np.where(loop)[0]
        for index_dipole in which_dipole:
            indices_nodes=np.where(self.A[:, index_dipole])[0]
            index_node_0 = indices_nodes[0]
            index_node_1 = indices_nodes[1]
            if self.A[index_node_0, index_dipole]==-1:
                coord_node_0=self.nodes[index_node_0]
                coord_node_1=self.nodes[index_node_1]
            else:
                coord_node_0=self.nodes[index_node_1]
                coord_node_1=self.nodes[index_node_0]
            edges.append((coord_node_0,coord_node_1))
            orientations.append(loop[index_dipole])
            if display and not(ax is None):
                ax.plot([coord_node_0[1]+0.1*(offset+1),coord_node_1[1]+0.1*(offset+1)],[coord_node_0[0]+0.1*(offset+1),coord_node_1[0]+0.1*(offset+1)],color='C%d'%index_dipole)
        return edges, orientations

    def c(self, loop):
        translated_loop=[]
        for ii, elmt in enumerate(loop):
            if abs(elmt) == 1:
                translated_loop.append(self.D[ii])
        return translated_loop

class Circuit(object):

    def __init__(self, circuit_array, let=[]):
        # dipoles_impedance should be a fct that returns a dictionnary
        # of dipole impedances. This function should have one argument
        # the drive frequency

        # dipoles_val should be a dict that has the values of the dipoles

        # circuit_array is an array of strings that contains nodes on even
        # indices and dipoles names in between those nodes. The nodes form a
        # square grid

        self.circuit_array = circuit_array

        # store dipoles
        self.dipoles = [] # contains the copied dipoles
        self.dipoles_val = {}
        self.dipoles_DC = []
        self.dipoles_AC = []

        # store nodes
        self.nodes = [] # contains the copied nodes
        self.nodes_val = {}
        self.eq_nodes = []
        self.eq_nodes_dict = {}

        #store holes
        self.holes = [] # contains the copied holes
        self.holes_val = {}

        self.parse()

        # store match dipoles and nodes
        self.dipoles_nodes = {}
        self._build_dipoles_nodes() # match dipoles and nodes

        self.rep_raw_DC = Representation('raw', self, self.dipoles_DC) # should occur before equivalent nodes
        is_DC_loop = self.rep_raw_DC.find_loops()
        if is_DC_loop:
            self.rep_raw_DC.associate_loop_with_phiext()

        self._equivalent_nodes()

        # Now solve DC representation
        # Be careful one should make any superconducting loop explicit for now
        is_DC_loop=False
        if is_DC_loop:
            self.rep_DC = self.rep_raw_DC.convert_raw_to_eq()
            self.rep_DC.build_constrain('DC')
            self.rep_DC.solve_DC(debug=False)


        self.rep_AC = Representation('equ', self, self.dipoles_AC)
        print('dipoles AC')
        print(self.rep_AC.dipoles)
        if False:
            # not really interesting to be printed
            print('nodes AC')
            print(self.rep_AC.nodes)
            print('ker')
            print(self.rep_AC.ker)
        self.rep_AC.find_loops()
        self.rep_AC.build_constrain('AC')        


    def parse(self,):
        self.plotted_elt = set()
        circuit = self.circuit_array
        for jj, line in enumerate(circuit):
            for ii, elt in enumerate(line):
                if elt is None:
                    pass
                else:
                    if jj%2==0: # can be node or horizontal dipole or a source DC I horizontal
                        if ii%2==0: # it is a node
                            elt.assign_coor(self, array_to_plot((jj, ii))) # the circuit is passed as a first argument
                        else: # it is a dipole horizontal or a source DC I horizontal
                            elt.assign_coor(self, array_to_plot((jj, ii-1)), array_to_plot((jj, ii+1)))
                    else: # can be a vertical dipole or a flux
                        if ii%2==0: # it is a vertical dipole or a source DC I horizontal
                            elt.assign_coor(self, array_to_plot((jj+1, ii)), array_to_plot((jj-1, ii)))
                        else: # it is a flux
                            elt.assign_coor(self, array_to_plot((jj, ii)))
                    self.plotted_elt.add(elt)

        name_elements(self.dipoles)

        dipoles = [dipole for dipole in self.dipoles]
        for dipole in dipoles:
            dipole.fill_dipoles()

        name_elements(self.nodes)
        name_elements(self.holes)

        self.update_vals()

    def update_vals(self):
        for dipole in self.dipoles:
            self.dipoles_val[dipole.name]=dipole.val
        for hole in self.holes:
            self.holes_val[hole.name]=hole.val

    def plot(self, ax, debug=False, lw_scale=1):
        if not debug:
            ax.axis('off')
        ax.set_xlim(-1, len(self.circuit_array[0]))
        ax.set_ylim(-len(self.circuit_array), 1)
        for elt in self.dipoles:
            elt.plot(ax, lw_scale=lw_scale)
        for elt in self.nodes:
            elt.plot(ax, lw_scale=lw_scale)
        for elt in self.holes:
            elt.plot(ax, lw_scale=lw_scale)
        ax.set_aspect('equal')

    def _build_dipoles_nodes(self):
        for dipole in self.dipoles:
            for node in self.nodes:
                if (node.coor==dipole.start).all():
                    node0 = node
                elif (node.coor==dipole.end).all():
                    node1 = node
            self.dipoles_nodes[dipole] = [node0, node1] # this dict map a dipole to a pair of nodes

    def _equivalent_nodes(self):

        node_pairs = []
        for dipole in self.dipoles: # each time dipole is wire, the pair consist of equivalent nodes
            if dipole.kind==W:
                node_pairs.append(self.dipoles_nodes[dipole].copy())

        parents = []
        node_groups = []
        for node in self.nodes: # each node with same parent are equivalent
            parent = node.parent
            if parent.name!='':
                if parent not in parents:
                    parents.append(parent)
                    node_groups.append([node])
                else:
                    index = parents.index(parent)
                    node_groups[index].append(node)

        node_pairs += node_groups
        count = 0
        while count<len(node_pairs): # find the most general equivalent groups
            pair = node_pairs.pop(count)
            for ii, group in enumerate(node_pairs[count:]):
                if any([elt in group for elt in pair]):
                    node_pairs[ii+count]+=pair # fusionner la paire avec le groupe
                    break
            else:
                node_pairs = [pair]+node_pairs
                count +=1

        for group in node_pairs:
            # The reference node is either the one with a name or the most common one
            list_bool = [(elt.kind==G or elt.kind==E) for elt in group]
            if any(list_bool):
                ref = group[list_bool.index(True)]
            else:
                ref = max(set(group), key=group.count)
            # print(ref, group)
            for node in group:
                self.eq_nodes_dict[node]=ref

        eq_nodes = list(set(self.eq_nodes_dict.values()))
        # print(self.eq_nodes_dict)
        # print(eq_nodes)
        # put ground first
        list_bool = [elt.name[0]=='G' for elt in eq_nodes]
        if any(list_bool):
            ground = eq_nodes.pop(list_bool.index(True))
            eq_nodes = [ground] + eq_nodes

        self.eq_nodes = eq_nodes

    def sweep(self, elt, vals, guesses):
        list_eig_omegas = []
        list_eig_phizpfs = []
        current_guesses = guesses
        guess_DC =None
        for ii, val in enumerate(vals):
            elt.val = val
#            print(elt.val)
            if elt.kind in [F, L, J] or ii==0: # then the DC solution changes
                guess_DC = self.rep_DC.solve_DC(guess=guess_DC, verbose=False)
                print('guess_DC')
            try:
                eig_omegas, eig_phizpfs = self.rep_AC.solve_EIG(current_guesses, verbose=False)
#                print('solved')
#                print(eig_omegas)
            except Exception:
                eig_omegas = np.ones((len(guesses),))*np.nan
                eig_phizpfs = None
            else:
                current_guesses = eig_omegas
                #TODO phizpfs
            list_eig_omegas.append(eig_omegas)
            list_eig_phizpfs.append(eig_phizpfs)
        return np.array(list_eig_omegas).T, list_eig_phizpfs#np.array(list_eig_phizpfs).T

    def sweep_2_params(self,elt_1,vals_1,elt_2,vals_2,guesses):
        matrix_eig_omegas = []
#        matrix_eig_phizpfs =[]
        current_guesses =guesses
        guess_DC=None

        if not(elt_1.kind in [F, L, J]) and elt_2.kind in [F, L, J]:
                mat_guesses_omegas= self.sweep_2_params(self,elt_2,vals_2,elt_1,vals_1,guesses)
            #we go back to the case where elt1 is F,L,J and elt2 no.
        else:
            jj=0
            continue_ok=True
            print_plot=False
            while jj<len(vals_2) and continue_ok:
                print(jj)
                val2=vals_2[jj]
                list_eig_omegas_jj=[]
    #            list_eig_phizpfs_ii=[]
                for ii, val1 in enumerate(vals_1):
#                    print(ii)
                    elt_1.val=val1
                    elt_2.val=val2
                    if elt_1.kind in [F, L, J] and elt_2.kind in [F, L, J]:
                        #actualize the guess DC each time
                        guess_DC=self.rep_DC.solve_DC(guess=guess_DC, verbose=False)

                        if ii==0:
                            guess_DC_mem=guess_DC
                        elif ii==len(vals_1)-1:
                            guess_DC=guess_DC_mem

                    elif elt_1.kind in [F, L, J] and not(elt_2.kind in [F, L, J]) :
                        if ii==0: #begining of the column
                            guess_DC=self.rep_DC.solve_DC(guess=guess_DC, verbose=False)
                    elif jj==0 and ii==0: #not flj and first step
                        guess_DC=self.rep_DC.solve_DC(guess=guess_DC, verbose=False)

                        print('Plot')
                        eig_omegas = np.ones((len(guesses),))*np.nan
                        phi1_vec=np.linspace(0,2*np.pi, 101)
                        phi2_vec=np.linspace(0,2*np.pi,101)
                        U_mat=[]
                        for j_phi, phi2 in enumerate(phi2_vec):
                            for i_phi,phi1 in enumerate(phi1_vec):
                                Lb=self.rep_DC.dipoles[1].val
                                Lp1=self.rep_DC.dipoles[-1].val
                                Lp2=self.rep_DC.dipoles[-2].val
                                phiJ1=Lb*(phi1/Lp1+phi2/Lp2)
                                U_mat.append(([phiJ1,phi1,phi2]))
                        fig, ax=plt.subplots()
                        cl=ax.imshow(U_mat, cmap='viridis')
                        fig.colorbar(cl,ax=ax)
                        ax.set_title('First')
                        raise Exception

                     #AC checking
                    if jj==51 and ((ii==(len(vals_1)-1)) or ii==0) :
                        fig_eom,ax_eom=plt.subplots()
                        print('Current guess')
                        print(current_guesses)
                        omegas=np.linspace(2*np.pi*0,2*np.pi*15,1001)
                        self.rep_AC.display_eom(ax_eom, omegas)
                        ymin,ymax=ax_eom.get_ylim()
                        ax_eom.vlines(np.array(current_guesses)/2/np.pi,ymin,ymax)
                        ax_eom.set_title('ii = %d, jj = %d'%(ii, jj))
                    elif jj==52 and ii==0:
                        fig_eom,ax_eom=plt.subplots()
                        print('Current guess')
                        print(current_guesses)
                        omegas=np.linspace(2*np.pi*0,2*np.pi*15,1001)
                        self.rep_AC.display_eom(ax_eom, omegas)
                        ymin,ymax=ax_eom.get_ylim()
                        ax_eom.vlines(np.array(current_guesses)/2/np.pi,ymin,ymax)
                        ax_eom.set_title('ii = %d, jj = %d'%(ii, jj))


                    try:
                         eig_omegas, eig_phizpfs = self.rep_AC.solve_AC(current_guesses, verbose=False)
                    except Exception:
#                        continue_ok=False
#                        print_exc()
                        print_plot=False
                        print('NAN')
                        eig_omegas = np.ones((len(guesses),))*np.nan
#                        raise Exception
                        # DC checking
#                        phi1_vec=np.linspace(0,0.05, 101)
#                        phi2_vec=np.linspace(0,0.05,101)
#                        U_mat=[]
#                        for j_phi, phi2 in enumerate(phi2_vec):
#                            for i_phi,phi1 in enumerate(phi1_vec):
#                                Lb=self.rep_DC.dipoles[1].val
#                                Lp1=self.rep_DC.dipoles[-1].val
#                                Lp2=self.rep_DC.dipoles[-2].val
#                                phiJ1=Lb*(-phi1/Lp1+phi2/Lp2)-phi1+val1
#                                U_mat.append(self.rep_DC.U([phiJ1,phi2,phi1]))
#                        print(Lb, Lp1, Lp2)
#                        U_mat=np.array(U_mat).reshape((101,101))
#                        print('Guess DC')
#                        print(guess_DC)
#                        print(val1)
#                        print('Reconstructed guess')
#                        phi1_min=guess_DC[2]
#                        phi2_min=guess_DC[1]
#                        print([Lb*(-phi1_min/Lp1+phi2_min/Lp2)-phi1_min+val1,phi2_min, phi1_min])
#                        fig, ax=plt.subplots()
#                        print('minarray')
#                        print(np.amin(U_mat))
#                        cl=ax.imshow(U_mat-np.amin(U_mat), cmap='viridis',extent=[phi1_vec[0], phi1_vec[-1],phi2_vec[0], phi2_vec[-1]],origin='lower',norm=LogNorm())
#                        fig.colorbar(cl,ax=ax)
#                        ax.set_title('NAN')
#                        ax.plot(guess_DC[2],guess_DC[1],'or')
#
#                        #AC checking
#                        fig_eom,ax_eom=plt.subplots()
#                        print('Current guess')
#                        print(current_guesses)
#                        omegas=np.linspace(2*np.pi*0,2*np.pi*15,1001)
#                        self.rep_AC.display_eom(ax_eom, omegas)
#                        ymin,ymax=ax_eom.get_ylim()
#                        ax_eom.vlines(np.array(current_guesses)/2/np.pi,ymin,ymax)
#                        ax_eom.set_title('ii = %d, jj = %d'%(ii, jj))
##                        raise Exception
#
#                        break
    #                    eig_phizpfs=None
                        if ii==(len(vals_1)-1):
                            current_guesses=list_eig_omegas_jj[0]
                        elif not (jj==0):
                            current_guesses=matrix_eig_omegas[jj-1][ii]
                    else:
                        if not(ii==(len(vals_1)-1)):
                            current_guesses = eig_omegas
                        else: #we are going to a new line : take the begining of the line
                            current_guesses=list_eig_omegas_jj[0]
                    list_eig_omegas_jj.append(eig_omegas)
                jj+=1
                matrix_eig_omegas.append(list_eig_omegas_jj)

            matrix_eig_omegas=np.array(matrix_eig_omegas)

            # Try to get back the values with nan
            print('FIRST PASSAGE DONE')
            N2=len(vals_2)
            N1=len(vals_1)
            for jj_ind_1 in range(N2):
                for ii_ind_1 in range(N1):
                    jj_ind=N2-1-jj_ind_1
#                    jj_ind=N2-1-jj_ind_1
                    ii_ind=ii_ind_1
                    if np.isnan(matrix_eig_omegas[jj_ind][ii_ind][0]):
                        print('nan')
                        if ii_ind==0 and jj_ind==0: #top left corner
                            index_tab=[(jj_ind,ii_ind+1),(jj_ind+1,ii_ind+1),(jj_ind+1,ii_ind)]
                            try_to_solve(self,elt_1,vals_1,elt_2,vals_2,matrix_eig_omegas,jj_ind,ii_ind, index_tab)
                        elif ii_ind==0 and jj_ind==N2-1 : #top right corner
                            index_tab=[(jj_ind,ii_ind-1),(jj_ind+1,ii_ind-1),(jj_ind+1,ii_ind)]
                            try_to_solve(self,elt_1,vals_1,elt_2,vals_2,matrix_eig_omegas,jj_ind,ii_ind, index_tab)
                        elif ii_ind==N1-1 and jj_ind==0: #bottom left corner
                            index_tab=[(jj_ind-1,ii_ind),(jj_ind-1,ii_ind+1),(jj_ind,ii_ind+1)]
                            try_to_solve(self,elt_1,vals_1,elt_2,vals_2,matrix_eig_omegas,jj_ind,ii_ind, index_tab)
                        elif ii_ind==N1-1 and jj_ind==N2-1: #bottom right corner
                            index_tab=[(jj_ind-1,ii_ind),(jj_ind-1,ii_ind-1),(jj_ind,ii_ind-1)]
                            try_to_solve(self,elt_1,vals_1,elt_2,vals_2,matrix_eig_omegas,jj_ind,ii_ind, index_tab)
                        elif ii_ind==0: # left border
                            index_tab=[(jj_ind-1,ii_ind),(jj_ind-1,ii_ind+1),(jj_ind,ii_ind+1),(jj_ind+1,ii_ind+1),(jj_ind+1,ii_ind)]
                            try_to_solve(self,elt_1,vals_1,elt_2,vals_2,matrix_eig_omegas,jj_ind,ii_ind, index_tab)
                        elif ii_ind==N1-1: #right border
                            index_tab=[(jj_ind-1,ii_ind),(jj_ind-1,ii_ind-1),(jj_ind,ii_ind-1),(jj_ind+1,ii_ind-1),(jj_ind+1,ii_ind)]
                            try_to_solve(self,elt_1,vals_1,elt_2,vals_2,matrix_eig_omegas,jj_ind,ii_ind, index_tab)
                        elif jj_ind==0: #top border
                            index_tab=[(jj_ind,ii_ind-1),(jj_ind+1,ii_ind-1),(jj_ind+1,ii_ind),(jj_ind+1,ii_ind+1),(jj_ind,ii_ind+1)]
                            try_to_solve(self,elt_1,vals_1,elt_2,vals_2,matrix_eig_omegas,jj_ind,ii_ind, index_tab)
                        elif jj_ind==N2-1: #bottom border
                            index_tab=[(jj_ind,ii_ind-1),(jj_ind-1,ii_ind-1),(jj_ind-1,ii_ind),(jj_ind-1,ii_ind+1),(jj_ind,ii_ind+1)]
                            try_to_solve(self,elt_1,vals_1,elt_2,vals_2,matrix_eig_omegas,jj_ind,ii_ind, index_tab)
                        else :
                            index_tab=[(jj_ind-1,ii_ind-1),(jj_ind-1,ii_ind),(jj_ind-1,ii_ind+1),(jj_ind,ii_ind+1),(jj_ind+1,ii_ind+1),(jj_ind+1,ii_ind),(jj_ind+1,ii_ind-1),(jj_ind,ii_ind-1)]
                            try_to_solve(self,elt_1,vals_1,elt_2,vals_2,matrix_eig_omegas,jj_ind,ii_ind,index_tab)

            print('matrix size')
            print(np.array(matrix_eig_omegas).shape)
            ##Re-arrangement of the results, classified with guesses
            mat_guesses_omegas=np.moveaxis(matrix_eig_omegas,-1,0)


        return(mat_guesses_omegas)


def try_to_solve(self,elt_1,vals_1,elt_2,vals_2,matrix, jj,ii,index_neighbours):
    #no return, modify matrix internally
    number_neighbours=0
    cannot_solve=True
    guess_DC=None
    number_max_neighbours=len(index_neighbours)
    while number_neighbours<number_max_neighbours and cannot_solve:
        neighbour=index_neighbours[number_neighbours]
        if not(matrix[neighbour][0] is np.nan):
            print('neighbour is not nan')
            guesses=matrix[neighbour]
            elt_1.val=vals_1[ii]
            elt_2.val=vals_2[jj]
            guess_DC = self.rep_DC.solve_DC(guess=guess_DC, verbose=False)
            try:
                eig_omegas, eig_phizpfs = self.rep_AC.solve_AC(guesses, verbose=False)
            except Exception:
                eig_omegas = np.ones((len(guesses),))*np.nan
            else:
                cannot_solve=False
            matrix[jj][ii]=eig_omegas
        number_neighbours+=1

    if cannot_solve:
        print('not solved')
    else:
        print('solved !')
            #Plot checking
        print('Current guess')
        print(guesses)
        print(jj, ii)
        if ii==84 or ii==87:
            fig_eom,ax_eom=plt.subplots()
            omegas=np.linspace(2*np.pi*0,2*np.pi*15,1001)
            self.rep_AC.display_eom(ax_eom, omegas)
            ymin,ymax=ax_eom.get_ylim()
            ax_eom.vlines(np.array(guesses)/2/np.pi,ymin,ymax)
            ax_eom.set_title('ii = %d, jj = %d'%(ii, jj))


