#!/usr/bin/env python
# -*- coding: utf-8 -*-

#The MIT License (MIT)

#Copyright (c) 2015 James O'Grady

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.

import sys

import numpy as np
import numpy.ma as ma
import scipy.sparse
import scipy.spatial
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools as it
import time
import os
import errno
import calcforce
import cProfile
import pstats
import StringIO
#import pprofile

from PyTrilinos import Epetra
from PyTrilinos import EpetraExt
from PyTrilinos import Teuchos
from PyTrilinos import Isorropia
from PyTrilinos import NOX

# from ensight import Ensight
# VIZ_PATH='/Applications/paraview.app/Contents/MacOS/paraview'


class PlateProblem(
        NOX.Epetra.Interface.Required,
        NOX.Epetra.Interface.Jacobian
        ):
    def __init__(self,dimensions,discretization,matl_properties,bendingproperties=None,importname=None):
        """Initializes entire problem"""
        NOX.Epetra.Interface.Required.__init__(self)
        NOX.Epetra.Interface.Jacobian.__init__(self)

        #profiler = pprofile.Profile()

        t_start=time.time()

        self.verbose    = True
        self.__comm     = Epetra.PyComm()
        self.__myPID    = comm.MyPID()
        self.rank       = self.__myPID
        self.size       = comm.NumProc()

        self.time_compute=0.0


        [plate_length,plate_width,thickness,extension] = dimensions
        [nodesAcrossLength,nodesAcrossWidth,horizon] = discretization
        [shear_mod,bulk_mod,yieldstrain,mat_type] = matl_properties
        if bendingproperties != None:
            self.separatebending=True
            self.bendingrigidity=bendingproperties[0]
            self.gaussianstiffness=bendingproperties[1]

        self.plate_length = plate_length
        self.plate_width = plate_width
        self.nodesAcrossLength = nodesAcrossLength
        self.nodesAcrossWidth = nodesAcrossWidth
        self.horizon = horizon
        self.extension = extension
        self.thickness = thickness
        self.material = mat_type

        self.eps = 1.0E-6
        self.PAHHB = False

        #Private attributes
        self.__max_neighbors_returned = 200

        if importname != None:
            #Import problem grid
            self.__load_grid(importname)
        if importname == None:
            #Setup problem grid
            self.__create_grid()
        #Determine derived parameters
        self.__derive_parameters(dimensions,discretization,matl_properties)
        #Find the global family array
        self.__get_families()
        #Load balance
        self.__load_balance()
        #Initialize data
        self.__init_data()
        #Initialize BCs
        self.step_BCs(0,1)
        #Initialize the graph for coloring
        self.__init_graph()
         #Initialize the Jacobian Matrix
         #self.__init_jacobian()
        self.time_init=time.time()-t_start

    def __load_grid(self,importname):
        """Private member function that imports node centroids and volumes"""
        if self.rank == 0:
            centroids = np.load(importname)
            self.volumes = centroids['volumes']
            self.nodes = centroids['centroids']
            my_num_nodes = len(self.nodes)
            print "imported",my_num_nodes,"nodes from",importname
            ring=True
            if ring:
                anglesort = np.arctan2(self.nodes[:,2],(self.nodes[:,0]-(self.plate_length/2))).argsort()
            else:
                anglesort = self.nodes[:,0].argsort()
            self.volumes = self.volumes[anglesort]
            self.nodes = self.nodes[anglesort]
            my_num_nodes = len(self.nodes)
            numYDivs=1
            numAngleDivs=np.floor(self.size/numYDivs)
            numDivs = numYDivs*numAngleDivs
            print "making",numDivs,"blocks of nodes"
            if numDivs < self.size:
                print "Some processors will not receive any nodes"
            if numDivs > self.size:
                print "Warning! not enough processors, some nodes will not be modeled!"

            if ring:
                angle = np.arctan2(self.nodes[:,2],(self.nodes[:,0]-(self.plate_length/2)))
            else:
                angle = self.nodes[:,0]
            eps = 1E-9
            minY = np.amin(self.nodes[:,1])-eps
            maxY = np.amax(self.nodes[:,1])+eps
            yDivs = np.linspace(minY,maxY,num=numYDivs+1)
            minAngle = np.amin(angle)-eps
            maxAngle = np.amax(angle)+eps
            angleDivs = np.linspace(minAngle,maxAngle,num=numAngleDivs+1)
            print "y limits",yDivs
            if ring:
                print "angle limits",angleDivs
            else:
                print "x limits",angleDivs
            self.nodeblocks = -1.0*np.ones((numDivs,my_num_nodes),dtype=np.int32)
            block = 0
            for i in range(len(angleDivs)-1):
                angleMin = angleDivs[i]
                angleMax = angleDivs[i+1]
                for j in range(len(yDivs)-1):
                    yMin = yDivs[j]
                    yMax = yDivs[j+1]
                    [temp] = np.array(np.where(np.logical_and(
                        np.logical_and(
                            np.greater(self.nodes[:,1],yMin),
                            np.less_equal(self.nodes[:,1],yMax)),
                        np.logical_and(
                            np.greater(angle,angleMin),
                            np.less_equal(angle,angleMax)))),dtype=np.int32)

                    self.nodeblocks[block,:len(temp)]=temp
                    block = block+1
            self.nodeblocks = np.array(self.nodeblocks,dtype=np.int32)
            self.blocksizes = (self.nodeblocks != -1).sum(axis=1)
            if ring:
                print "Shifting to center on origin"
                self.nodes[:,0]-=self.plate_length/2
        else:
            self.nodes = np.array([],dtype=np.double)
            my_num_nodes = len(self.nodes)
            numDivs = 0
            self.volumes = np.array([],dtype=np.double)
        self.__global_number_of_nodes = self.__comm.SumAll(my_num_nodes)
        self.__global_number_of_blocks = self.__comm.SumAll(int(numDivs))
        return

    def __derive_parameters(self,dimensions,discretization,matl_properties):
        """Translates material/discretization properties into needed parameters"""
        [plate_length,plate_width,thickness,extension] = dimensions
        [nodesAcrossLength,nodesAcrossWidth,horizon] = discretization
        [shear_mod,bulk_mod,yieldstrain,mat_type] = matl_properties
        meshthickness = thickness
        if comm.MyPID()==0: print "only good for thickness =",meshthickness
        self.areas = self.volumes/meshthickness
        #self.grid_spacing = plate_length/(nodesAcrossLength-1)
        if comm.MyPID()==0:
            self.grid_spacing = np.mean(np.sqrt(self.areas))
        else:
            self.grid_spacing = 0.0
        self.grid_spacing = comm.MaxAll(self.grid_spacing)
        # small horizon is used to find sources for virtual points
        self.smallhorizon = max((2.0*self.grid_spacing,0.5*horizon))
        self.yieldstrain = yieldstrain
        self.is_beam = True if nodesAcrossWidth == 1 else False
        #Properties common to beams and plates
        nu0 = 1.0/3.0 #One parameter model
        #Extension stiffness E for nu=1/3
        modulus0 = 2.0*shear_mod*(1.0+nu0)
        
        # actual poisson ratio
        nu = (3*bulk_mod-2*shear_mod)/(6*bulk_mod+2*shear_mod)
        if comm.MyPID()==0: print "actual poisson ratio = ",str(nu)
        youngs_mod=3.0*bulk_mod*(1.0-2.0*nu)
        fracture_toughness=5.0E7
        fracture_energy=fracture_toughness*fracture_toughness/youngs_mod
        if self.is_beam:    
            #rectangular beam bending stiffness
            area_moment = plate_width*(thickness**3.0)/12.0
            #Bending Stiffness EI
            self.c = modulus0*area_moment
            #Extension Stiffness EA
            self.ext_modulus = modulus0*thickness*plate_width
            self.isoBendMod = 4*shear_mod*area_moment*(3.0*nu-1.0)/(1.0-nu)
            self.iso_ext_modulus = bulk_mod*thickness*((3.0*nu-1.0)/(1.0-nu))
        else:
            #plate bending stiffness
            area_moment = (thickness**3.0)/12.0
            #Bending Stiffness D = (2 G t^3)/[12(1-nu)] = (3 G t^3)/12
            if self.separatebending:
                self.c=-1.5*self.gaussianstiffness
                self.isoBendMod=6.0*self.gaussianstiffness+4.0*self.bendingrigidity
            else:
                self.c = 2*shear_mod*area_moment/(1.0-nu0)
                self.isoBendMod = 4*shear_mod*area_moment*(3.0*nu-1.0)/(1.0-nu)
            if comm.MyPID()==0: print "isotropic bending modulus = ",str(self.isoBendMod)
            #extension stiffness 8Gt
            self.ext_modulus = 8.0*shear_mod*thickness
            self.iso_ext_modulus = bulk_mod*thickness*((3.0*nu-1.0)/(1.0-nu))
            if comm.MyPID()==0: print "isotropic extension modulus = ",str(self.iso_ext_modulus)
            self.pair_crit_energy_density=0.75*fracture_energy*thickness/(self.horizon**3.0)
        
        #transverse force normalized based on bending stiffness
        self.forcenorm = (self.c*(yieldstrain/(thickness)))
        #edge force normalized based on plate in tension
        self.forcenorm2 = modulus0*yieldstrain*plate_width*thickness
        
        if (mat_type=="Elastic" or yieldstrain==0.0):
            self.criticalAngle = 2.0
        else:
            self.criticalAngle = 2.0*horizon*yieldstrain/thickness
        
        return

    def __create_grid(self):
        """Private member function that creates initial rectangular grid"""
        
        if self.rank == 0:
            #Create uniform rectangular grid
            j = np.complex(0,1)
            ext = self.extension
            grid = np.mgrid[
                -ext:
                self.plate_length+ext:
                ((1.0+2.0*ext/self.plate_length)*(self.nodesAcrossLength-1)+1)*j,
                -ext:
                self.plate_width+ext:
                ((1.0+2.0*ext/self.plate_width)*(self.nodesAcrossWidth-1)+1)*j]
            
            print "Grid has ",(len(grid[0])*len(grid[0][0]))," nodes"
            #Create x,y tuple of node positions
            self.nodes = np.array(zip(
                grid[0].ravel(),
                grid[1].ravel(),
                np.zeros_like(grid[0]).ravel()),
                dtype=np.double)
            
            my_num_nodes = len(self.nodes)
            
            if self.size<4:
                numYDivs=1
            elif self.size<16:
                numYDivs=2
            else:
                numYDivs=4
            numXDivs=np.floor(self.size/numYDivs)
            numDivs = numYDivs*numXDivs
            print "making",numDivs,"blocks of nodes"
            if numDivs < self.size:
                print "Some processors will not receive any nodes"
            if numDivs > self.size:
                print "Warning! not enough processors, some nodes will not be modeled!"
            
            eps = 1E-9
            minY = np.amin(self.nodes[:,1])-eps
            maxY = np.amax(self.nodes[:,1])+eps
            yDivs = np.linspace(minY,maxY,num=numYDivs+1)
            minX = np.amin(self.nodes[:,0])-eps
            maxX = np.amax(self.nodes[:,0])+eps
            XDivs = np.linspace(minX,maxX,num=numXDivs+1)
            print "y limits",yDivs
            print "x limits",XDivs

            numDivs=self.size
            self.nodeblocks = -1.0*np.ones((numDivs,my_num_nodes),dtype=np.int32)
            block = 0
            for i in range(len(XDivs)-1):
                XMin = XDivs[i]
                XMax = XDivs[i+1]
                for j in range(len(yDivs)-1):
                    yMin = yDivs[j]
                    yMax = yDivs[j+1]
                    [temp] = np.array(np.where(np.logical_and(
                        np.logical_and(
                            np.greater(self.nodes[:,1],yMin),
                            np.less_equal(self.nodes[:,1],yMax)),
                        np.logical_and(
                            np.greater(self.nodes[:,0],XMin),
                            np.less_equal(self.nodes[:,0],XMax)))),dtype=np.int32)

                    self.nodeblocks[block,:len(temp)]=temp
                    block = block+1

            self.nodeblocks = np.array(self.nodeblocks,dtype=np.int32)
            self.blocksizes = (self.nodeblocks != -1).sum(axis=1)


            spacing = self.plate_length/(self.nodesAcrossLength-1)

            self.volumes = spacing*spacing*self.thickness*np.ones_like(self.nodes[:,0],dtype=np.double)
        else:
            self.nodes = np.array([],dtype=np.double)
            my_num_nodes = len(self.nodes)
            numDivs = 0
            self.volumes = np.array([],dtype=np.double)
        
        self.__global_number_of_nodes = self.__comm.SumAll(my_num_nodes)
        self.__global_number_of_blocks = self.__comm.SumAll(int(numDivs))
        
        return

    def __get_families(self):
        """Find node neighborhoods, add virtual points, supports, weights"""
        if self.rank == 0:
            #Create a kdtree to do nearest neighbor search
            tree = scipy.spatial.cKDTree(self.nodes)
            if False:
                #make symmetry lists, will have no impact without symmetry BC
                self.SymSource = np.arange(len(self.nodes))
                #Find ghost nodes and their sources
                print "symmetric in x and z planes"
                for xn,x in enumerate(self.nodes):
                    if (x[0]<0.0):
                        _, zn = tree.query([-x[0],x[1],x[2]],
                            k=1, eps=0.0, p=2,
                            distance_upper_bound=(self.grid_spacing/1000.0))
                        self.SymSource[xn]=zn
                        if zn == tree.n:
                            print "Error: symmetry source not found"
                    if (x[2]<0.0):
                        _, zn = tree.query([x[0],x[1],-x[2]],
                            k=1, eps=0.0, p=2,
                            distance_upper_bound=(self.grid_spacing/1000.0))
                        self.SymSource[xn]=zn
                        if zn == tree.n:
                            print "Error: symmetry source not found"
            
            #Get all families
            if self.PAHHB:
                _, families = tree.query(self.nodes,
                    k=self.__max_neighbors_returned, eps=0.0, p=2,
                    distance_upper_bound=self.horizon+0.5*self.grid_spacing)
            else:
                _, families = tree.query(self.nodes,
                    k=self.__max_neighbors_returned, eps=0.0, p=2,
                    distance_upper_bound=self.horizon)
            #Replace the default integers at the end of the arrays with -1's
            families = np.delete(np.where(families ==  tree.n, -1,
                families),0,1)
            
            #Find the maximum length of any family, we will use this to recreate
            #the families array such that it minimizes masked entries.
            self.__max_family_length = np.max((families != -1).sum(axis=1))
            if self.PAHHB:
                print "Maximum family size including partial areas:",self.__max_family_length
            else:
                print "Maximum family size:",self.__max_family_length

            #Recast the families array to be of minimum size possible
            self.families = families[:,:self.__max_family_length]

            #Make ordered and extended families
            # ofamilies reorders the family so that
            # ofamimlies[x,0::2] and ofamilies[x,1::2] are pairs with x as pivot node
            ofamilies = np.empty_like(np.hstack((self.families,self.families)))
            ofamilies.fill(-1)

            # efamilies holds the extended families of each node, including
            # the node itself, its family members, and the nodes needed to
            # interpolate properties at its virtual family members
            efamilies = -1*np.ones(
                (self.__global_number_of_nodes,
                (self.__max_family_length+1)*self.__max_family_length),dtype=np.int32)
            efamilies[:,0]=range(self.__global_number_of_nodes)
            efamilies[:,1:self.__max_family_length+1] = (
                families[:,:self.__max_family_length])

            #Virtual points are defined wherever a perfect opposite bond is not
            # defined. Irregular meshes may have as many virtual nodes as real bonds
            vpt_estLen = self.__global_number_of_nodes*self.__max_family_length
            #vpts associated with each node
            pivot_vpts= -np.ones_like(self.families,dtype=np.int32)
             #vpt_sources = -np.ones((vpt_estLen,self.__max_family_length),dtype=np.int32)
             #vpt_distances = -np.ones((vpt_estLen,self.__max_family_length))
            vpt_sources = -np.ones((vpt_estLen,19),dtype=np.int32)
            vpt_distances = -np.ones((vpt_estLen,19))
            vpt_nodes= -np.ones((vpt_estLen,3),dtype=np.float64)
            num_vpts=0
            vpt_counter = self.__global_number_of_nodes


            #Connectivity mtx will determine which points need to be distributed
            # to each processor
            connectivity = -1*np.ones(
                (self.__global_number_of_nodes,
                (self.__max_family_length+1)*self.__max_family_length),dtype=np.int32)
            connectivity[:,:self.__max_family_length+1] = (
                efamilies[:,:self.__max_family_length+1])

            if False:
                #Jacobian connectivity will be used to determine finite-difference coloring
                jGraph = -1*np.ones(
                    (self.__global_number_of_nodes,
                    (self.__max_family_length*2+1)*self.__max_family_length),dtype=np.int32)
                jGraph[:,:self.__max_family_length+1] = (
                    efamilies[:,:self.__max_family_length+1])
            print "Build extended and virtual families..."
            try:
                for xn, fam in enumerate(self.families):
                    xloc = self.nodes[xn]
                    oindex=0
                    pivot_idx = 0
                    for index2,yn in enumerate(fam):
                        if yn == -1 :
                            continue

                        yloc = self.nodes[yn]
                        #find opposite node for pair around x
                        zloc = 2*xloc-yloc
                        _, zn = tree.query(zloc,
                            k=1, eps=0.0, p=2,
                            distance_upper_bound=(self.grid_spacing/10000.0))
                        if zn < yn: #each pair only goes in once
                            ofamilies[xn,oindex]=yn
                            ofamilies[xn,oindex+1]=zn
                            oindex+=2
                        #No opposite node
                        elif (zn == tree.n):
                            #add virtual point at opposite location
                            vpt_nodes[num_vpts]=zloc
                            pivot_vpts[xn,pivot_idx]=(num_vpts +
                                self.__global_number_of_nodes)
                            pivot_idx+=1
                            #Find real nodes from which to estimate virtual node props
                            vpt_distances[num_vpts], vpt_sources[num_vpts]=tree.query(
                                zloc,k=19, eps=0.0, p=2,
                                distance_upper_bound=2.0*self.horizon)
                            #Add support nodes to connectivity and efamily
                            addUnique(connectivity[xn],
                                vpt_sources[num_vpts][vpt_sources[num_vpts]!=tree.n])
                            addUnique(connectivity[yn],
                                vpt_sources[num_vpts][vpt_sources[num_vpts]!=tree.n])
                            addUnique(efamilies[xn],
                                vpt_sources[num_vpts][vpt_sources[num_vpts]!=tree.n])
                            #Add virtual node to ofamily
                            ofamilies[xn,oindex]=yn
                            ofamilies[xn,oindex+1]= (num_vpts +
                                self.__global_number_of_nodes)
                            oindex+=2
                            num_vpts=num_vpts+1



            except:
                print "Family search error"

            print "Created ",num_vpts," virtual points"
            if num_vpts>0:
                print "Virtual x min =",np.amin(vpt_nodes[vpt_nodes[:,0]>-1,0]),
                print ", x max =",np.amax(vpt_nodes[:,0])
                print "Virtual y min =",np.amin(vpt_nodes[vpt_nodes[:,1]>-1,1]),
                print ", y max =",np.amax(vpt_nodes[:,1])
            self.vpt_nodes = vpt_nodes[:num_vpts]
            vpt_sources = vpt_sources[:num_vpts]

            print "Calculating support weights"
            if self.is_beam:    
                #support weight for linear inter/extra-polation
                vpt_sources=vpt_sources[:,:2]
                Anodes = self.nodes[vpt_sources[:,0]]
                Bnodes = self.nodes[vpt_sources[:,1]]
                Cnodes = self.vpt_nodes
                AB = Bnodes-Anodes
                ABnorm = np.sqrt(np.sum(AB*AB,axis=1))
                AC = Cnodes-Anodes
                BC = Cnodes-Bnodes
                ACp = AB*((np.sum(AB*AC,axis=1)/(ABnorm**2.0)))[...,None]
                ACpnorm = np.sqrt(np.sum(ACp*ACp,axis=1))
                BCp = AB*((np.sum(AB*BC,axis=1)/(ABnorm**2.0)))[...,None]
                BCp_norm = np.sqrt(np.sum(BCp*BCp,axis=1))
                wB = np.where(BCp_norm<ABnorm,ACpnorm/ABnorm,-ACpnorm/ABnorm)
                wA = 1-wB
                vpt_weights = np.transpose([wA,wB])

            else:
                #Barycentric/planar interpolation weight calculation
                A_nodes = self.nodes[vpt_sources[:,0],:]
                B_nodes = self.nodes[vpt_sources[:,1],:]
                C_nodes = self.nodes[vpt_sources[:,2],:]
                repeat = 2
                while repeat>0:
                    #Replace C supports where A,B,C are colinear
                    AB = B_nodes-A_nodes
                    AC = C_nodes-A_nodes
                    BA = -AB
                    BC = C_nodes-B_nodes
                    CA = -AC
                    CB = -BC
                    A_cross = np.cross(BC,BA)
                    sinSQ = (A_cross*A_cross).sum(axis=-1)/((BC*BC).sum(axis=-1)*(BA*BA).sum(axis=-1))
                    triangular = (sinSQ>0.3) #corresponds to about 33 degrees
                    if np.all(triangular):
                        break
                    numlinear = np.where(triangular,0.0,1.0).sum()
                    repeat = repeat+1
                    print "fixing",numlinear,"colinear supports",repeat
                    vpt_sources[:,2]=np.where(
                        triangular,vpt_sources[:,2],vpt_sources[:,repeat])
                    C_nodes = self.nodes[vpt_sources[:,2],:]

                B_cross = np.cross(CA,CB)
                C_cross = np.cross(AB,AC)
                A_cross = A_cross/(np.sqrt((A_cross*A_cross).sum(axis=-1))[:,None])
                B_cross = B_cross/(np.sqrt((B_cross*B_cross).sum(axis=-1))[:,None])
                C_cross = C_cross/(np.sqrt((C_cross*C_cross).sum(axis=-1))[:,None])
                AX = A_nodes-self.vpt_nodes
                BX = B_nodes-self.vpt_nodes
                CX = C_nodes-self.vpt_nodes

                WA = (BX*np.cross(BC,A_cross)).sum(axis=-1)
                WB = (CX*np.cross(CA,B_cross)).sum(axis=-1)
                WC = (AX*np.cross(AB,C_cross)).sum(axis=-1)
                vpt_sources=vpt_sources[:,:3]
                vpt_weights=np.transpose([WA,WB,WC])/((WA+WB+WC)[:,None])

            if num_vpts>0:
                print "Vpt weight min =",np.amin(vpt_weights),
                print "max =",np.amax(vpt_weights)

            self.__max_connectivity_length=np.max((connectivity != -1).sum(axis=1))
            self.connectivity=connectivity[:,:self.__max_connectivity_length]

            self.__max_pivot_vpts = np.max((pivot_vpts != -1).sum(axis=1))
            #Recast the pivot vpts array to be of minimum size possible
            self.pivot_vpts = pivot_vpts[:,:self.__max_pivot_vpts]

            self.__max_ofamily_length = np.max((ofamilies != -1).sum(axis=1))
            #Recast the ofamilies array to be of minimum size possible
            self.ofamilies = ofamilies[:,:self.__max_ofamily_length]

            self.__max_efamily_length = np.max((efamilies != -1).sum(axis=1))
            #Recast the efamilies array to be of minimum size possible
            self.efamilies = efamilies[:,:self.__max_efamily_length]

            self.__max_vsources = np.max((vpt_sources != -1).sum(axis=1))
            #Recast the vsources array to be of minimum size possible
            self.vpt_sources = vpt_sources[:,:self.__max_vsources]
            self.vpt_weights = vpt_weights[:,:self.__max_vsources]

            if False:
                #Fill Jacobian graph with the efamilies
                for x,efamX in enumerate(self.efamilies):
                    graphX = np.unique(self.efamilies[efamX,:])
                    jGraph[x,:(len(graphX)-1)]=graphX[1:]

                self.__max_jGraph = np.max((jGraph != -1).sum(axis=1))
                #Recast the jGraph array to be of minimum size possible
                self.jGraph = jGraph[:,:self.__max_vsources]
        else:
            #Setup empty data on other ranks
            self.__max_family_length = 0
            self.families = np.array([],dtype=np.int32)
            self.__max_ofamily_length = 0
            self.ofamilies = np.array([],dtype=np.int32)
            self.__max_efamily_length = 0
            self.efamilies = np.array([],dtype=np.int32)
            self.__max_connectivity_length = 0
            self.connectivity = np.array([],dtype=np.int32)
            self.__max_pivot_vpts = 0
            self.pivot_vpts = np.array([],dtype=np.int32)

            self.vpt_nodes = np.array([],dtype=np.int32)
            self.__max_vsources = 0
            self.vpt_sources=np.array([],dtype=np.int32)
            self.vpt_weights=np.array([],dtype=np.float)
            if False:
                self.SymSource=np.array([],dtype=np.float)
                self.jGraph = np.array([],dtype=np.int32)
                self.__max_jGraph = 0
        return

    def __load_balance(self):
        """Load balancing function."""
        #Create node map with all the data on the rank 0 processor
        unbalanced_map = Epetra.Map(self.__global_number_of_nodes,
                len(self.nodes), 0, self.__comm)
        self.unbalanced_map = unbalanced_map
        #Create and populate distributed Epetra vector to the hold the
        #unbalanced positions and areas.
        my_nodes = Epetra.MultiVector(unbalanced_map, 3)
        my_nodes[:] = self.nodes.T
        my_areas = Epetra.Vector(unbalanced_map)
        my_areas[:] = self.areas
         #my_SymSource = Epetra.Vector(unbalanced_map)
         #my_SymSource[:] = self.SymSource
        #Create and populate an Epetra mulitvector to store the families data
        self.__max_family_length = self.__comm.MaxAll(self.__max_family_length)
        my_families = Epetra.MultiVector(unbalanced_map,
                self.__max_family_length)
        my_families[:] = self.families.T

        #Create and populate an Epetra mulitvector to store the ofamilies data
        self.__max_ofamily_length = self.__comm.MaxAll(self.__max_ofamily_length)
        my_ofamilies = Epetra.MultiVector(unbalanced_map,
                self.__max_ofamily_length)
        my_ofamilies[:] = self.ofamilies.T

        #Create and populate an Epetra mulitvector to store the efamilies data
        self.__max_efamily_length = self.__comm.MaxAll(self.__max_efamily_length)
        my_efamilies = Epetra.MultiVector(unbalanced_map,
                self.__max_efamily_length)
        my_efamilies[:] = self.efamilies.T

        #Create and populate an Epetra mulitvector to store the connectivity data
        self.__max_connectivity_length = self.__comm.MaxAll(self.__max_connectivity_length)
        my_connectivity = Epetra.MultiVector(unbalanced_map,
                self.__max_connectivity_length)
        my_connectivity[:] = self.connectivity.T

        #Create and populate an Epetra mulitvector to store the connectivity data
        self.max_pivot_vpts = self.__comm.MaxAll(self.__max_pivot_vpts)
        my_pivot_vpts = Epetra.MultiVector(unbalanced_map,
                self.max_pivot_vpts)
        my_pivot_vpts[:] = self.pivot_vpts.T


        if False:
            #Old isorropia Load balance
            if self.rank == 0: print "Load balancing...",
            #Create Teuchos parameter list to pass parameter to ZOLTAN for load
            #balancing
            parameter_list = Teuchos.ParameterList()
            parameter_list.set("Partitioning Method","RCB")
            if not self.verbose:
                parameter_sublist = parameter_list.sublist("ZOLTAN")
                parameter_sublist.set("DEBUG_LEVEL", "0")
            #Create a partitioner to load balance the grid
            partitioner = Isorropia.Epetra.Partitioner(my_nodes, parameter_list)
            #And a redistributer
            redistributer = Isorropia.Epetra.Redistributor(partitioner)
            #Redistribute nodes
            self.my_nodes_balanced = redistributer.redistribute(my_nodes)

            #The new load balanced map
            self.balanced_map = Epetra.Map(self.__global_number_of_nodes,
                    self.my_nodes_balanced.Map().NumMyElements(), 0, self.__comm)
            balanced_map = self.get_balanced_map()

            #Create importer and exporters to move data between balanced and
            #unbalanced maps
            self.importer = Epetra.Import(balanced_map, unbalanced_map)
            self.exporter = Epetra.Export(balanced_map, unbalanced_map)
        
        if self.rank == 0:
            print "Load balancing using blocks...",
             #tempSymSource=np.array(self.SymSource,dtype=np.int32)
        if self.rank != 0:
            self.nodeblocks = np.empty((self.__global_number_of_blocks,self.__global_number_of_nodes),dtype=np.int32)
             #tempSymSource=np.empty((self.__global_number_of_nodes),dtype=np.int32)
        comm.Broadcast(self.nodeblocks,0)
         #comm.Broadcast(tempSymSource,0)
        myGIDS = np.array(self.nodeblocks[self.rank],dtype=np.int32)
        myGIDS = myGIDS[np.where(myGIDS>-1)]

        #The new load balanced map
        self.balanced_map = Epetra.Map(self.__global_number_of_nodes,
                myGIDS, 0, self.__comm)
        balanced_map = self.get_balanced_map()

        #Create importer and exporters to move data between balanced and
        #unbalanced maps
        self.importer = Epetra.Import(balanced_map, unbalanced_map)
        self.exporter = Epetra.Export(balanced_map, unbalanced_map)

        self.my_nodes_balanced = Epetra.MultiVector(balanced_map,3)
        self.my_nodes_balanced.Import(my_nodes,self.importer, Epetra.Insert)

        #Create map for 3 degrees of freedom per node in one vector
        dof_map_Xids = np.array(3*balanced_map.MyGlobalElements())
        dof_map_GIDs = np.sort(np.concatenate((dof_map_Xids,dof_map_Xids+1,dof_map_Xids+2)))
        global_dof_map_elements = 3*self.__global_number_of_nodes
        self.balanced_dof_map = Epetra.Map(global_dof_map_elements,
                dof_map_GIDs, 0, self.__comm)

        #Create distributed vectors to store the balanced node positions
        self.my_x = Epetra.Vector(balanced_map)
        self.my_y = Epetra.Vector(balanced_map)
        self.my_z = Epetra.Vector(balanced_map)
        self.my_area = Epetra.Vector(balanced_map)
         #self.my_SymSource = Epetra.Vector(balanced_map)
        #Import the balanced node positions and family information
        self.my_x.Import(my_nodes[0],self.importer, Epetra.Insert)
        self.my_y.Import(my_nodes[1],self.importer, Epetra.Insert)
        self.my_z.Import(my_nodes[2],self.importer, Epetra.Insert)
        self.my_area.Import(my_areas,self.importer, Epetra.Insert)
         #self.my_SymSource.Import(my_SymSource,self.importer, Epetra.Insert)
        my_families_balanced = Epetra.MultiVector(balanced_map,
                self.__max_family_length)
        my_families_balanced.Import(my_families,self.importer, Epetra.Insert)

        my_ofamilies_balanced = Epetra.MultiVector(balanced_map,
                self.__max_ofamily_length)
        my_ofamilies_balanced.Import(my_ofamilies,self.importer, Epetra.Insert)

        my_efamilies_balanced = Epetra.MultiVector(balanced_map,
                self.__max_efamily_length)
        my_efamilies_balanced.Import(my_efamilies,self.importer, Epetra.Insert)

        my_connectivity_balanced = Epetra.MultiVector(balanced_map,
                self.__max_connectivity_length)
        my_connectivity_balanced.Import(my_connectivity,self.importer, Epetra.Insert)

        my_pivot_vpts_balanced = Epetra.MultiVector(balanced_map,
                self.max_pivot_vpts)
        my_pivot_vpts_balanced.Import(my_pivot_vpts,self.importer, Epetra.Insert)
        
        #Convert to integer data type for indexing purposes later
        self.my_families = np.array(my_families_balanced.T, dtype=np.int32)
        self.my_ofamilies = np.array(my_ofamilies_balanced.T, dtype=np.int32)
        self.my_efamilies = np.array(my_efamilies_balanced.T, dtype=np.int32)
        self.my_connectivity = np.array(my_connectivity_balanced.T, dtype=np.int32)
        self.my_pivot_vpts = np.array(my_pivot_vpts_balanced.T, dtype=np.int32)


        #Create a flattened list of all family global indices (locally owned
        #+ ghosts)
        my_global_ids_required = np.unique(
                self.my_efamilies[self.my_efamilies != -1])
         #my_global_ids_required = np.array(np.unique(np.concatenate(
             #(my_global_ids_required,
             #tempSymSource[my_global_ids_required]))),dtype=np.int32)
        #Create a list of locally owned global ids
        my_owned_ids = np.array(balanced_map.MyGlobalElements())
        #And its length
        self.my_num_owned = len(my_owned_ids)
        #The ghost indices required by the local processor is the relative
        #complement of my_global_ids_required and my_owned_ids
        my_ghost_ids = np.setdiff1d(my_global_ids_required, my_owned_ids)
        #Get total length of worker array, this is len(owned) + len(ghosts)
        #summed over all processors
        length_of_global_worker_arr = self.__comm.SumAll(len(my_owned_ids)
                + len(my_ghost_ids))
        #Worker ids
        my_worker_ids = np.array(np.concatenate((my_owned_ids, my_ghost_ids)),
                dtype=np.int32)
        self.numWorkers = len(my_worker_ids)
        #Create the map that will be used by worker vectors
        self.my_worker_map = Epetra.Map(length_of_global_worker_arr,my_worker_ids, 0, self.__comm)
        #Create worker map for 3 degrees of freedom per node in one vector
        dof_map_Xids = np.array(3*my_worker_ids)
        dof_map_Yids = dof_map_Xids+1
        dof_map_Zids = dof_map_Xids+2
        dof_map_GIDs = np.ravel(np.vstack((dof_map_Xids,dof_map_Yids,dof_map_Zids)).T)
        global_dof_map_elements = 3*length_of_global_worker_arr
        self.dof_worker_map = Epetra.Map(global_dof_map_elements,
                dof_map_GIDs, 0, self.__comm)

        #Create the worker import/export operators to move data between the grid
        #data and the worker data
        self.worker_importer = Epetra.Import(self.my_worker_map, balanced_map)
        self.worker_exporter = Epetra.Export(self.my_worker_map, balanced_map)

        self.dof_worker_importer = Epetra.Import(self.dof_worker_map, self.balanced_dof_map)
        self.dof_worker_exporter = Epetra.Export(self.dof_worker_map, self.balanced_dof_map)


        #Set up unbalanced and balanced virtual point maps and
        #distribute virtual points/supports/weights
        self.__global_num_vpts = self.__comm.SumAll(len(self.vpt_nodes))

        vpts_required = self.my_pivot_vpts.flatten()
        vpts_required = vpts_required[vpts_required != -1]-self.__global_number_of_nodes
        self.numVirtual = len(vpts_required)
        self.numAugmented = self.numWorkers + self.numVirtual
        unbalanced_vpt_map = Epetra.Map(self.__global_num_vpts,
            len(self.vpt_nodes), 0, self.__comm)

        balanced_vpt_map = Epetra.Map(self.__global_num_vpts,
            vpts_required,
            0, self.__comm)
        self.balanced_vpt_map = balanced_vpt_map
        virtual_importer = Epetra.Import(unbalanced_vpt_map, balanced_vpt_map)

        my_vnodes = Epetra.MultiVector(unbalanced_vpt_map, 3)
        my_vnodes[:] = self.vpt_nodes.T
        my_vnodes_balanced = Epetra.MultiVector(balanced_vpt_map,3)
        my_vnodes_balanced.Export(my_vnodes,virtual_importer,Epetra.Insert)
        self.my_vnodes = np.array(my_vnodes_balanced.T, dtype=np.float64)

        self.__max_vsources = self.__comm.MaxAll(self.__max_vsources)
        my_vsources = Epetra.MultiVector(unbalanced_vpt_map,self.__max_vsources)
        my_vsources[:] = self.vpt_sources.T
        my_vsources_balanced = Epetra.MultiVector(balanced_vpt_map,
            self.__max_vsources)
        my_vsources_balanced.Export(my_vsources,virtual_importer,Epetra.Insert)
        self.my_vsources_local = np.array(my_vsources_balanced.T, dtype=np.int32)

        for row,sources in enumerate(self.my_vsources_local):
            for col,source in enumerate(sources):
                if source ==-1:
                    continue
                else:
                    self.my_vsources_local[row,col]=self.my_worker_map.LID(
                        self.my_vsources_local[row,col])

        my_vweights = Epetra.MultiVector(unbalanced_vpt_map,self.__max_vsources)
        my_vweights[:] = self.vpt_weights.T
        my_vweights_balanced = Epetra.MultiVector(balanced_vpt_map,
            self.__max_vsources)
        my_vweights_balanced.Export(my_vweights,virtual_importer,Epetra.Insert)
        self.my_vweights = np.array(my_vweights_balanced.T, dtype=np.float)
        if self.rank==0: print "Done balancing"
        return

    def __init_data(self):
        """
            Initialize and compute worker arrays and data defined only in the
            reference position.
        """
        #Create worker vectors (owned + ghosts)
        self.my_x_worker = Epetra.Vector(self.my_worker_map)
        self.my_y_worker = Epetra.Vector(self.my_worker_map)
        self.my_z_worker = Epetra.Vector(self.my_worker_map)
        self.my_area_worker = Epetra.Vector(self.my_worker_map)
        #Import the needed components for local operations
        self.my_x_worker.Import(self.my_x, self.worker_importer, Epetra.Insert)
        self.my_y_worker.Import(self.my_y, self.worker_importer, Epetra.Insert)
        self.my_z_worker.Import(self.my_z, self.worker_importer, Epetra.Insert)
        self.my_area_worker.Import(self.my_area, self.worker_importer, Epetra.Insert)
        if False:
            my_SymSource_worker = Epetra.Vector(self.my_worker_map)
            my_SymSource_worker.Import(self.my_SymSource, self.worker_importer, Epetra.Insert)
            self.my_xSym_owned = np.where(self.my_x<0.0)
            self.my_ySym_owned = np.where(self.my_y<0.0)
            self.my_zSym_owned = np.where(self.my_z<0.0)
            print self.rank,"shape my_xSym_owned",np.shape(self.my_xSym_owned[0])
            self.my_allSym_owned  = np.unique(np.concatenate(
                (self.my_xSym_owned[0],self.my_ySym_owned[0],self.my_zSym_owned[0])))
            self.my_xSym_worker = np.where(self.my_x_worker<0.0)
            self.my_ySym_worker = np.where(self.my_y_worker<0.0)
            self.my_zSym_worker = np.where(self.my_z_worker<0.0)
            self.my_SymSource_local = [self.my_worker_map.LID(np.int(source)) for source in my_SymSource_worker]
        #Convert the global node ids in the family array to local ids
        self.my_families_local = np.array([self.my_worker_map.LID(i)
            for i in self.my_families.flatten()])
        #Mask local family array
        self.my_families_local.shape = (len(self.my_families),-1)
        self.my_families_local = ma.masked_equal(self.my_families_local, -1)
        self.my_families_local.harden_mask()

        #Create matrix that will take real node displacements and spit out displacements
        #for real and virtual nodes
        sourcetmp = self.my_vsources_local[:]
        desttmp = np.empty_like(sourcetmp)
        desttmp[:]=np.arange(self.numWorkers,self.numWorkers+len(desttmp))[:,None]
        sourcetmp=ma.masked_equal(sourcetmp,-1)
        desttmp = ma.array(desttmp,mask=sourcetmp.mask)
        wtmp = ma.array(self.my_vweights,mask=sourcetmp.mask)

        builderRow = np.concatenate(
            (np.arange(self.numWorkers),desttmp.compressed()))
        builderCol = np.concatenate(
            (np.arange(self.numWorkers),sourcetmp.compressed()))
        builderDat = np.concatenate(
            (np.ones((self.numWorkers)),wtmp.compressed()))

        self.augmentbuilder=scipy.sparse.csr_matrix(
            (builderDat,(builderRow,builderCol)),
            shape=(len(self.my_vsources_local)+self.numWorkers,self.numWorkers))

        #Convert the global node ids in the ofamily array to local ids
        self.my_ofamilies_local=self.my_ofamilies[:]
        for row,pairs in enumerate(self.my_ofamilies):
            for col,GID in enumerate(pairs):
                if GID == -1:
                    self.my_ofamilies_local[row,col]=-1
                elif GID < self.__global_number_of_nodes:
                    LID=self.my_worker_map.LID(GID)
                    self.my_ofamilies_local[row,col]=LID
                else:
                    LID=self.balanced_vpt_map.LID(GID-self.__global_number_of_nodes)
                    self.my_ofamilies_local[row,col]=(self.numWorkers
                        + LID)
        self.my_ofamilies_local = ma.masked_equal(self.my_ofamilies_local, -1)
        self.my_ofamilies_local.harden_mask()

        #create augmented position/area vectors
        my_x_augmented = np.concatenate((self.my_x_worker[:],self.my_vnodes[:,0]))
        my_y_augmented = np.concatenate((self.my_y_worker[:],self.my_vnodes[:,1]))
        my_z_augmented = np.concatenate((self.my_z_worker[:],self.my_vnodes[:,2]))
        self.my_area_augmented = np.concatenate(
            (self.my_area_worker[:],np.zeros_like(self.my_vnodes[:,2])))
        self.my_augmented_reference = np.transpose(
            [my_x_augmented,my_y_augmented,my_z_augmented])


        #use family and ofamily data to make bond and pair lists
        famsizes = np.array([ (row > -1).sum()
            for row in self.my_families_local], dtype=np.int32)
        xplength = famsizes.sum()

        self.radius = Epetra.Vector(self.balanced_map)
        self.radius.PutScalar(self.grid_spacing)

        self.my_bond_x_local = np.empty([xplength], dtype=np.int32)
        self.my_bond_p_local = np.empty([xplength], dtype=np.int32)


        counter = 0
        for x,fam in enumerate(self.my_families_local):
            for p in fam.compressed():
                self.my_bond_x_local[counter] = x
                self.my_bond_p_local[counter] = p
                counter = counter + 1

        my_ref_bonds = np.transpose(
                [my_x_augmented[self.my_bond_p_local] -
                my_x_augmented[self.my_bond_x_local],
                my_y_augmented[self.my_bond_p_local] -
                my_y_augmented[self.my_bond_x_local],
                my_z_augmented[self.my_bond_p_local] -
                my_z_augmented[self.my_bond_x_local]])

        self.my_ref_bondLengths = np.sqrt(np.sum(my_ref_bonds*my_ref_bonds,axis=-1))


        ofamsizes = np.array([ (row > -1).sum()
            for row in self.my_ofamilies_local], dtype=np.int32)
        xpqlength = (ofamsizes.sum())/2

        self.my_bondpair_x_local = np.empty([xpqlength], dtype=np.intc)
        self.my_bondpair_p_local = np.empty([xpqlength], dtype=np.intc)
        self.my_bondpair_q_local = np.empty([xpqlength], dtype=np.intc)

        counter = 0
        for x,ofamily in enumerate(self.my_ofamilies_local):
            for p,q in zip(ofamily.compressed()[::2],ofamily.compressed()[1::2]):
                self.my_bondpair_x_local[counter] = x
                self.my_bondpair_p_local[counter] = p
                self.my_bondpair_q_local[counter] = q
                counter = counter + 1

        #put pairs with virtual nodes at end of list
        virtualsort = self.my_bondpair_q_local.argsort()
        self.sort=virtualsort
        self.unsort = virtualsort.argsort()
        self.my_bondpair_x_local = self.my_bondpair_x_local[virtualsort]
        self.my_bondpair_p_local = self.my_bondpair_p_local[virtualsort]
        self.my_bondpair_q_local = self.my_bondpair_q_local[virtualsort]
        self.numRealPairs = np.where(self.my_bondpair_q_local>=self.numWorkers,0.0,1.0).sum().astype(np.intc)
        self.my_realpair_q_local = np.array(self.my_bondpair_q_local[:self.numRealPairs],copy=True)
        self.my_virtpair_q_local = np.array(self.my_bondpair_q_local[self.numRealPairs:],copy=True)
        if not self.is_beam:
            self.my_vsources_A_local= self.my_vsources_local[:,0]
            self.my_vweights_A = self.my_vweights[:,0]
            self.my_vsources_B_local= self.my_vsources_local[:,1]
            self.my_vweights_B = self.my_vweights[:,1]
            self.my_vsources_C_local= self.my_vsources_local[:,2]
            self.my_vweights_C = self.my_vweights[:,2]
        self.my_vpt_pivot_local = self.my_bondpair_x_local[self.numRealPairs:]

        #vnodes not in the plane defined by their supports have an offset
        self.my_vpt_offset=self.my_vnodes-(
            self.my_vweights_A[:,None]*self.my_augmented_reference[self.my_vsources_A_local]+
            self.my_vweights_B[:,None]*self.my_augmented_reference[self.my_vsources_B_local]+
            self.my_vweights_C[:,None]*self.my_augmented_reference[self.my_vsources_C_local]
            )

        my_bondpair_ref_P = np.transpose(
            [my_x_augmented[self.my_bondpair_p_local] -
            my_x_augmented[self.my_bondpair_x_local],
            my_y_augmented[self.my_bondpair_p_local] -
            my_y_augmented[self.my_bondpair_x_local],
            my_z_augmented[self.my_bondpair_p_local] -
            my_z_augmented[self.my_bondpair_x_local]])

        my_bondpair_ref_Q = np.transpose(
            [my_x_augmented[self.my_bondpair_q_local] -
            my_x_augmented[self.my_bondpair_x_local],
            my_y_augmented[self.my_bondpair_q_local] -
            my_y_augmented[self.my_bondpair_x_local],
            my_z_augmented[self.my_bondpair_q_local] -
            my_z_augmented[self.my_bondpair_x_local]])

        #Create lists of bond indices for each pair for energy/failure coupling
        self.bond1=np.empty((xpqlength),dtype=np.intc)
        self.bond2=np.empty((self.numRealPairs),dtype=np.intc)
        if self.material != "elastic":
            for i in range(xpqlength):
                x=self.my_bondpair_x_local[i]
                p=self.my_bondpair_p_local[i]
                q=self.my_bondpair_q_local[i]
                bondindex=np.nonzero(np.logical_and(self.my_bond_x_local==x,self.my_bond_p_local==p))
                assert len(bondindex)==1
                self.bond1[i]=bondindex[0]
            for i in range(self.numRealPairs):
                x=self.my_bondpair_x_local[i]
                p=self.my_bondpair_p_local[i]
                q=self.my_bondpair_q_local[i]
                bondindex=np.nonzero(np.logical_and(self.my_bond_x_local==x,self.my_bond_p_local==q))
                assert len(bondindex)==1
                self.bond2[i]=bondindex[0]

        #Compute bond extension and bond-pair bending coefficients
        #Bond coefficients
        my_bondLengths = np.sqrt(np.sum(my_ref_bonds*my_ref_bonds,axis=-1))
        my_bond_x_area = self.my_area_augmented[self.my_bond_x_local]
        my_bond_p_area = self.my_area_augmented[self.my_bond_p_local]

        my_P_bondLengths = np.sqrt(np.sum(my_bondpair_ref_P*my_bondpair_ref_P,axis=-1))
        my_x_area = self.my_area_augmented[self.my_bondpair_x_local]
        my_pq_area = (self.my_area_augmented[self.my_bondpair_p_local]+
            self.my_area_augmented[self.my_bondpair_q_local])

        if self.is_beam:    
            #Beam Coefficients
            ext_weights = my_bondLengths*my_bond_p_area
            if self.rank==0:print "Weighted for linear elasticity using PA-HHB"
            ext_weights = my_bond_p_area/my_bondLengths
            ext_m_partials = ext_weights*my_bondLengths*my_bondLengths
            ext_m_total = np.bincount(self.my_bond_x_local,ext_m_partials)
            ext_m_dist = ext_m_total[self.my_bond_x_local]

            self.my_extension_alpha = np.divide(
                self.ext_modulus*my_bond_x_area,ext_m_dist)
            self.my_extension_stiffness = ext_weights*self.my_extension_alpha

            self.iso_ext_coefficients = (2.0*ext_weights*my_bondLengths/ext_m_dist)
            self.my_iso_ext_stiffness = (self.iso_ext_modulus*self.iso_ext_coefficients*
                (self.grid_spacing**2.0))

            #Bond-pair coefficients
            my_x_area = self.my_area_augmented[self.my_bondpair_x_local]
            my_pq_area = (self.my_area_augmented[self.my_bondpair_p_local]+
                self.my_area_augmented[self.my_bondpair_q_local])

            my_P_bondLengths = np.sqrt(np.sum(my_bondpair_ref_P*my_bondpair_ref_P,axis=-1))
            widths = np.ones_like(my_P_bondLengths)
            widthweights = my_P_bondLengths*my_pq_area
            if self.rank==0:print "Weighted for linear elasticity only"
            widthweights = widths*my_pq_area/my_P_bondLengths

            my_m_partials = my_P_bondLengths*my_P_bondLengths*widthweights
            my_m_total = np.bincount(self.my_bondpair_x_local,my_m_partials)
            if self.rank==0:
                print "bend_m max", my_m_total.max()
                print "bend_m min", my_m_total.min()
            my_m_dist = my_m_total[self.my_bondpair_x_local]

            my_pair_alpha = np.divide((my_x_area)*self.c,my_m_dist)
            self.my_pair_stiffness = np.multiply(widthweights,my_pair_alpha)

            iso_m_partials = widthweights
            iso_m_total = 2.0*np.bincount(self.my_bondpair_x_local,iso_m_partials)
            iso_m_dist = iso_m_total[self.my_bondpair_x_local]

            self.iso_pair_coefficients = (2.0*widthweights/
                (my_P_bondLengths*my_P_bondLengths*iso_m_dist))
            self.my_isoBend_stiffness = (self.iso_pair_coefficients*
                self.isoBendMod*(my_x_area))
        else:
            #Plate Coefficients
            ext_weights = my_bondLengths*my_bond_p_area
            if self.PAHHB:
                if self.rank==0:print "Weighted for linear elasticity using PA-HHB"
                bond_partial_areas = np.where(
                        my_bondLengths<self.horizon-0.5*self.grid_spacing,
                        1.0,0.5+(self.horizon-my_bondLengths)/self.grid_spacing)
                my_bond_p_area = my_bond_p_area*bond_partial_areas
            else:
                if self.rank==0:print "Weighted for linear elasticity"
            ext_weights = my_bond_p_area
            ext_m_partials = ext_weights*my_bondLengths*my_bondLengths
            ext_m_total = np.bincount(self.my_bond_x_local,ext_m_partials)
            ext_m_dist = ext_m_total[self.my_bond_x_local]
            self.my_extension_alpha = np.divide(
                self.ext_modulus*my_bond_x_area,ext_m_dist)
            self.my_extension_stiffness = ext_weights*self.my_extension_alpha
            self.iso_ext_coefficients = (2.0*ext_weights*my_bondLengths/ext_m_dist)
            self.my_iso_ext_stiffness = (self.iso_ext_modulus*self.iso_ext_coefficients*
                (self.grid_spacing**2.0))
             #self.my_iso_ext_stiffness = (self.iso_ext_modulus*self.iso_ext_coefficients*
                 #my_bond_x_area)
            #Bond-pair coefficients
            my_x_area = self.my_area_augmented[self.my_bondpair_x_local]
            my_pq_area = (self.my_area_augmented[self.my_bondpair_p_local]+
                self.my_area_augmented[self.my_bondpair_q_local])

            my_P_bondLengths = np.sqrt(np.sum(my_bondpair_ref_P*my_bondpair_ref_P,axis=-1))
            if self.PAHHB:
                pair_partial_areas = np.where(
                        my_P_bondLengths<self.horizon-0.5*self.grid_spacing,
                        1.0,0.5+(self.horizon-my_P_bondLengths)/self.grid_spacing)
                my_pq_area = my_pq_area*pair_partial_areas
            widths = np.ones_like(my_P_bondLengths)
            widthweights = my_P_bondLengths*my_pq_area
             #if self.rank==0:print "Weighted for linear elasticity only"
            widthweights = widths*my_pq_area

            my_m_partials = my_P_bondLengths*my_P_bondLengths*widthweights
            my_m_total = np.bincount(self.my_bondpair_x_local,my_m_partials)
            my_m_dist = my_m_total[self.my_bondpair_x_local]
            my_pair_alpha = np.divide(16.0*(my_x_area)*self.c/3.0,my_m_dist)
            self.my_pair_stiffness = np.multiply(widthweights,my_pair_alpha)/2.0
            #divide by two because each pair gets applied to both p and q
            iso_m_partials = widthweights
            iso_m_total = 2.0*np.bincount(self.my_bondpair_x_local,iso_m_partials)
            iso_m_dist = iso_m_total[self.my_bondpair_x_local]
            self.iso_pair_coefficients = (2.0*widthweights/
                (my_P_bondLengths*my_P_bondLengths*iso_m_dist))
            self.my_isoBend_stiffness = (self.iso_pair_coefficients*
                self.isoBendMod*(my_x_area))
            self.pair_critical_energy = self.pair_crit_energy_density*my_x_area*my_pq_area
        
        #Form maps to bring all bondhealth/pairhealth to PID 0 and back
        num_bonds_local=len(my_bondLengths)
        all_num_bonds=self.__comm.GatherAll(num_bonds_local)
        my_first_global_index=np.add.accumulate(all_num_bonds)
        total_num_bonds=np.sum(all_num_bonds)
        
        if self.rank==0:
            unbalanced_num=total_num_bonds
            balanced_GEs=np.arange(my_first_global_index[self.rank-1],dtype=np.int32)
        else:
            unbalanced_num=0
            balanced_GEs=np.arange(my_first_global_index[self.rank-1],my_first_global_index[self.rank-1],dtype=np.int32)
        
        self.bondmap_unbalanced=Epetra.Map(total_num_bonds,unbalanced_num,0,self.__comm)
        self.bondmap_balanced=Epetra.Map(total_num_bonds,balanced_GEs,0,self.__comm)
        
        self.bondimporter = Epetra.Import(self.bondmap_balanced,self.bondmap_unbalanced)
        self.bondexporter = Epetra.Export(self.bondmap_balanced,self.bondmap_unbalanced)
        
        self.bondhealth_unbalanced=Epetra.Vector(self.bondmap_unbalanced)
        self.bondhealth_balanced=Epetra.Vector(self.bondmap_balanced)
        self.my_bond_health=np.ones_like(my_bondLengths)
        
        num_pairs_local=len(my_P_bondLengths)
        all_num_pairs=self.__comm.GatherAll(num_pairs_local)
        my_first_global_index=np.add.accumulate(all_num_pairs)
        total_num_pairs=np.sum(all_num_pairs)
        
        if self.rank==0:
            unbalanced_num=total_num_pairs
            balanced_GEs=np.arange(my_first_global_index[self.rank],dtype=np.int32)
        else:
            unbalanced_num=0
            balanced_GEs=np.arange(my_first_global_index[self.rank-1],my_first_global_index[self.rank],dtype=np.int32)
        
        self.pairmap_unbalanced=Epetra.Map(total_num_pairs,unbalanced_num,0,self.__comm)
        self.pairmap_balanced=Epetra.Map(total_num_pairs,balanced_GEs,0,self.__comm)
        
        self.pairimporter = Epetra.Import(self.pairmap_balanced,self.pairmap_unbalanced)
        self.pairexporter = Epetra.Export(self.pairmap_balanced,self.pairmap_unbalanced)
        
        self.pairhealth_unbalanced=Epetra.Vector(self.pairmap_unbalanced)
        self.pairhealth_balanced=Epetra.Vector(self.pairmap_balanced)
        self.my_pair_health=np.ones_like(my_P_bondLengths)
        
        self.my_pair_health = np.ones_like(my_P_bondLengths,dtype=np.float_)
        self.my_num_pairs = np.bincount(self.my_bondpair_x_local,
            minlength = len(self.my_families))
        
        with np.errstate(invalid='ignore'):
            self.my_node_health = np.divide(
                np.bincount(self.my_bondpair_x_local,weights=self.my_pair_health,
                minlength = len(self.my_families)),self.my_num_pairs)
        
        self.my_bondpair_plasticAngle = np.zeros([xpqlength])

        self.my_pair_p_vectors=np.zeros((xpqlength,3),dtype=np.float_)
        self.my_pair_q_vectors=np.zeros((xpqlength,3),dtype=np.float_)
        self.my_pair_p_lengths=np.zeros((xpqlength),dtype=np.float_)
        self.my_pair_q_lengths=np.zeros((xpqlength),dtype=np.float_)
        self.my_iso_bending=np.zeros((self.my_num_owned,3),dtype=np.float_)
        self.my_bond_vectors=np.zeros((xplength,3),dtype=np.float_)
        self.my_bond_lengths=np.zeros((xplength),dtype=np.float_)
        self.my_bond_energies=np.zeros((xplength),dtype=np.float_)
        self.my_bond_stretches=np.zeros((xplength),dtype=np.float_)
        self.my_iso_extension=np.zeros((self.my_num_owned),dtype=np.float_)
        self.my_HpqEff=np.zeros((xpqlength,3),dtype=np.float_)
        self.my_f_tmp = np.zeros((self.numWorkers,3),dtype=np.float_)
        self.my_ref_position = np.transpose(
            [self.my_x_worker[:],self.my_y_worker[:],self.my_z_worker[:]])

        self.num_broken=0

        #The new load balanced map
        balanced_map = self.get_balanced_map()


        #Create distributed vectors (owned only)
        self.my_force_x = Epetra.Vector(balanced_map)
        self.my_force_y = Epetra.Vector(balanced_map)
        self.my_force_z = Epetra.Vector(balanced_map)
        self.my_u = Epetra.Vector(self.balanced_dof_map)

        #Create distributed worker vectors (owned + ghosts)
        self.my_force_x_worker = Epetra.Vector(self.my_worker_map)
        self.my_fx_worker = self.my_force_x_worker.ExtractView()
        self.my_force_y_worker = Epetra.Vector(self.my_worker_map)
        self.my_fy_worker = self.my_force_y_worker.ExtractView()
        self.my_force_z_worker = Epetra.Vector(self.my_worker_map)
        self.my_fz_worker = self.my_force_z_worker.ExtractView()
        self.my_u_worker = Epetra.Vector(self.dof_worker_map,True)
        self.uReshape=self.my_u_worker.ExtractView()
        self.uReshape.shape=(-1,3)
        self.my_u_worker.shape=(-1,3)
        self.xReshape=Epetra.Vector(self.dof_worker_map,self.my_ref_position)

        #Create vectors for temporary use
        self.ux_tmp = Epetra.Vector(self.balanced_map)
        self.uy_tmp = Epetra.Vector(self.balanced_map)
        self.uz_tmp = Epetra.Vector(self.balanced_map)

        self.counter=0
        return

    def step_BCs(self,step,numsteps):
        """Assigns nodes to various BCs, sets force BC values"""
        if step==0:
            if self.rank==0: print "setup BCs"
            
            fraction = np.float(0.0)
            self.my_fixedBC = np.transpose([self.my_x,self.my_y,self.my_z])
            self.undeformed = np.transpose([self.my_x,self.my_y,self.my_z])
            self.my_fixedNodes_local = []
            self.my_pinnedNodes_local = []
            self.my_pinnedNodes2_local = []
            self.my_forceNodes_local = []
            self.my_forceNodes2_local = []
            self.my_forceNodes3_local = []
            self.loadnodes = []
            local_derate_nodes = []
            self.breakablepairs=np.ones_like(self.my_bondpair_x_local)
            self.breakablebonds=np.ones_like(self.my_bond_x_local)
            self.my_fixed_X_local = np.array([],dtype=np.int8)
            self.my_fixed_Y_local = np.array([],dtype=np.int8)
            self.my_fixed_Z_local = np.array([],dtype=np.int8)
            self.load_area = self.plate_width*self.plate_length
            fixedthickness = 3.0*self.grid_spacing/2.0
            forcethickness = 11.0*self.grid_spacing/2.0
            pinthickness = self.grid_spacing*0.55
            #Beam BCs
            #-------------------------------------------------------------
            if False: #Symmetry Fixed-free uniform-loaded cantilever validation
                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]
                    if (xpos < -fixedthickness):
                        self.my_symNodes_local=np.append(self.my_symNodes_local,lid)
                    elif (fixedthickness < xpos and self.plate_length > xpos):
                        self.my_forceNodes_local=np.append(self.my_forceNodes_local,lid)
                    elif (-fixedthickness < xpos and xpos < fixedthickness):
                        self.my_fixedNodes_local=np.append(self.my_fixedNodes_local,lid)

                self.my_symsources_local = np.empty_like(self.my_symsources[:,0])
                for index,gSource in enumerate(self.my_symsources[:,0]):
                    self.my_symsources_local[index] = self.balanced_map.LID(gSource)

                my_num_forcenodes = len(self.my_forceNodes_local)
                self.total_forcenodes=self.__comm.SumAll(my_num_forcenodes)
            #-------------------------------------------------------------

            if False: #Symmetry Fixed-free end-loaded cantilever validation
                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]
                    if (xpos < -fixedthickness):
                        self.my_symNodes_local=np.append(self.my_symNodes_local,lid)
                    elif (np.absolute(xpos-self.plate_length)< forcethickness):
                        self.my_forceNodes_local=np.append(self.my_forceNodes_local,lid)
                    elif (np.absolute(xpos)< fixedthickness):
                        self.my_fixedNodes_local=np.append(self.my_fixedNodes_local,lid)

                self.my_symsources_local = np.empty_like(self.my_symsources[:,0])
                for index,gSource in enumerate(self.my_symsources[:,0]):
                    self.my_symsources_local[index] = self.balanced_map.LID(gSource)

                my_num_forcenodes = len(self.my_forceNodes_local)
                self.total_forcenodes=self.__comm.SumAll(my_num_forcenodes)

            #-------------------------------------------------------------

            if False: #Symmetry Fixed-free end-displacement cantilever validation
                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]
                    if (xpos < -fixedthickness):
                        self.my_symNodes_local=np.append(self.my_symNodes_local,lid)
                    elif (np.absolute(xpos-self.plate_length)< pinthickness):
                        self.my_pinnedNodes_local=np.append(self.my_pinnedNodes_local,lid)
                    elif (-fixedthickness < xpos < fixedthickness):
                        self.my_fixedNodes_local=np.append(self.my_fixedNodes_local,lid)

                self.my_symsources_local = np.empty_like(self.my_symsources[:,0])
                for index,gSource in enumerate(self.my_symsources[:,0]):
                    self.my_symsources_local[index] = self.balanced_map.LID(gSource)

                my_num_dispnodes = len(self.my_pinnedNodes_local)
                self.total_num_dispnodes=self.__comm.SumAll(my_num_dispnodes)

             #-------------------------------------------------------------

            if False: #Pinned-Pinned with point load
                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]
                    if (np.absolute(xpos)< pinthickness):
                        self.my_fixedNodes_local=np.append(self.my_fixedNodes_local,lid)
                    elif (np.absolute(xpos-self.plate_length)< pinthickness):
                        self.my_fixedNodes_local=np.append(self.my_fixedNodes_local,lid)
                    elif (np.absolute(xpos-(loadlocation*self.plate_length))< forcethickness):
                        self.my_forceNodes_local=np.append(self.my_forceNodes_local,lid)

                my_num_dispnodes = len(self.my_pinnedNodes_local)
                self.total_num_dispnodes=self.__comm.SumAll(my_num_dispnodes)

                my_num_dispnodes2 = len(self.my_pinnedNodes2_local)
                self.total_num_dispnodes2=self.__comm.SumAll(my_num_dispnodes2)

                my_num_forcenodes = len(self.my_forceNodes_local)
                self.total_forcenodes=self.__comm.SumAll(my_num_forcenodes)

            #-------------------------------------------------------------

            if False: #Pinned-Pinned with point displacement
                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]
                    if (np.absolute(xpos)< pinthickness):
                        self.my_fixedNodes_local=np.append(self.my_fixedNodes_local,lid)
                    elif (np.absolute(xpos-self.plate_length)< pinthickness):
                        self.my_fixedNodes_local=np.append(self.my_fixedNodes_local,lid)
                    elif (np.absolute(xpos-(loadlocation*self.plate_length))< pinthickness):
                        self.my_pinnedNodes_local=np.append(self.my_pinnedNodes_local,lid)

                my_num_dispnodes = len(self.my_pinnedNodes_local)
                self.total_num_dispnodes=self.__comm.SumAll(my_num_dispnodes)

                my_num_dispnodes2 = len(self.my_pinnedNodes2_local)
                self.total_num_dispnodes2=self.__comm.SumAll(my_num_dispnodes2)

                my_num_forcenodes = len(self.my_forceNodes_local)
                self.total_forcenodes=self.__comm.SumAll(my_num_forcenodes)

            #-------------------------------------------------------------

            if False:#Pinned-Pinned with uniform load
                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]
                    if (np.absolute(xpos)< pinthickness):
                        self.my_fixedNodes_local=np.append(self.my_fixedNodes_local,lid)
                    elif (np.absolute(xpos-self.plate_length)< pinthickness):
                        self.my_fixedNodes_local=np.append(self.my_fixedNodes_local,lid)
                    elif (0.0<xpos<self.plate_length):
                        self.my_forceNodes_local=np.append(self.my_forceNodes_local,lid)
                my_num_dispnodes = len(self.my_pinnedNodes_local)
                self.total_num_dispnodes=self.__comm.SumAll(my_num_dispnodes)

                my_num_dispnodes2 = len(self.my_pinnedNodes2_local)
                self.total_num_dispnodes2=self.__comm.SumAll(my_num_dispnodes2)

                my_num_forcenodes = len(self.my_forceNodes_local)
                self.total_forcenodes=self.__comm.SumAll(my_num_forcenodes)

            #-------------------------------------------------------------

            if False: #Displacement Double Torsion Plate
                border = 0.1
                ycrack = self.plate_width/2.0

                supportnodes = []
                loadnodes = []
                distantnodes = []

                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]

                    if (-pinthickness<border-xpos<pinthickness):
                        if (-pinthickness<border-ypos<pinthickness):
                            supportnodes=np.append(supportnodes,lid)
                        elif (-pinthickness<self.plate_width-border-ypos<pinthickness):
                            supportnodes=np.append(supportnodes,lid)
                        elif (-pinthickness<ycrack+border-ypos<pinthickness):
                            loadnodes=np.append(loadnodes,lid)
                        elif (-pinthickness<ycrack-border-ypos<pinthickness):
                            loadnodes=np.append(loadnodes,lid)

                    if (-pinthickness<self.plate_length-border-xpos<pinthickness):
                        if (-pinthickness<border-ypos<pinthickness):
                            distantnodes=np.append(distantnodes,lid)
                        elif (-pinthickness<self.plate_width-border-ypos<pinthickness):
                            distantnodes=np.append(distantnodes,lid)

                self.my_fixed_X_local = distantnodes
                self.my_fixed_Y_local = distantnodes

                self.my_fixed_Z_local = np.concatenate(
                    (supportnodes,loadnodes,distantnodes))

                self.loadnodes=loadnodes
                self.my_forceNodes_local = []
                self.my_forceNodes2_local = []

                #Make pairs centered on BC points unbreakable
                self.breakablebonds=np.ones_like(self.my_bondpair_x_local)
                self.breakablebonds=np.where(
                    np.in1d(self.my_bondpair_x_local,self.my_fixed_Z_local),0.0,1.0)


                my_num_dispnodes = len(self.my_pinnedNodes_local)
                self.total_num_dispnodes=self.__comm.SumAll(my_num_dispnodes)

                my_num_dispnodes2 = len(self.my_pinnedNodes2_local)
                self.total_num_dispnodes2=self.__comm.SumAll(my_num_dispnodes2)

                my_num_forcenodes = len(self.my_forceNodes_local)
                self.total_forcenodes=self.__comm.SumAll(my_num_forcenodes)
                my_num_forcenodes2 = len(self.my_forceNodes2_local)
                self.total_forcenodes2=self.__comm.SumAll(my_num_forcenodes2)

            #-------------------------------------------------------------

            if False: #Circular edge-supported with uniform load
                edge = []
                interior = []
                minx = []
                maxx = []

                for lid,position in enumerate(self.undeformed):
                    radius = np.sqrt(position[0]**2 + position[1]**2)

                    if (-pinthickness<radius-(self.plate_width/2)<pinthickness):
                        edge=np.append(edge,lid)
                    if (radius<(self.plate_width/2)-pinthickness):
                        interior=np.append(interior,lid)

                if self.rank==0:
                    minx = np.amin(self.undeformed[:,0])
                    self.my_fixed_X_local = np.append(self.my_fixed_X_local,minx)
                    self.my_fixed_Y_local = np.append(self.my_fixed_Y_local,minx)

                if self.rank==(self.size-1):
                    maxx = np.amax(self.undeformed[:,0])
                    self.my_fixed_Y_local = np.append(self.my_fixed_Y_local,maxx)

                numedge=self.__comm.SumAll(len(edge))
                numInterior=self.__comm.SumAll(len(interior))
                if self.rank==0:
                    print numedge," edge nodes"
                    print numInterior," interior nodes"

                self.my_fixed_Z_local = edge
                self.my_forceNodes_local = interior


                my_num_dispnodes = len(self.my_pinnedNodes_local)
                self.total_num_dispnodes=self.__comm.SumAll(my_num_dispnodes)

                my_num_dispnodes2 = len(self.my_pinnedNodes2_local)
                self.total_num_dispnodes2=self.__comm.SumAll(my_num_dispnodes2)

                my_num_forcenodes = len(self.my_forceNodes_local)
                self.total_forcenodes=self.__comm.SumAll(my_num_forcenodes)
                self.my_load_area = (self.my_area[np.asarray(interior,dtype=int)]).sum()
                self.total_load_area = self.__comm.SumAll(self.my_load_area)

                my_num_forcenodes2 = len(self.my_forceNodes2_local)
                self.total_forcenodes2=self.__comm.SumAll(my_num_forcenodes2)

            #-------------------------------------------------------------

            if False: #Half-circle clip or full ring
                x0edge = []
                xLedge = []
                y0edge = []
                yLedge = []
                interior = []

                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]
                    zpos = position[2]

                    if (-pinthickness<xpos<pinthickness and
                        -pinthickness<zpos<pinthickness):
                        x0edge=np.append(x0edge,lid)
                    if (-pinthickness<xpos-self.plate_length<pinthickness and
                        -pinthickness<zpos<pinthickness):
                        xLedge=np.append(xLedge,lid)
                    if (-pinthickness<ypos<pinthickness):
                        y0edge=np.append(y0edge,lid)

                numx0=self.__comm.SumAll(len(x0edge))
                numy0=self.__comm.SumAll(len(y0edge))
                numxL=self.__comm.SumAll(len(xLedge))
                if self.rank==0:
                    print numx0," x0 nodes"
                    print numy0," y0 nodes"
                    print numxL," xL nodes"
                self.my_fixed_X_local = np.asarray(x0edge,dtype=np.intc)
                self.my_fixed_Y_local = np.asarray(y0edge,dtype=np.intc)
                self.my_fixed_Z_local = np.asarray(
                    np.unique(np.concatenate((x0edge,xLedge))),dtype=np.intc)
                self.my_forceNodes_local = np.asarray(xLedge,dtype=np.intc)

                my_num_forcenodes = len(xLedge)
                self.total_forcenodes=self.__comm.SumAll(my_num_forcenodes)
                self.my_load_area = (self.my_area[np.asarray(xLedge,dtype=int)]).sum()
                self.total_load_area = self.__comm.SumAll(self.my_load_area)

            #-------------------------------------------------------------

            if False: #Ring centered at x=0
                x0edge = []
                xLedge = []
                y0edge = []
                yLedge = []
                interior = []

                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]
                    zpos = position[2]

                    if (-pinthickness<xpos<pinthickness):
                        interior=np.append(interior,lid)
                    if (-pinthickness<xpos+self.plate_length/2<pinthickness and
                        -pinthickness<zpos<pinthickness):
                        x0edge=np.append(x0edge,lid)
                    if (-pinthickness<xpos-self.plate_length/2<pinthickness and
                        -pinthickness<zpos<pinthickness):
                        xLedge=np.append(xLedge,lid)
                    if (-pinthickness<ypos<pinthickness):
                        y0edge=np.append(y0edge,lid)

                numx0=self.__comm.SumAll(len(x0edge))
                numy0=self.__comm.SumAll(len(y0edge))
                numxL=self.__comm.SumAll(len(xLedge))
                if self.rank==0:
                    print numx0," x0 nodes"
                    print numy0," y0 nodes"
                    print numxL," xL nodes"
                self.my_fixed_X_local = np.asarray(interior,dtype=np.intc)
                self.my_fixed_Y_local = np.asarray(y0edge,dtype=np.intc)
                self.my_fixed_Z_local = np.asarray(
                    np.unique(np.concatenate((x0edge,xLedge))),dtype=np.intc)
                
                self.my_forceNodes_local = np.asarray(xLedge,dtype=np.intc)
                my_num_forcenodes = len(xLedge)
                self.total_forcenodes=self.__comm.SumAll(my_num_forcenodes)
                self.my_load_area = (self.my_area[np.asarray(xLedge,dtype=int)]).sum()
                self.total_load_area = self.__comm.SumAll(self.my_load_area)
                
                self.my_forceNodes2_local = np.asarray(x0edge,dtype=np.intc)
                my_num_forcenodes2 = len(x0edge)
                self.total_forcenodes2=self.__comm.SumAll(my_num_forcenodes2)

            #-------------------------------------------------------------

            if True: #SSSS uniformly loaded plate
                x0edge = []
                xLedge = []
                y0edge = []
                yLedge = []
                interior = []

                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]

                    if (-pinthickness<xpos<pinthickness and
                        -pinthickness<ypos<(self.plate_width+pinthickness)):
                        x0edge=np.append(x0edge,lid)
                    if (-pinthickness<(xpos-self.plate_length)<pinthickness and
                        -pinthickness<ypos<(self.plate_width+pinthickness)):
                        xLedge=np.append(xLedge,lid)
                    if (-pinthickness<xpos<(self.plate_length+pinthickness) and
                        -pinthickness<ypos<pinthickness):
                        y0edge=np.append(y0edge,lid)
                    if (-pinthickness<(xpos)<(self.plate_width+pinthickness) and
                        -pinthickness<(ypos-self.plate_width)<pinthickness):
                        yLedge=np.append(yLedge,lid)
                    if (pinthickness<(xpos)<(self.plate_width-pinthickness) and
                        pinthickness<(ypos)<(self.plate_width-pinthickness)):
                        interior=np.append(interior,lid)


                numx0=self.__comm.SumAll(len(x0edge))
                numy0=self.__comm.SumAll(len(y0edge))
                numxL=self.__comm.SumAll(len(xLedge))
                numyL=self.__comm.SumAll(len(yLedge))
                numInt=self.__comm.SumAll(len(interior))
                if self.rank==0:
                    print numx0," x0 nodes"
                    print numy0," y0 nodes"
                    print numxL," xL nodes"
                    print numyL," yL nodes"
                    print numInt," interior nodes"

                self.my_fixed_X_local = np.asarray(x0edge,dtype=np.intc)
                self.my_fixed_Y_local = np.asarray(y0edge,dtype=np.intc)
                self.my_fixed_Z_local = np.asarray(np.unique(
                    np.concatenate((x0edge,y0edge,yLedge,xLedge))),dtype=np.intc)
                self.my_forceNodes_local = np.asarray(interior,dtype=np.intc)
                self.my_forceNodes2_local = []


                my_num_forcenodes = len(self.my_forceNodes_local)
                self.total_forcenodes=self.__comm.SumAll(my_num_forcenodes)
                self.my_load_area = (self.my_area[np.asarray(interior,dtype=int)]).sum()
                self.total_load_area = self.__comm.SumAll(self.my_load_area)
            #-------------------------------------------------------------

            if False: # 4-point force bend plate
                x0edge = []
                xLedge = []
                y0edge = []
                yLedge = []
                interior = []

                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]

                    if (-pinthickness<xpos<pinthickness and
                        -pinthickness<ypos<(self.plate_width+pinthickness)):
                        x0edge=np.append(x0edge,lid)
                    if (-pinthickness<(xpos-self.plate_length)<pinthickness and
                        -pinthickness<ypos<(self.plate_width+pinthickness)):
                        xLedge=np.append(xLedge,lid)
                    if (-pinthickness<xpos<pinthickness and
                        -pinthickness<ypos<pinthickness):
                        y0edge=np.append(y0edge,lid)
                    if (-pinthickness<(xpos-self.plate_length)<pinthickness and
                        -pinthickness<ypos<pinthickness):
                        y0edge=np.append(y0edge,lid)
                    if (-self.horizon/2.0<(xpos-0.25*self.plate_length)<(self.horizon/2.0) and
                        -pinthickness<(ypos)<(self.plate_width+pinthickness)):
                        interior=np.append(interior,lid)
                    if (-self.horizon/2.0<(xpos-0.75*self.plate_length)<(self.horizon/2.0) and
                        -pinthickness<(ypos)<(self.plate_width+pinthickness)):
                        interior=np.append(interior,lid)


                numx0=self.__comm.SumAll(len(x0edge))
                numy0=self.__comm.SumAll(len(y0edge))
                numxL=self.__comm.SumAll(len(xLedge))
                numyL=self.__comm.SumAll(len(yLedge))
                numInt=self.__comm.SumAll(len(interior))
                if self.rank==0:
                    print numx0," x0 nodes"
                    print numy0," y0 nodes"
                    print numxL," xL nodes"
                    print numyL," yL nodes"
                    print numInt," interior nodes"

                self.my_fixed_X_local = np.asarray(x0edge,dtype=np.intc)
                self.my_fixed_Y_local = np.asarray(y0edge,dtype=np.intc)
                self.my_fixed_Z_local = np.asarray(np.unique(
                    np.concatenate((x0edge,y0edge,yLedge,xLedge))),dtype=np.intc)
                self.my_forceNodes_local = np.asarray(interior,dtype=np.intc)
                self.my_forceNodes2_local = np.asarray(interior,dtype=np.intc)


                my_num_forcenodes = len(self.my_forceNodes_local)
                self.total_forcenodes=self.__comm.SumAll(my_num_forcenodes)
                self.my_load_area = (self.my_area[np.asarray(interior,dtype=int)]).sum()
                self.total_load_area = self.__comm.SumAll(self.my_load_area)

                my_num_forcenodes2 = len(self.my_forceNodes2_local)
                self.total_forcenodes2=self.__comm.SumAll(my_num_forcenodes2)

            #-------------------------------------------------------------
            if False: # Double Torsion plate
                x0edge = []
                xLedge = []
                y0edge = []
                yLedge = []
                interior = []
                
                border = 0.1
                ycrack = self.plate_width/2.0


                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]

                    if (-pinthickness<xpos-border<pinthickness and
                        -pinthickness<ypos-border<pinthickness):
                        x0edge=np.append(x0edge,lid)
                    if (-pinthickness<xpos-border<pinthickness and
                        -pinthickness<ypos+border-self.plate_width<pinthickness):
                        x0edge=np.append(x0edge,lid)
                    if (-pinthickness<xpos+border-self.plate_length<pinthickness and
                        -pinthickness<ypos-border<pinthickness):
                        xLedge=np.append(xLedge,lid)
                    if (-pinthickness<xpos+border-self.plate_length<pinthickness and
                        -pinthickness<ypos+border-self.plate_width<pinthickness):
                        xLedge=np.append(xLedge,lid)
                    if (-pinthickness<xpos-border<pinthickness and
                        -pinthickness<ypos-border<pinthickness):
                        y0edge=np.append(y0edge,lid)
                    if (-pinthickness<xpos+border-self.plate_length<pinthickness and
                        -pinthickness<ypos-border<pinthickness):
                        y0edge=np.append(y0edge,lid)
                    if (-pinthickness<(xpos-border)<(pinthickness) and
                        -pinthickness<ypos-border-ycrack<pinthickness):
                        interior=np.append(interior,lid)
                    if (-pinthickness<(xpos-border)<(pinthickness) and
                        -pinthickness<ypos+border-ycrack<pinthickness):
                        interior=np.append(interior,lid)


                numx0=self.__comm.SumAll(len(x0edge))
                numy0=self.__comm.SumAll(len(y0edge))
                numxL=self.__comm.SumAll(len(xLedge))
                numyL=self.__comm.SumAll(len(yLedge))
                numInt=self.__comm.SumAll(len(interior))
                if self.rank==0:
                    print numx0," x0 nodes"
                    print numy0," y0 nodes"
                    print numxL," xL nodes"
                    print numyL," yL nodes"
                    print numInt," interior nodes"

                self.my_fixed_X_local = np.asarray(x0edge,dtype=np.intc)
                self.my_fixed_Y_local = np.asarray(y0edge,dtype=np.intc)
                self.my_fixed_Z_local = np.asarray(x0edge,dtype=np.intc)
                self.loadnodes = np.asarray(interior,dtype=np.intc)
                self.load_dof = 3*self.loadnodes+2
                self.load_velocity=0.0
                self.load_duration=0.0
                if True:
                    if self.rank==0:print "Make load nodes unbreakable"
                    unbreakablenodes = np.asarray(np.unique(
                        np.concatenate((x0edge,y0edge,yLedge,xLedge,interior))),dtype=np.intc)
                    pairunbreakable = np.in1d(self.my_bondpair_x_local,unbreakablenodes)
                    self.pair_critical_energy[pairunbreakable]*=100
                
            #-------------------------------------------------------------
            if False: # 4-point displacement bend plate
                x0edge = []
                xLedge = []
                y0edge = []
                yLedge = []
                interior = []

                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]

                    if (-pinthickness<xpos<pinthickness and
                        -pinthickness<ypos<(self.plate_width+pinthickness)):
                        x0edge=np.append(x0edge,lid)
                    if (-pinthickness<(xpos-self.plate_length)<pinthickness and
                        -pinthickness<ypos<(self.plate_width+pinthickness)):
                        xLedge=np.append(xLedge,lid)
                    if (-pinthickness<xpos<pinthickness and
                        -pinthickness<ypos<pinthickness):
                        y0edge=np.append(y0edge,lid)
                    if (-pinthickness<(xpos-self.plate_length)<pinthickness and
                        -pinthickness<ypos<pinthickness):
                        y0edge=np.append(y0edge,lid)
                    if (-pinthickness<(xpos-0.25*self.plate_length)<(pinthickness) and
                        -pinthickness<(ypos)<(self.plate_width+pinthickness)):
                        interior=np.append(interior,lid)
                    if (-pinthickness<(xpos-0.75*self.plate_length)<(pinthickness) and
                        -pinthickness<(ypos)<(self.plate_width+pinthickness)):
                        interior=np.append(interior,lid)


                numx0=self.__comm.SumAll(len(x0edge))
                numy0=self.__comm.SumAll(len(y0edge))
                numxL=self.__comm.SumAll(len(xLedge))
                numyL=self.__comm.SumAll(len(yLedge))
                numInt=self.__comm.SumAll(len(interior))
                if self.rank==0:
                    print numx0," x0 nodes"
                    print numy0," y0 nodes"
                    print numxL," xL nodes"
                    print numyL," yL nodes"
                    print numInt," interior nodes"

                self.my_fixed_X_local = np.asarray(x0edge,dtype=np.intc)
                self.my_fixed_Y_local = np.asarray(y0edge,dtype=np.intc)
                self.my_fixed_Z_local = np.asarray(np.unique(
                    np.concatenate((x0edge,y0edge,yLedge,xLedge))),dtype=np.intc)
                self.loadnodes = np.asarray(interior,dtype=np.intc)
                self.load_dof = 3*self.loadnodes+2
                self.load_velocity=0.0
                self.load_duration=0.0
                if True:
                    if self.rank==0:print "Make load nodes unbreakable"
                    pairunbreakable = np.in1d(self.my_bondpair_x_local,self.loadnodes)
                    self.pair_critical_energy[pairunbreakable]*=100

            #-------------------------------------------------------------
            if False: # Plate wave test
                x0edge = []
                xLedge = []
                y0edge = []
                yLedge = []
                interior = []

                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]

                    if (xpos<2.0*pinthickness and
                        -pinthickness<ypos<(self.plate_width+pinthickness)):
                        x0edge=np.append(x0edge,lid)
                    if (-2.0*pinthickness<(xpos-self.plate_length) and
                        -pinthickness<ypos<(self.plate_width+pinthickness)):
                        xLedge=np.append(xLedge,lid)
                    if (-pinthickness<xpos<pinthickness and
                        -pinthickness<ypos<pinthickness):
                        y0edge=np.append(y0edge,lid)
                    if (-pinthickness<(xpos-self.plate_length)<pinthickness and
                        -pinthickness<ypos<pinthickness):
                        y0edge=np.append(y0edge,lid)


                numx0=self.__comm.SumAll(len(x0edge))
                numy0=self.__comm.SumAll(len(y0edge))
                numxL=self.__comm.SumAll(len(xLedge))
                numyL=self.__comm.SumAll(len(yLedge))
                numInt=self.__comm.SumAll(len(interior))
                if self.rank==0:
                    print numx0," x0 nodes"
                    print numy0," y0 nodes"
                    print numxL," xL nodes"
                    print numyL," yL nodes"
                    print numInt," interior nodes"

                self.my_fixed_X_local = np.asarray(np.unique(
                    np.concatenate((x0edge,xLedge))),dtype=np.intc)
                self.my_fixed_Y_local = np.asarray(y0edge,dtype=np.intc)
                self.loadnodes = np.asarray(np.unique(
                    np.concatenate((x0edge,xLedge))),dtype=np.intc)
                self.load_dof = 3*self.loadnodes+2
                self.load_velocity=0.0
                self.load_duration=0.0
                self.load_amplitude=0.0

            #-------------------------------------------------------------
            if False: # one-sided wave
                x0edge = []
                xLedge = []
                y0edge = []
                yLedge = []
                interior = []

                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]

                    if (xpos<2.0*pinthickness and
                        -pinthickness<ypos<(self.plate_width+pinthickness)):
                        x0edge=np.append(x0edge,lid)
                    if (-2.0*pinthickness<(xpos-self.plate_length) and
                        -pinthickness<ypos<(self.plate_width+pinthickness)):
                        xLedge=np.append(xLedge,lid)
                    if (-pinthickness<xpos<pinthickness and
                        -pinthickness<ypos<pinthickness):
                        y0edge=np.append(y0edge,lid)
                    if (-pinthickness<(xpos-self.plate_length)<pinthickness and
                        -pinthickness<ypos<pinthickness):
                        y0edge=np.append(y0edge,lid)


                numx0=self.__comm.SumAll(len(x0edge))
                numy0=self.__comm.SumAll(len(y0edge))
                numxL=self.__comm.SumAll(len(xLedge))
                numyL=self.__comm.SumAll(len(yLedge))
                numInt=self.__comm.SumAll(len(interior))
                if self.rank==0:
                    print numx0," x0 nodes"
                    print numy0," y0 nodes"
                    print numxL," xL nodes"
                    print numyL," yL nodes"
                    print numInt," interior nodes"

                self.my_fixed_X_local = np.asarray(np.unique(
                    np.concatenate((x0edge,xLedge))),dtype=np.intc)
                self.my_fixed_Y_local = np.asarray(y0edge,dtype=np.intc)
                self.my_fixed_Z_local = np.asarray(x0edge,dtype=np.intc)
                self.loadnodes = np.asarray(xLedge,dtype=np.intc)
                self.load_dof = 3*self.loadnodes+2
                self.load_velocity=0.0
                self.load_duration=0.0
                self.load_amplitude=0.0

            #-------------------------------------------------------------
            if False: # SS uniform load
                x0edge = []
                xLedge = []
                y0edge = []
                yLedge = []
                interior = []

                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]

                    if (-pinthickness<xpos<pinthickness and
                        -pinthickness<ypos<(self.plate_width+pinthickness)):
                        x0edge=np.append(x0edge,lid)
                    if (-pinthickness<(xpos-self.plate_length)<pinthickness and
                        -pinthickness<ypos<(self.plate_width+pinthickness)):
                        xLedge=np.append(xLedge,lid)
                    if (-pinthickness<xpos<pinthickness and
                        -pinthickness<ypos<pinthickness):
                        y0edge=np.append(y0edge,lid)
                    if (-pinthickness<(xpos-self.plate_length)<pinthickness and
                        -pinthickness<ypos<pinthickness):
                        y0edge=np.append(y0edge,lid)
                    if (pinthickness<(xpos)<(self.plate_width-pinthickness) and
                        -pinthickness<(ypos)<(self.plate_width+pinthickness)):
                        interior=np.append(interior,lid)


                numx0=self.__comm.SumAll(len(x0edge))
                numy0=self.__comm.SumAll(len(y0edge))
                numxL=self.__comm.SumAll(len(xLedge))
                numyL=self.__comm.SumAll(len(yLedge))
                numInt=self.__comm.SumAll(len(interior))
                if self.rank==0:
                    print numx0," x0 nodes"
                    print numy0," y0 nodes"
                    print numxL," xL nodes"
                    print numyL," yL nodes"
                    print numInt," interior nodes"

                self.my_fixed_X_local = np.asarray(x0edge,dtype=np.intc)
                self.my_fixed_Y_local = np.asarray(y0edge,dtype=np.intc)
                self.my_fixed_Z_local = np.asarray(np.unique(
                    np.concatenate((x0edge,y0edge,yLedge,xLedge))),dtype=np.intc)
                self.my_forceNodes_local = np.asarray(interior,dtype=np.intc)
                self.my_forceNodes2_local = np.asarray(interior,dtype=np.intc)


                my_num_forcenodes = len(self.my_forceNodes_local)
                self.total_forcenodes=self.__comm.SumAll(my_num_forcenodes)
                self.my_load_area = (self.my_area[np.asarray(interior,dtype=int)]).sum()
                self.total_load_area = self.__comm.SumAll(self.my_load_area)

                my_num_forcenodes2 = len(self.my_forceNodes2_local)
                self.total_forcenodes2=self.__comm.SumAll(my_num_forcenodes2)

            #-------------------------------------------------------------
            if False: #Cantilever plate
                x0edge = []
                xLedge = []
                y0edge = []
                yLedge = []
                interior = []

                for lid,position in enumerate(self.undeformed):
                    xpos = position[0]
                    ypos = position[1]
                    if (xpos<self.horizon):
                        x0edge=np.append(x0edge,lid)
                    if (-pinthickness<(ypos)<pinthickness):
                        y0edge=np.append(xLedge,lid)
                    if (-self.horizon<(xpos-self.plate_length)<self.horizon):
                        xLedge=np.append(xLedge,lid)

                numx0=self.__comm.SumAll(len(x0edge))
                numy0=self.__comm.SumAll(len(y0edge))
                numxL=self.__comm.SumAll(len(xLedge))
                numyL=self.__comm.SumAll(len(yLedge))
                numInt=self.__comm.SumAll(len(interior))
                if self.rank==0:
                    print numx0," x0 nodes"
                    print numy0," y0 nodes"
                    print numxL," xL nodes"
                    print numyL," yL nodes"
                    print numInt," interior nodes"

                self.my_fixed_X_local = np.asarray(x0edge,dtype=np.intc)
                self.my_fixed_Y_local = np.asarray(y0edge,dtype=np.intc)
                self.my_fixed_Z_local = np.asarray(x0edge,dtype=np.intc)
                self.my_forceNodes_local = np.asarray(xLedge,dtype=np.intc)
                self.my_forceNodes2_local = np.asarray(xLedge,dtype=np.intc)


                my_num_forcenodes = len(self.my_forceNodes_local)
                self.total_forcenodes=self.__comm.SumAll(my_num_forcenodes)
                self.my_load_area = (self.my_area[np.asarray(xLedge,dtype=int)]).sum()
                self.total_load_area = self.__comm.SumAll(self.my_load_area)

                my_num_forcenodes2 = len(self.my_forceNodes2_local)
                self.total_forcenodes2=self.__comm.SumAll(my_num_forcenodes2)

            #-------------------------------------------------------------
            self.my_x_forceBCs = Epetra.Vector(self.balanced_map)
            self.my_y_forceBCs = Epetra.Vector(self.balanced_map)
            self.my_z_forceBCs = Epetra.Vector(self.balanced_map)
        
        if step != 0:
            fraction = np.float(step)/np.float(numsteps)
            
            self.my_x_forceBCs.PutScalar(0.0)
            self.my_y_forceBCs.PutScalar(0.0)
            self.my_z_forceBCs.PutScalar(0.0)


            if len(self.my_forceNodes_local)>0:
                pressure = fraction*self.gamma*self.forcenorm/self.total_load_area
                for lid in self.my_forceNodes_local:
                    self.my_z_forceBCs[lid] = (pressure*self.my_area[lid])
            myforcetotal = self.my_z_forceBCs.sum()
            totalZforce = self.__comm.SumAll(myforcetotal)
            if self.rank == 0: print "Total Z force:",totalZforce

            #if len(self.my_forceNodes_local)>0:
                #pressure = fraction*self.gamma*self.forcenorm/self.total_load_area
                #for lid in self.my_forceNodes_local:
                    #self.my_x_forceBCs[lid] = (pressure*self.my_area[lid])
            #if len(self.my_forceNodes2_local)>0:
                #pressure = -fraction*self.gamma*self.forcenorm/self.total_load_area
                #for lid in self.my_forceNodes2_local:
                    #self.my_x_forceBCs[lid] = (pressure*self.my_area[lid])
        return True

    def getGraph(self):
        return self.__graph

    def __init_graph(self):
        """Initializes a graph based on the discretization"""


        self.__num_per_row = np.array(
            [ (row > -1).sum() for row in self.my_connectivity],
            dtype=np.int32)

        self.__num_dof_per_row = np.repeat((3*self.__num_per_row),3,axis=0)


        #Create the distributed graph
        self.__graph = Epetra.CrsGraph(Epetra.Copy,
            self.balanced_dof_map,self.__num_dof_per_row, True)

        #Fill the graph
        gids = self.balanced_dof_map.MyGlobalElements()

        for lid, gid in enumerate(gids):
            nodeID = lid/3
            row1 = self.my_connectivity[nodeID]
            row3 = np.concatenate((3*row1,1+(3*row1),2+(3*row1)))
            insert_indices = np.asarray(np.sort(np.unique(row3[row3 > -1])),dtype='int32')
            self.__graph.InsertGlobalIndices(gid, insert_indices)


        self.__graph.FillComplete()

        return

    def __init_jacobian(self):
        """Initialize Jacobian from graph"""
        self.__jac = Epetra.CrsMatrix(Epetra.Copy, self.__graph)
        return

    def reshapeU(self,u):
        """takes u-vector and distributes positions"""
        self.ux_tmp[:] = u[0::3]
        self.uy_tmp[:] = u[1::3]
        self.uz_tmp[:] = u[2::3]


        self.my_x_worker.Import(self.ux_tmp,self.worker_importer,Epetra.Insert)
        self.my_y_worker.Import(self.uy_tmp,self.worker_importer,Epetra.Insert)
        self.my_z_worker.Import(self.uz_tmp,self.worker_importer,Epetra.Insert)

        if False:
            self.my_x_worker[:]=self.my_x_worker[self.my_SymSource_local]
            self.my_x_worker[self.my_xSym_worker]=-self.my_x_worker[self.my_xSym_worker]
            self.my_y_worker[:]=self.my_y_worker[self.my_SymSource_local]
            self.my_y_worker[self.my_ySym_worker]=-self.my_y_worker[self.my_ySym_worker]
            self.my_z_worker[:]=self.my_z_worker[self.my_SymSource_local]
            self.my_z_worker[self.my_zSym_worker]=-self.my_z_worker[self.my_zSym_worker]


        if False: #Old reshapeU
            self.my_ux_augmented = self.augmentbuilder.dot(self.my_x_worker)
            self.my_uy_augmented = self.augmentbuilder.dot(self.my_y_worker)
            self.my_uz_augmented = self.augmentbuilder.dot(self.my_z_worker)
            uReshape = np.vstack(
                (self.my_ux_augmented,self.my_uy_augmented,self.my_uz_augmented)).T

        uReshape = np.vstack((self.my_x_worker,self.my_y_worker,self.my_z_worker)).T
        return uReshape

    #@profile
    def computeF(self,u,F,flag):
        #if self.rank==0:profiler.enable()
        t_start = time.time()

        try:
            #Initialize force components
            self.my_force_x[:] = 0.0
            self.my_force_y[:] = 0.0
            self.my_force_z[:] = 0.0

            self.my_force_x_worker[:] = 0.0
            self.my_force_y_worker[:] = 0.0
            self.my_force_z_worker[:] = 0.0

            self.my_u_worker.Import(u,self.dof_worker_importer,Epetra.Insert)
            self.my_u_worker.Update(1.0,self.xReshape,1.0)

            calcforce.bondvectors(
                self.my_u_worker,
                self.my_bond_vectors,self.my_bond_lengths,
                self.my_bond_x_local,self.my_bond_p_local)

            calcforce.bondstretches(
                self.my_bond_vectors,self.my_bond_lengths,
                self.my_bond_x_local,
                self.my_ref_bondLengths,
                self.my_bond_stretches,
                self.iso_ext_coefficients,self.my_iso_extension)

            calcforce.pairvectors(
                self.my_u_worker,
                self.my_vpt_offset,
                self.my_pair_p_vectors,self.my_pair_q_vectors,
                self.my_pair_p_lengths,self.my_pair_q_lengths,
                self.my_vsources_A_local,self.my_vsources_B_local,
                self.my_vsources_C_local,
                self.my_vweights_A,self.my_vweights_B,self.my_vweights_C,
                self.my_bondpair_x_local,self.my_bondpair_p_local,
                self.my_bondpair_q_local)

            calcforce.pairbending(
                self.my_HpqEff,
                self.my_pair_p_vectors,self.my_pair_q_vectors,
                self.my_pair_p_lengths,self.my_pair_q_lengths,
                self.my_pair_health,self.my_pair_stiffness,
                self.iso_pair_coefficients,
                self.my_iso_bending,
                self.my_bondpair_x_local)

            if flag == 6:
                calcforce.breakcoupled(
                    self.my_bond_health,self.my_extension_stiffness,
                    self.my_bond_stretches,self.my_bond_energies,
                    self.iso_ext_coefficients,self.my_iso_ext_stiffness,
                    self.my_iso_extension,
                    self.my_bond_x_local,self.my_bond_p_local,
                    self.my_HpqEff,
                    self.my_pair_p_vectors,self.my_pair_q_vectors,
                    self.my_pair_p_lengths,self.my_pair_q_lengths,
                    self.my_pair_health,self.my_pair_stiffness,
                    self.pair_critical_energy,
                    self.bond1,self.bond2,
                    self.my_isoBend_stiffness,
                    self.iso_pair_coefficients,
                    self.my_iso_bending,
                    self.my_bondpair_x_local,
                    self.num_broken)
                self.num_broken=self.__comm.SumAll(self.num_broken)

            calcforce.bondforces(
                self.my_fx_worker,self.my_fy_worker,self.my_fz_worker,
                self.my_bond_vectors,self.my_bond_lengths,
                self.my_bond_health,self.my_extension_stiffness,
                self.my_bond_stretches,
                self.iso_ext_coefficients,self.my_iso_ext_stiffness,
                self.my_iso_extension,
                self.my_bond_x_local,self.my_bond_p_local)

            calcforce.pairforces(
                self.my_fx_worker,self.my_fy_worker,self.my_fz_worker,
                self.my_HpqEff,
                self.my_pair_p_vectors,self.my_pair_q_vectors,
                self.my_pair_p_lengths,self.my_pair_q_lengths,
                self.my_pair_health,self.my_pair_stiffness,
                self.my_vsources_A_local,self.my_vsources_B_local,
                self.my_vsources_C_local,
                self.my_vweights_A,self.my_vweights_B,self.my_vweights_C,
                self.iso_pair_coefficients,
                self.my_isoBend_stiffness,
                self.my_iso_bending,
                self.my_bondpair_x_local,
                self.my_bondpair_p_local,
                self.my_bondpair_q_local)


            self.my_force_x.Export(self.my_force_x_worker, self.worker_exporter,
                Epetra.Add)
            self.my_force_y.Export(self.my_force_y_worker, self.worker_exporter,
                Epetra.Add)
            self.my_force_z.Export(self.my_force_z_worker, self.worker_exporter,
                Epetra.Add)

            self.my_force_x.Update(1.0,self.my_x_forceBCs,1.0)
            self.my_force_y.Update(1.0,self.my_y_forceBCs,1.0)
            self.my_force_z.Update(1.0,self.my_z_forceBCs,1.0)

            if False:
                self.my_force_x[self.my_allSym_owned]=0.0
                self.my_force_y[self.my_allSym_owned]=0.0
                self.my_force_z[self.my_allSym_owned]=0.0

            self.my_force_x[self.my_fixed_X_local]=np.subtract(
                    self.my_fixedBC[self.my_fixed_X_local,0],
                    self.uReshape[self.my_fixed_X_local,0])
            self.my_force_y[self.my_fixed_Y_local]=np.subtract(
                    self.my_fixedBC[self.my_fixed_Y_local,1],
                    self.uReshape[self.my_fixed_Y_local,1])
            self.my_force_z[self.my_fixed_Z_local]=np.subtract(
                    self.my_fixedBC[self.my_fixed_Z_local,2],
                    self.uReshape[self.my_fixed_Z_local,2])

            F[::3] = self.my_force_x[:]
            F[1::3] = self.my_force_y[:]
            F[2::3] = self.my_force_z[:]
            self.time_compute=self.time_compute+time.time()-t_start


        except Exception, e:
            print "Python exception raised in computeF"
            print e
            return False
        #if self.rank==0: profiler.disable()
        return True


    def cutLine2D(self,u,pt1,pt2):
        (p1x,p1y)=pt1
        (q1x,q1y)=pt2
        #uReshape = self.reshapeU(u)+self.my_augmented_reference
        self.my_u_worker.Import(u,self.dof_worker_importer,Epetra.Insert)
        self.my_u_worker.Update(1.0,self.xReshape,1.0)
        #p_points = self.uReshape[self.my_bondpair_p_local]
        #q_points = self.uReshape[self.my_bondpair_q_local]
        bendbreak = np.zeros_like(self.my_bondpair_x_local)
        extbreak = np.zeros_like(self.my_bond_x_local)
        
        calcforce.pairvectors(
            self.my_u_worker,
            self.my_pair_p_vectors,self.my_pair_q_vectors,
            self.my_pair_p_lengths,self.my_pair_q_lengths,
            self.my_vsources_A_local,self.my_vsources_B_local,
            self.my_vsources_C_local,
            self.my_vweights_A,self.my_vweights_B,self.my_vweights_C,
            self.my_bondpair_x_local,self.my_bondpair_p_local,
            self.my_bondpair_q_local)
        p_points=self.uReshape[self.my_bondpair_x_local]+self.my_pair_p_vectors
        q_points=self.uReshape[self.my_bondpair_x_local]+self.my_pair_q_vectors

        for index in range(len(self.my_bondpair_p_local)):
            [p2x,p2y,p2z]=p_points[index]
            [q2x,q2y,q2z]=q_points[index]
            o1 = orientation(p1x,p1y,q1x,q1y,p2x,p2y)
            o2 = orientation(p1x,p1y,q1x,q1y,q2x,q2y)
            o3 = orientation(p2x,p2y,q2x,q2y,p1x,p1y)
            o4 = orientation(p2x,p2y,q2x,q2y,q1x,q1y)

            if (o1!=o2 and o3!=o4):
                bendbreak[index]=1
            elif (o1==0 and onSegment(p1x,p1y,p2x,p2y,q1x,q1y)):
                bendbreak[index]=1
            elif (o2==0 and onSegment(p1x,p1y,q2x,q2y,q1x,q1y)):
                bendbreak[index]=1
            elif (o3==0 and onSegment(p2x,p2y,p1x,p1y,q2x,q2y)):
                bendbreak[index]=1
            elif (o4==0 and onSegment(p2x,p2y,q1x,q1y,q2x,q2y)):
                bendbreak[index]=1

        my_breakable = np.sum(bendbreak)
        
        bend_breakable = self.__comm.SumAll(my_breakable)
        if bend_breakable > 0:
            my_num_broken = np.sum(bendbreak)

            self.my_pair_health = np.where(
                bendbreak>0.5,
                0.0,self.my_pair_health)

        return 1

    def explicitStep(self,time,u,num_exSteps,timestep,damping,init=False,v_init=None):
        if init:
            self.inv_mass = Epetra.Vector(u.Map())
            self.inv_mass[:] = np.repeat(np.reciprocal(8000.0*self.my_area*self.thickness),3)
            self.u_current = Epetra.Vector(u)
            self.u_old = Epetra.Vector(u.Map())
            self.f_current = Epetra.Vector(u.Map())
            self.a_current = Epetra.Vector(u.Map())
            self.v_current = Epetra.Vector(u.Map())
            self.v_old = Epetra.Vector(u.Map())
            self.computeF(self.u_current,self.f_current,0)
            self.a_current.Multiply(1.0,self.f_current,self.inv_mass,0.0)
            residual = self.f_current.Norm2()
            if comm.MyPID()==0: print "initial residual: "+str(residual)

        if v_init is not None:
            self.v_current[:]=v_init[:]
        # Verlet integration
        self.u_current[:] = u[:]

        tsqrd=timestep**2
        tdamp = 1.0-timestep*damping/2
        with np.errstate(invalid='raise'):
            for exStep in range(num_exSteps):
                self.v_current.Update(timestep/2,self.a_current,tdamp)
                #if len(self.load_dof)>0:
                    #self.v_current.ReplaceMyValues(self.load_velocity*np.ones_like(self.load_dof),self.load_dof)
                self.u_current.Update(timestep,self.v_current,1.0)
                #self.computeF(self.u_current,self.f_current,6)
                self.computeF(self.u_current,self.f_current,0)
                self.a_current.Multiply(1.0,self.f_current,self.inv_mass,0.0)
                self.v_current.Update(timestep/2,self.a_current,tdamp)
                #if len(self.load_dof)>0:
                    #self.v_current.ReplaceMyValues(disp_velocity,self.load_dof)
                time+=timestep
                #if len(self.load_dof)>0:
                    ##if time<self.load_duration:
                    ##if time<0.5*self.load_duration:
                    ##disp=self.load_amplitude*np.sin(np.pi*time/self.load_duration)
                    #disp=self.load_velocity*time
                    ##else:
                        ###disp=0.0
                        ##disp=self.load_amplitude
                    #self.u_current.ReplaceMyValues(disp*np.ones_like(self.load_dof),self.load_dof)
                    #self.v_current.ReplaceMyValues(self.load_velocity*np.ones_like(self.load_dof),self.load_dof)

        residual = self.f_current.Norm2()
        if comm.MyPID()==0: print "residual: "+str(residual)
        return (time,self.u_current,self.v_current)

    def updatePlasticity(self,u):
        """Won't work with SWIG ComputeF, not sure if worth updating"""
        try:
            uReshape = self.reshapeU(u)+self.my_augmented_reference

            self.my_pairPvectors = (
                uReshape[self.my_bondpair_p_local]-uReshape[self.my_bondpair_x_local])
            self.my_pairQvectors = (
                uReshape[self.my_bondpair_q_local]-uReshape[self.my_bondpair_x_local])
            self.my_pairPlengths = np.sqrt(
                np.sum(self.my_pairPvectors*self.my_pairPvectors,axis=1))
            self.my_pairQlengths = np.sqrt(
                np.sum(self.my_pairQvectors*self.my_pairQvectors,axis=1))

            self.my_Hpq = np.cross(self.my_pairPvectors,self.my_pairQvectors)/(
                (self.my_pairPlengths*self.my_pairQlengths)[...,None])
            self.my_HpqNorms = np.sum(self.my_Hpq**2,axis=-1)**(1./2)
            self.my_HpqElastic = self.my_HpqNorms - self.my_bondpair_plasticAngle

            condition = [self.my_HpqElastic > self.criticalAngle,
                        self.my_HpqElastic < - self.criticalAngle,
                        self.my_HpqElastic > -2.0]
            choice = [self.my_HpqNorms-self.criticalAngle,
                        self.my_HpqNorms+self.criticalAngle,
                        self.my_bondpair_plasticAngle]
            self.my_bondpair_plasticAngle = np.select(condition,choice)



        except Exception, e:
            print "Python exception raised in updatePlasticity"
            print e
            return False
        return True

    def computeJacobian(self, x, Jac):
        """
           Computes the Jacobian matrix via coloring method.
        """

        try:

            #Instantiate NOX's built in finite difference coloring algorithm
            fdc = NOX.Epetra.FiniteDifferenceColoring({ }, self, x,
                    self.get_jacobian_graph(), True, False)
            #Compute the Jacobian with finite difference coloring
            fdc.computeJacobian(x)
            #Get the underlying matrix so we can apply the symmetric boundary
            #conditions
            Jac = fdc.getUnderlyingMatrix()

            #Optimize storage and complete fill of Jac
            Jac.FillComplete()
            Jac.Scale(-1.0)

            #Print Jacobian (for debugging)
            EpetraExt.RowMatrixToMatlabFile('NinePt.mat',Jac)


        except Exception, e:
            print "Exception in SteadyStateFluidProblem.computeJacobian method"
            print e
            return False
        return True

    def setPairHealth(self,unbalancedpairhealth):
        self.pairhealth_unbalanced[:]=unbalancedpairhealth
        self.pairhealth_balanced.Import(self.pairhealth_unbalanced,self.pairimporter,Epetra.Insert)
        self.my_pair_health[:]=self.pairhealth_balanced
        self.iso_pair_coefficients = self.iso_pair_coefficients*self.my_pair_health
        self.my_isoBend_stiffness = self.my_isoBend_stiffness*self.my_pair_health
        return True
 
    def initBondHealth(self):
        self.my_bond_health[self.bond1]=self.my_pair_health[:]
        self.my_bond_health[self.bond2]=self.my_pair_health[:self.numRealPairs]
        self.iso_ext_coefficients = self.iso_ext_coefficients*self.my_bond_health
        self.my_iso_ext_stiffness = self.my_iso_ext_stiffness*self.my_bond_health
        return True
 
    #Public "getter" functions

    def getSolution(self):
        return self.my_u

    def getX0(self):
        return self.my_x

    def getY0(self):
        return self.my_y

    def getZ0(self):
        return self.my_z

    def getPairHealth(self):
        self.pairhealth_balanced[:]=self.my_pair_health[:]
        self.pairhealth_unbalanced.Export(self.pairhealth_balanced,self.pairimporter,Epetra.Insert)
        return self.pairhealth_unbalanced[:]

    def getPairPlasticity(self):
        return self.my_bondpair_plasticAngle

    def getNodeHealth(self):
        with np.errstate(invalid='ignore'):
            self.my_node_health = np.divide(
                np.bincount(self.my_bondpair_x_local,weights=self.my_pair_health,
                minlength = len(self.my_families)),
                self.my_num_pairs)
        self.my_node_health = np.fmin([1.0],self.my_node_health)
        return self.my_node_health

    def get_graph(self):
        return self.__graph

    def get_jacobian_graph(self):
        return self.jacobian_graph

    def get_jacobian(self):
        return self.__jac

    def get_preconditioner(self):
        return self.__prec

    def get_balanced_map(self):
        return self.balanced_map

    def get_global_number_of_nodes(self):
        return self.__global_number_of_nodes

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def orientation(px,py,qx,qy,rx,ry): #orientation of pq and pr
    cross = (qy-py)*(rx-qx)-(qx-px)*(ry-qy)
    if (cross == 0):
        return 0
    elif (cross > 0):
        return 1
    else:
        return 2

def onSegment(px,py,qx,qy,rx,ry): #does q lie on pr for colinear pqr
    if (qx<=max(px,rx) and qx>=min(px,rx) and qy<=max(py,ry) and qy>=min(py,ry)):
        return True
    return False

def inLimits(shape,size,center,point):
    vector = point-center
    if shape=='rectangle':
        if (np.absolute(vector[0])<=size[0]/2.0 and
            np.absolute(vector[1])<=size[1]/2.0 and
            np.absolute(vector[2])<=size[2]/2.0):
            return True
    elif shape=='circle':
        return (np.sum(vector[:2]*vector[:2]) <= size*size)
    elif shape=='sphere':
        return (np.sum(vector*vector) <= size*size)
    else:
        print "Undefined shape ",shape
    return False


def addUnique(target,source):
    #sets target to contain the combined unique values of target and source
    #target must be long enough initially,
    #unique values must be > default value at end of target
    np.put(target,
        np.arange(np.unique(np.concatenate((target,source)))[1:].size),
        np.unique(np.concatenate((target,source)))[1:])
    return True

#Begin program
#pr = cProfile.Profile()
#pr.enable()
tBegin = time.time()
name = "GrapheneSquare_n21_h3_1" 
centroidfile = '../centroidfiles/Ring_t001_n20.npz'
positionfile = "../results/RingPinchCenter_n201_h3_3/RingPinchCenter_n201_h3_3_0_imp_2.npz"
parameterfile = "../results/Ring_t001_n40_g01/Ring_t001_n40_g01_parameters.npz"
pathname = "../results/"+name
make_sure_path_exists(pathname)
namebase = pathname+"/"+name+"_" 
loadParameters = False 
loadPositions = False 
#loadPositions = True
plotting = False

if (loadParameters): # Load parameters from file
    parameterFileName = namebase+"parameters"
    parameters = np.load(parameterfile)

    #Plate dimensions
    plate_length = parameters['plate_length']
    plate_width = parameters['plate_width']
    thickness = parameters['thickness']
    extension = parameters['extension']
    dimensions = [plate_length,plate_width,thickness,extension]

    #Plate material properties
    shear_mod = parameters['shear_mod']
    bulk_mod = parameters['bulk_mod']
    yieldstrain = parameters['yieldstrain']
    mat_type = parameters['mat_type']
    matl_properties = [shear_mod,bulk_mod,yieldstrain,mat_type]

    #Discretization
    nodesAcrossLength = parameters['nodesAcrossLength']
    nodesAcrossWidth = parameters['nodesAcrossWidth']
    horizon = parameters['horizon']
    discretization = [nodesAcrossLength,nodesAcrossWidth,horizon]

    np.savez(parameterFileName,
        plate_length = plate_length,
        plate_width = plate_width,
        thickness = thickness,
        extension = extension,
        shear_mod = shear_mod,
        bulk_mod = bulk_mod,
        yieldstrain = yieldstrain,
        mat_type = mat_type,
        nodesAcrossLength = nodesAcrossLength,
        nodesAcrossWidth = nodesAcrossWidth,
        horizon = horizon)

else: # Set parameters and save
    parameterFileName = namebase+"parameters"
    #Plate dimensions 
    plate_length = 1.0E1 
    plate_width = 1.0E1
    thickness = 3.35E-1
    extension = 1.0
    dimensions = [plate_length,plate_width,thickness,extension]

    #Plate material properties
    shear_mod = 4.374E-7
    bulk_mod = 5.11E-7
    yieldstrain = 1.0E-3
    mat_type = "elastic"
    matl_properties = [shear_mod,bulk_mod,yieldstrain,mat_type]
    bendingrigidity=2.31E-10
    gaussianstiffness=2.43E-10
    bendingproperties=[bendingrigidity,gaussianstiffness]

    #Discretization
    nodesAcrossLength = 21
    nodesAcrossWidth =  21
    horizonNorm = 3.001
    horizon = horizonNorm*plate_length/(nodesAcrossLength-1)
    #horizon = 0.01001
    discretization = [nodesAcrossLength,nodesAcrossWidth,horizon]

    np.savez(parameterFileName,
        plate_length = plate_length,
        plate_width = plate_width,
        thickness = thickness,
        extension = extension,
        shear_mod = shear_mod,
        bulk_mod = bulk_mod,
        yieldstrain = yieldstrain,
        mat_type = mat_type,
        nodesAcrossLength = nodesAcrossLength,
        nodesAcrossWidth = nodesAcrossWidth,
        horizon = horizon)

#Solver parameters
solvertype = "explicit"
#solvertype = "implicit"
startingstep = 1
loadlist = [1.0]
tensionmultiplier = 4.05*(3.0*bulk_mod+4.0*shear_mod)/(6.0*bulk_mod+2.0*shear_mod)
numLoadsteps = len(loadlist)
numIsubsteps = 3
numEsubsteps = 3
longstep = 1
medstep = 500
medstep = 50000
#medstep = 2000
#medstep = 5000
num_medsteps = 500
num_medsteps = 20
shortstep = 1
#timestep = 1.0E-6
#timestep = 1.0E-8
timestep = 5.0E-7
timestep = 1.0E-10
timestep = timestep
damping = 500.0
damping = 5000.0
#damping = 0.0
maxbreak = 30
simulation_time=0.0

#applied load parameters
gamma = 0.1
gamma = 0.00010
gamma2 = 0.0
loadDisplacement = -0.01
load_velocity = 0.01
#load_velocity = 0.1
#load_velocity = 0.00
load_duration = 1.0e-2
load_amplitude=1.0e-3
#load_amplitude=5.0e-5

comm = Epetra.PyComm()

if True:#(solvertype=='implicit'): #Set NOX parameters
    nlParams = NOX.Epetra.defaultNonlinearParameters(comm,2)
    printParams = nlParams["Printing"]
    lsParams = nlParams["Linear Solver"]
    lsParams["Preconditioner"]="None"
    lsParams["Preconditioner Operator"]="Use Jacobian"
    nlParams["Direction"]["Newton"]={}
    nlParams["Direction"]["Newton"]["Linear Solver"]={}
    nlParams["Direction"]["Newton"]["Linear Solver"]["Jacobian Operator"]="Matrix-Free"
    nlParams["Direction"]["Newton"]["Linear Solver"]["Preconditioner"]="None"
    nlParams["Direction"]["Newton"]["Linear Solver"]["Max Iterations"]=4000
    if False: #Other implicit solver options
        lsParams["Aztec Solver"] = "CGS"
        nlParams["Newton"]["Forcing Term Method"]="Type 1"
        nlParams["Direction"]["Method"]="NonlinearCG"
        nlParams["Line Search"]["Method"]="NonlinearCG"
        nlParams["Direction"]["Method"]="Steepest Descent"
        nlParams["Line Search"]["Method"]="More'-Thuente"
        nlParams["Line Search"]["More'-Thuente"]={}
        nlParams["Line Search"]["More'-Thuente"]["Sufficient Decrease"] = 0.1
        nlParams["Line Search"]["More'-Thuente"]["Curvature Condition"] = 0.1
        nlParams["Line Search"]["More'-Thuente"]["Recovery Step"] = 0.00001
        nlParams["Line Search"]["Method"]="Full Step"
        nlParams["Line Search"]["Full Step"]={}
        nlParams["Line Search"]["Full Step"]["Full Step"]=0.8
        nlParams["Line Search"]["Method"]="Backtrack"
        nlParams["Line Search"]["Backtrack"]={}
        nlParams["Line Search"]["Backtrack"]["Default Step"]=0.1
        nlParams["Line Search"]["Backtrack"]["Reduction Factor"]=0.5
        nlParams["Line Search"]["Backtrack"]["Recovery Step"] = 0.00000001
        nlParams["Line Search"]["Method"]="Polynomial"
        nlParams["Line Search"]["Polynomial"]={}
        nlParams["Line Search"]["Polynomial"]["Interpolation Type"]="Quadratic3"
        nlParams["Line Search"]["Polynomial"]["Force Interpolation"]=True
        nlParams["Line Search"]["Polynomial"]["Default Step"]=2.0
        nlParams["Line Search"]["Polynomial"]["Min Bounds Factor"]=0.05
        nlParams["Line Search"]["Polynomial"]["Max Bounds Factor"]=0.95
        nlParams["Line Search"]["Polynomial"]["Recovery Step"]=0.1
        nlParams["Line Search"]["Polynomial"]["Alpha Factor"]=0.05
        nlParams["Line Search"]["Polynomial"]["Sufficient Decrease Condition"]="Ared/Pred"
        if comm.MyPID()==0:
            print "nlParams"
            print nlParams


problem = PlateProblem(dimensions,discretization,matl_properties,bendingproperties)
#problem = PlateProblem(dimensions,discretization,matl_properties,None,centroidfile)
#problem = PlateProblem(dimensions,discretization,matl_properties)

problem.gamma=gamma
problem.load_velocity=load_velocity
problem.load_duration=load_duration
problem.load_amplitude=load_amplitude
# problem.gamma2=gamma2
# problem.loadDisplacement = loadDisplacement
ringNorm = (9.0/8.0)*gamma*((plate_length/2.0)**3.0)*(0.149)/(plate_width)
#plate ring is 9/8 stiffness of beam ring
ringNorm = gamma*((plate_length/2.0)**3.0)*(0.149)/(plate_width)
plate4ptnorm = (99.0/6144.0)*gamma*(plate_length**3.0)*(yieldstrain/thickness)/plate_width
halfRingNorm = ringNorm/2.0
norm = halfRingNorm
#norm = plate4ptnorm 
if comm.MyPID()==0: print "Expected ux",norm
if comm.MyPID()==0: print (time.time()-tBegin), " time elapsed"
filename = namebase+str(0)

# vector_variables = []
# scalar_variables = ['ux','uy','uz','health','radius']
# outfile = Ensight('output', vector_variables, scalar_variables,problem.comm, viz_path=VIZ_PATH)
if True:#initialize Epetra vector variables
    #initialize unbalanced (xxx_u) variables (all on processor 0)
    ux_u = Epetra.Vector(problem.unbalanced_map)
    uy_u = Epetra.Vector(problem.unbalanced_map)
    uz_u = Epetra.Vector(problem.unbalanced_map)
    vx_u = Epetra.Vector(problem.unbalanced_map)
    vy_u = Epetra.Vector(problem.unbalanced_map)
    vz_u = Epetra.Vector(problem.unbalanced_map)
    fx_u = Epetra.Vector(problem.unbalanced_map)
    fy_u = Epetra.Vector(problem.unbalanced_map)
    fz_u = Epetra.Vector(problem.unbalanced_map)
    nodeHealth_u = Epetra.Vector(problem.unbalanced_map)

    #initialize balanced (xxx_b) variables (distributed)
    ux_b = Epetra.Vector(problem.balanced_map)
    uy_b = Epetra.Vector(problem.balanced_map)
    uz_b = Epetra.Vector(problem.balanced_map)
    vx_b = Epetra.Vector(problem.balanced_map)
    vy_b = Epetra.Vector(problem.balanced_map)
    vz_b = Epetra.Vector(problem.balanced_map)
    fx_b = Epetra.Vector(problem.balanced_map)
    fy_b = Epetra.Vector(problem.balanced_map)
    fz_b = Epetra.Vector(problem.balanced_map)
    nodeHealth_b = Epetra.Vector(problem.balanced_map)

    Ftemp = Epetra.Vector(problem.balanced_dof_map)

    u_b = Epetra.Vector(problem.balanced_dof_map)
    u_b[:] = problem.getSolution()
    v_b = Epetra.Vector(problem.balanced_dof_map)
    connectGraph = problem.getGraph()
    ufinal = Epetra.Vector(u_b.Map())
    x0_b = Epetra.Vector(problem.balanced_map)
    y0_b = Epetra.Vector(problem.balanced_map)
    z0_b = Epetra.Vector(problem.balanced_map)
    x0_u = Epetra.Vector(problem.unbalanced_map)
    y0_u = Epetra.Vector(problem.unbalanced_map)
    z0_u = Epetra.Vector(problem.unbalanced_map)

x0_b[:] = problem.getX0()
y0_b[:] = problem.getY0()
z0_b[:] = problem.getZ0()

x0_u.Export(x0_b,problem.importer,Epetra.Insert)
y0_u.Export(y0_b,problem.importer,Epetra.Insert)
z0_u.Export(z0_b,problem.importer,Epetra.Insert)

# init = startingstep*1.0/numLoadsteps
init = 1.0

if (loadPositions): #Load previous result file

    loadhealth=False
    loadvelocity=False
    pairhealthunbalanced=[]
    if comm.MyPID()==0:
        scale = 1.0
        print "Load disp from",positionfile,"scaled by",scale
        #print "Displacements multiplied by",scale
        loaddata = np.load(positionfile)
        ux0 = loaddata['ux']*(scale)
        uy0 = loaddata['uy']*(scale)
        uz0 = loaddata['uz']*(scale)
        h0 = loaddata['pairhealth']
        pairhealthunbalanced=h0

        if loadvelocity:
            simulation_time = loaddata['time']
        
        if loadvelocity:
            vx0 = loaddata['vx']*(scale)
            vy0 = loaddata['vy']*(scale)
            vz0 = loaddata['vz']*(scale)
        
        x0_load = loaddata['x0']
        y0_load = loaddata['y0']
        z0_load = loaddata['z0']
        loadnodes = np.transpose([x0_load,y0_load,z0_load])

        loadtree = scipy.spatial.cKDTree(loadnodes)
        mynodes = np.transpose([x0_u,y0_u,z0_u])

        load_sources = np.empty((len(mynodes),9),dtype=np.int32)

        #Find real nodes from which to estimate virtual node props
        for ix,xnode in enumerate(mynodes):
            _, load_sources[ix]=loadtree.query(
                xnode,k=9, eps=0.0, p=2,
                distance_upper_bound=2.0*horizon)

        A_nodes = loadnodes[load_sources[:,0],:]
        B_nodes = loadnodes[load_sources[:,1],:]
        C_nodes = loadnodes[load_sources[:,2],:]
        repeat = 2
        while repeat>0:
            AB = B_nodes-A_nodes
            AC = C_nodes-A_nodes
            BA = -AB
            BC = C_nodes-B_nodes
            CA = -AC
            CB = -BC
            A_cross = np.cross(BC,BA)
            sinSQ = (A_cross*A_cross).sum(axis=-1)/((BC*BC).sum(axis=-1)*(BA*BA).sum(axis=-1))
            triangular = (sinSQ>0.3)
            if np.all(triangular):
                break
            numlinear = np.where(triangular,0.0,1.0).sum()
            repeat = repeat+1
            print "fixing",numlinear,"colinear supports",repeat
            load_sources[:,2]=np.where(
                triangular,load_sources[:,2],load_sources[:,repeat])
            C_nodes = loadnodes[load_sources[:,2],:]

        B_cross = np.cross(CA,CB)
        C_cross = np.cross(AB,AC)
        A_cross = A_cross/(np.sqrt((A_cross*A_cross).sum(axis=-1))[:,None])
        B_cross = B_cross/(np.sqrt((B_cross*B_cross).sum(axis=-1))[:,None])
        C_cross = C_cross/(np.sqrt((C_cross*C_cross).sum(axis=-1))[:,None])
        AX = A_nodes-mynodes
        BX = B_nodes-mynodes
        CX = C_nodes-mynodes

        WA = (BX*np.cross(BC,A_cross)).sum(axis=-1)
        WB = (CX*np.cross(CA,B_cross)).sum(axis=-1)
        WC = (AX*np.cross(AB,C_cross)).sum(axis=-1)
                 #print "use nearest neighbor interpolation instead"
                 #WA = np.ones_like(WA)
                 #WB = np.zeros_like(WB)
                 #WC = np.zeros_like(WC)
             #vpt_sources=np.where(vpt_sources ==  tree.n, -1,
                 #vpt_sources)
             #vpt_distances = vpt_distances[:num_vpts]
             #vpt_weights=1/vpt_distances
             #vpt_weights=np.divide(vpt_weights,np.sum(vpt_weights,axis=1)[:,None])
        load_sources=load_sources[:,:3]
        load_weights=np.transpose([WA,WB,WC])/((WA+WB+WC)[:,None])
        for ix,xnode in enumerate(mynodes):
            ux_u[ix]= np.sum(load_weights[ix]*ux0[load_sources[ix]])
            uy_u[ix]= np.sum(load_weights[ix]*uy0[load_sources[ix]])
            uz_u[ix]= np.sum(load_weights[ix]*uz0[load_sources[ix]])
            if loadvelocity:
                vx_u[ix]= np.sum(load_weights[ix]*vx0[load_sources[ix]])
                vy_u[ix]= np.sum(load_weights[ix]*vy0[load_sources[ix]])
                vz_u[ix]= np.sum(load_weights[ix]*vz0[load_sources[ix]])

    ux_b.Import(ux_u,problem.importer, Epetra.Insert)
    u_b[0::3] = ux_b[:]
    uy_b.Import(uy_u,problem.importer, Epetra.Insert)
    u_b[1::3] = uy_b[:]
    uz_b.Import(uz_u,problem.importer, Epetra.Insert)
    u_b[2::3] = uz_b[:]

    if loadvelocity:
        simulation_time=comm.MaxAll(simulation_time)
        vx_b.Import(vx_u,problem.importer, Epetra.Insert)
        v_b[0::3] = vx_b[:]
        vy_b.Import(vy_u,problem.importer, Epetra.Insert)
        v_b[1::3] = vy_b[:]
        vz_b.Import(vz_u,problem.importer, Epetra.Insert)
        v_b[2::3] = vz_b[:]
    else:
        if comm.MyPID()==0:print "don't import velocity"
    if loadhealth:
        problem.setPairHealth(pairhealthunbalanced)
    else:
        if comm.MyPID()==0:print "don't import health"
    if mat_type != "elastic":
        problem.initBondHealth()

else:
    if False:#recenter to origin
        xOffset = plate_length/2.0
        multiplier = (init*gamma*(yieldstrain/thickness)
            *((plate_length/2.0)**3.0)*(1.0/8.0)/(np.pi*plate_width))
        abeta = np.absolute(np.arctan2(z0_b,x0_b-xOffset))
        ux_b[:] = multiplier*(8.0-2.0*np.pi**2+2*np.pi*abeta+8*np.cos(abeta)
            -4*(np.pi-2*abeta)*np.sin(abeta)+np.pi*np.sin(2*abeta))-halfRingNorm
        uz_b[:] = -multiplier*(z0_b/np.absolute(z0_b))*(3*np.pi-4*(np.pi-2*abeta)*np.cos(abeta)
            + np.pi*np.cos(2*abeta)-8*np.sin(abeta))
    if False:   
        if comm.MyPID()==0:print "initial disp of pinched ring,",
        multiplier = -(init*gamma*(yieldstrain/thickness)
            *((plate_length/2.0)**3.0)*(1.0/8.0)/(np.pi*plate_width))
        abeta = np.absolute(np.arctan2(z0_b,x0_b))
        abetamax = np.absolute(np.arctan2(0,(1)))

        ux_b[:]= multiplier*(8.0-2.0*np.pi**2+2*np.pi*abeta+8*np.cos(abeta)
            -4*(np.pi-2*abeta)*np.sin(abeta)+np.pi*np.sin(2*abeta))

        offsetx = 0.5*multiplier*(8.0-2.0*np.pi**2+2*np.pi*abetamax+8*np.cos(abetamax)
            -4*(np.pi-2*abetamax)*np.sin(abetamax)+np.pi*np.sin(2*abetamax))
        ux_b[:]-=offsetx

        uz_b[:]= -multiplier*(z0_b/np.absolute(z0_b))*(3*np.pi-4*(np.pi-2*abeta)*np.cos(abeta)
            + np.pi*np.cos(2*abeta)-8*np.sin(abeta))

    if True:#SSSS elastic plate uniform load initial guess
        if comm.MyPID()==0:print "initial disp SSSS plate, uniform load,\n"
        multiplier = 16.0*init*(gamma/(plate_length*plate_width))*(yieldstrain/thickness)/(np.pi**6.0)
        for m in range(1,8,2):
            for n in range(1,8,2):
                denominator = (m*n*((m/plate_length)**2.0+(n/plate_width)**2.0)**2.0)
                mnterm =(np.sin(m*np.pi*x0_b/plate_length)
                    *np.sin(n*np.pi*y0_b/plate_width)
                    /denominator)
                uz_b[:] = uz_b[:]+mnterm
        uz_b[:] = uz_b[:]*multiplier
        uz_b[:] = uz_b[:]*(9.0*bulk_mod+12.0*shear_mod)/(12.0*bulk_mod+4.0*shear_mod)

    if False:#Plate with 4 point bend
        if comm.MyPID()==0:print "initial disp SS plate, 4pt load,\n"
        #if comm.MyPID()==0:print "cut center crack"
        #problem.cutLine2D(u_b,(0.5001,0.3999),(0.5001,0.6001))
        if comm.MyPID()==0:print "cut center crack at angle"
        #problem.cutLine2D(u_b,(0.4001,0.3999),(0.6001,0.6001))
        problem.cutLine2D(u_b,(0.4,0.4),(0.6,0.6))
        problem.initBondHealth()
        if False:#initial positions
            multiplier=(9.0/1536.0)*gamma*(yieldstrain/thickness)/plate_width
            left = x0_b*(9.0*plate_length**2-16.0*x0_b**2)
            center = -plate_length*(plate_length**2-48*plate_length*x0_b+48*x0_b**2)/4.0
            right=-7*plate_length**3+39*plate_length**2*x0_b-48*plate_length*x0_b**2+16*x0_b**3
            uz_b[:]=np.where(x0_b<plate_length*0.25,left,center)
            uz_b[:]=np.where(x0_b>plate_length*0.75,right,uz_b)
            uz_b[:] = uz_b[:]*multiplier
    if False:#Double torsion plate
        if comm.MyPID()==0:print "initial cut for double torsion test,",
        problem.cutLine2D(u_b,(-1.0,0.5001),(0.3001,0.5001))
        problem.initBondHealth()

u_b[0::3]=ux_b[:]
u_b[1::3]=uy_b[:]
u_b[2::3]=uz_b[:]

initGuess = Epetra.Vector(problem.balanced_dof_map,u_b)

ux_u.Export(ux_b,problem.importer,Epetra.Insert)
uy_u.Export(uy_b,problem.importer,Epetra.Insert)
uz_u.Export(uz_b,problem.importer,Epetra.Insert)
if comm.MyPID()==0:
    maxU = ux_u[np.argmax(np.absolute(ux_u))]
    print "largest ux:",maxU
    if np.absolute(norm)>0.0:
        print "Normalized to desired ux:",(maxU/norm)

x0_u.Export(x0_b,problem.importer,Epetra.Insert)
y0_u.Export(y0_b,problem.importer,Epetra.Insert)
z0_u.Export(z0_b,problem.importer,Epetra.Insert)
#initGuess[:]=0.0

if (comm.MyPID()==0 and plotting):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x0_u+ux_u,y0_u+uy_u,z0_u+uz_u, ls="None", marker="o")

     #plt.xlim((-extension,plate_length+extension))
    plt.savefig(namebase)
    plt.close(fig)

for loadstep,load in enumerate(loadlist):
    filename = namebase+str(loadstep+1)
    if comm.MyPID()==0: print filename," of ",str(len(loadlist))

     #problem.step_BCs(loadstep,numLoadsteps)
    problem.step_BCs(load,1.0)

    if (solvertype == "explicit"):
        nobreak = 0
        iteration = 0
        while (nobreak < num_medsteps):
            iteration = iteration + 1
            numbreak = 0 
            if iteration == 1:
                (simulation_time,initGuess,v_b)=problem.explicitStep(
                    simulation_time,initGuess,medstep,timestep,damping,init=True,v_init=v_b)
            else:
                (simulation_time,initGuess,v_b)=problem.explicitStep(
                    simulation_time,initGuess,medstep,timestep,damping,v_init=v_b)
            nobreak+=1

            filename = namebase+str(loadstep+1)+"_exp_"+str(iteration)

            ux_b[:] = initGuess[0::3]
            uy_b[:] = initGuess[1::3]
            uz_b[:] = initGuess[2::3]

            vx_b[:] = v_b[0::3]
            vy_b[:] = v_b[1::3]
            vz_b[:] = v_b[2::3]
            nodeHealth_b[:] = problem.getNodeHealth()
            
            ux_u.Export(ux_b,problem.importer,Epetra.Insert)
            uy_u.Export(uy_b,problem.importer,Epetra.Insert)
            uz_u.Export(uz_b,problem.importer,Epetra.Insert)
            
            vx_u.Export(vx_b,problem.importer,Epetra.Insert)
            vy_u.Export(vy_b,problem.importer,Epetra.Insert)
            vz_u.Export(vz_b,problem.importer,Epetra.Insert)
            
            fx_u.Export(problem.my_force_x,problem.importer,Epetra.Insert)
            fy_u.Export(problem.my_force_y,problem.importer,Epetra.Insert)
            fz_u.Export(problem.my_force_z,problem.importer,Epetra.Insert)
            unbalanced_pairhealth=problem.getPairHealth()
            nodeHealth_u.Export(nodeHealth_b,problem.importer,Epetra.Insert)

            if comm.MyPID()==0:
                print "Saving",filename # ," of ",numEsubsteps
                print "time elapsed: ", time.time()-tBegin,"simulated:",simulation_time
                #maxU = ux_u[np.argmax(np.absolute(ux_u))]
                #print "largest ux:",maxU,"Total health",np.mean(nodeHealth_u)
                maxU = uz_u[np.argmax(np.absolute(uz_u))]
                print "largest uz:",maxU,"Total health",np.mean(nodeHealth_u)
                #if np.absolute(norm)>0.0:
                    #print "Normalized to desired uz:",(maxU/norm)
                np.savez(filename,
                    ux=ux_u,uy=uy_u,uz=uz_u,
                    vx=vx_u,vy=vy_u,vz=vz_u,
                    #fx=fx_u,fy=fy_u,fz=fz_u,
                    x0=x0_u,y0=y0_u,z0=z0_u,
                    nodeHealth=nodeHealth_u,
                    pairhealth=unbalanced_pairhealth,
                    time=simulation_time)
                if plotting:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot(x0_u+ux_u,y0_u+uy_u,z0_u+uz_u, ls="None", marker="o")

                    plt.savefig(filename)
                    plt.close(fig)

    solvertype = "implicit"
    if (solvertype == "implicit"):
        noxInitGuess = NOX.Epetra.Vector(initGuess,NOX.Epetra.Vector.CreateView)
        fdc  = NOX.Epetra.FiniteDifferenceColoring(printParams, problem,
             noxInitGuess,connectGraph, False, False)

        for substep in range(1,numIsubsteps+1):
            filename = namebase+str(loadstep)+"_imp_"+str(substep)

            mf = NOX.Epetra.MatrixFree(printParams,problem,noxInitGuess)

            solver = NOX.Epetra.defaultSolver(initGuess, problem, mf, mf, fdc, fdc, nlParams,
                maxIters=5)

            solveStatus=solver.solve()

            if comm.MyPID()==0:
                print "Finished substep "+str(substep)
                print "time elapsed: ", time.time()-tBegin
                #print "tCompute",problem.time_compute

            finalGroup = solver.getSolutionGroup()
            ufinal[:] = finalGroup.getX()
            u_b = ufinal

            initGuess = Epetra.Vector(problem.balanced_dof_map,u_b)

            ux_b[:] = initGuess[0::3]
            uy_b[:] = initGuess[1::3]
            uz_b[:] = initGuess[2::3]
            nodeHealth_b[:] = problem.getNodeHealth()
            ux_u.Export(ux_b,problem.importer,Epetra.Insert)
            uy_u.Export(uy_b,problem.importer,Epetra.Insert)
            uz_u.Export(uz_b,problem.importer,Epetra.Insert)
            fx_u.Export(problem.my_force_x,problem.importer,Epetra.Insert)
            fy_u.Export(problem.my_force_y,problem.importer,Epetra.Insert)
            fz_u.Export(problem.my_force_z,problem.importer,Epetra.Insert)
            unbalanced_pairhealth=problem.getPairHealth()
            nodeHealth_u.Export(nodeHealth_b,problem.importer,Epetra.Insert)

            if comm.MyPID()==0:
                print "Saving",filename # ," of ",numEsubsteps
                print "time elapsed: ", time.time()-tBegin
                maxU = ux_u[np.argmax(np.absolute(ux_u))]
                print "largest ux:",maxU
                print "Normalized to desired ux:",(maxU/norm)
                #print "u"
                #print initGuess
                np.savez(filename,
                    ux=ux_u,uy=uy_u,uz=uz_u,
                    fx=fx_u,fy=fy_u,fz=fz_u,
                    x0=x0_u,y0=y0_u,z0=z0_u,
                    nodeHealth=nodeHealth_u,
                    pairhealth=unbalanced_pairhealth)
                if plotting:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot(x0_u+ux_u,y0_u+uy_u,z0_u+uz_u, ls="None", marker="o")
                     #ax.scatter(x0_u+ux_u,y0_u+uy_u,z0_u+uz_u, c=nodeHealth_u,cmap=plt.cm.rainbow, marker="o")

                    plt.savefig(filename)
                    plt.close(fig)

#pr.disable()
#s = StringIO.StringIO()
#sortby = 'cumulative'
#ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#ps.print_stats()
#if comm.MyPID()==0:
    ##profiler.print_stats()
    #ps.dump_stats("Plate_lineprofileSerial.profile")
    ##print s.getvalue()
