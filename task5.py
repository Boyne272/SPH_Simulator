# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

    

xx=[-1, 0 ,0.5, 1, 3, 0, 2, 0, 1, 2, 0, 2, 2, 1, 2, -2]
yy=np.array([20, 1, 1, 2, 6, 7, 8, 3, 3, 0, 4, 3, 4, 4, 5, 20])

tt=np.array([ 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])

boundary = np.array([1, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

# Remove Boundary particles
x = []
y = []
t = []
for i in range(len(boundary)):
    if boundary[i] == False:
        x.append(xx[i]);
        y.append(yy[i]);
        t.append(tt[i])


dt = np.diff(t)


# time values without repetition
ts= list(set(t))


def t_index(t):
    """
    input: a list
    Return: a list with coordiates where list[i] != list[i]
    """
    dt = np.diff(t)
    store=[]
    store.append(0)
    for i in range(len(dt)):
        
        # coordinate when t changes value
        coordinate=i+1
        if dt[i] != 0:
            store.append(coordinate)
    
    # last particle coordinate
    store.append(len(dt)+1)
    
    return store


# tindex = start index of each time
tindex = t_index(t)


def MVAVARAGE(List, N):
    """
    input:
        List- a list
        N- N point avarage
    Returns:
        N point avarage of the List
    
    """
    cumsum, moving_aves = [0], []
    
    for i, x in enumerate(List, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
        #else:
         #   moving_ave = 0
            moving_aves.append(moving_ave)
    return(moving_aves)

def peak(x, y, t, tindex, ts, mv_avN, wallpos):
    """
    Inputs: 
        ----------------------------------
        x - x_coordinates array (exclude walls)
        y - y_coordinates array (exclude walls)
        t - t_coordinates array (exclude walls)
        tindex - when time changes value (array)
        mv_avN - N move avarage (float or int)
        wallpos - x-coordinates of boundary (float or int)
        ---------------------------------
    Outputs:
        ---------------------------------
        Crest_xcoords (list), Crest_height (list), times(list), sloash time(flaot)
        ---------------------------------
    """

    Crest_height=[]
    Crest_xcoords=[]
    t_sloshing=[]
    for i in range(len(ts)):
        
        # Slice x-array to get a timeframe.
        gridpointsX = x[tindex[i] : tindex[i+1]]
        ypoints = y[tindex[i] : tindex[i+1]]
        gridpointsY = MVAVARAGE(ypoints, mv_avN)
        
        
        Peak_Indx = np.where(gridpointsY==max(gridpointsY))[0][0]
        Peak = gridpointsY[Peak_Indx]
        Crest_height.append(Peak)
        coord = gridpointsX[Peak_Indx]
        Crest_xcoords.append(coord)
        
        
        if coord >= 0.99*wallpos:
            t_sloshing.append(ts[i])
        
        #plt.plot(ts, Crest_height , 'o')
       
    tslosh = t_sloshing[0]
    
    print("Crest_height", Crest_height)
    print("Crest_xcoords", Crest_xcoords)
    print("tslosh", tslosh)
    print("ts", ts)
    
    return(Crest_xcoords, Crest_height, ts, tslosh)


"""
def t_repeat(t, ts):
    
    Nparticle = []
    for i in ts:
        NN = np.count_nonzero(t==i)
        Nparticle.append(NN)
        
    return(Nparticle)
"""
    
    