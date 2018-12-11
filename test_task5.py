## -*- coding: utf-8 -
#import numpy as np
#xx=[-1, 0 ,0.5, 1, 3, 0, 2, 0, 1, 2, 0, 2, 2, 1, 2, -2]
#yy=np.array([20, 1, 1, 2, 6, 7, 8, 3, 3, 0, 4, 3, 4, 4, 5, 20])
#
#tt=np.array([ 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
#
#boundary = np.array([1, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
#
## Remove Boundary particles
#x = []
#y = []
#t = []
#for i in range(len(boundary)):
#    if boundary[i] == False:
#        x.append(xx[i]);
#        y.append(yy[i]);
#        t.append(tt[i])
#
## time values without repetition
#ts= list(set(t))
#
#def t_index(t):
#    """
#    input: a list
#    Return: a list with coordiates where list[i] != list[i]
#    """
#    dt = np.diff(t)
#    store=[]
#    store.append(0)
#    for i in range(len(dt)):
#        
#        # coordinate when t changes value
#        coordinate=i+1
#        if dt[i] != 0:
#            store.append(coordinate)
#    
#    # last particle coordinate
#    store.append(len(dt)+1)
#    
#    return store
#
#tindex = t_index(t)
#
#
#def test_task5():
#    #Output_analytic = ode_solve.xx(inputs)
#    #output_numeric =
#    Crest_xcoords, Crest_height, ts, tslosh = peak(x, y, t, tindex, ts, 1, 2)
#
#    assert  (Crest_xcoords == [1, 2, 0, 0, 2])
#    assert  (Crest_height == [2.0, 8.0, 3.0, 4.0, 5.0])
#    assert  (ts == [0, 1, 2, 3, 4])
#    assert  (tslosh == 1)
