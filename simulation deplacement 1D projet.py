#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:39:29 2025

@author: lcabotsalar
"""

import random 
import numpy as np
import matplotlib . pyplot as plt
from math import exp, sqrt,pi, e


def particule(n):
    y=0
    for k in range (0,n):
        y+=random.choice([-1,1])
    return y



def coord(L):
    D={}
    
    for k in range (len(L)):
        a=L[k]
        n=0
        for i in range (len(L)):
            if L[i]==a:
                n+=1
        D[n]=a   
    return D

f= lambda x,d : sqrt(2/((pi*d)))*exp(-(x**2*0.5/d)) 

def particool(n,d):
    L=[particule(d)for i in range (n)]
    
    X=[k for k in coord(L).keys()]
    Y=[k for k in coord(L).values()]
    plt . figure ( " Graphe 1 " )
    plt . ylabel ( 'nombre de particules ')
    plt . xlabel ( 'position de la particule')
    x = np . linspace ( -30 ,30 ,500)
    y = [f(i,d)*n for i in x]
     
    plt . bar(Y, X, label='historigramme')
     
    plt . plot (x, y , "r", label = 'densité de proba')  
    plt.title('Comparaison entre historigramme et densité de probabilité de présence de la particule ')
    plt.legend()       
    plt . show () 
      
    

 

