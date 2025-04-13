#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 16:39:29 2025

@author: lcabotsalar
"""

import random 
import numpy as np
import matplotlib . pyplot as plt
from math import exp, sqrt, pi


def particule(n):
    """
   Simule le déplacement aléatoire d'une particule sur une ligne (marche aléatoire en 1D).

   À chaque pas, la particule avance de +1 ou -1 avec une probabilité égale.
   Le déplacement total après n pas est retourné.

   Paramètre :
   n (int) : nombre de pas effectués par la particule.

   Retour :
   int : position finale de la particule après n déplacements.
   """
    position=0
    for k in range (0,n):
        position+=random.choice([-1,1])
    return position



def coord(L):
    """
   Compte combien de fois chaque position apparaît dans la liste L.

   Paramètre :
   L (list) : liste des positions finales de particules.

   Retour :
   dict : dictionnaire {position: nombre_de_particules_à_cette_position}.
   """
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
f2= lambda x,d : sqrt(1/((8*pi*d)))*exp(-(x**2/8*d))   
f3=lambda k,d : sqrt(1/((pi*1/2*1*d)))*exp(-(k**2/(4*1/2*d)))

def particool(n,d):
    """
    Simule la diffusion de n particules effectuant d pas aléatoires, 
    puis compare l'histogramme des positions finales avec la densité 
    de probabilité théorique issue d'une loi normale.

    Paramètres :
    n (int) : nombre de particules simulées.
    d (int) : nombre de pas effectués par chaque particule.

    Retourne un graphique avec :
        - l'histogramme des positions finales,
        - la courbe théorique de densité (loi normale).
    """
    L=[particule(d) for i in range (n)]
    
    X=[k for k in coord(L).values()]
    Y=[k for k in coord(L).keys()]
    
    plt . figure ( " Graphe 1 " )
    plt . ylabel ( 'nombre de particules ')
    plt . xlabel ( 'position de la particule')
    
    k = np . linspace ( -30 ,30 ,500)
    y = [f3(i,d)*n for i in k]
     
    plt . bar(X, Y, label='histogramme')
     
    
    plt . plot (k, y , "r", label = 'densité de proba')  
    plt.title('Comparaison entre histogramme et densité de probabilité de présence de la particule ') 
    plt.legend()       
    plt . show ()
    

 

