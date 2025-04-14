from tracker_v15 import *


#Bonjour,voici le fichier permettant d'essayer le tracker de particules,

#Rentrez le chemin vers la vidéo des particules
video='Diam_0.50µm_Obj_60_9.avi'

"""
Pour faire fonctionner le tracker il suffit de lancer ce programme,
cliquer sur la particule que l'on veut suivre, ajuster le cadre autour de la particule 
à l'aide de la barre en bas de page, il est préférable d'utiliser un cadre un
peu plus grand que la particule, ensuite il faut choisir un seuil adéquat grâce
à la barre en dessous de l'image. Afin que le tracker soit efficace,
il faut que le seuil soit choisi de sorte à ce que le tracker soit à la limite 
de pouvoir détecter le fond de l'image comme expliqué dans le compte rendu.
Il ne reste plus qu'à observer la particule suivie en direct par le point vert
pour s'assurer que le tracking est réussi.
"""


tracking(video)
