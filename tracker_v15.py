import cv2 #pip3 install opencv-python==4.1.1.26
import matplotlib.pyplot as plt
import numpy as np

#vidéo et taille taille (en m) d'un pixel sur la vidéo
video="C:/Users/Utilisateur/Documents/Travail/Projet S4/video_projet4.mp4"
taille_pixel=0.143e-6



#partie du code liée au tracking:

    


def barycentre(img,seuil):
    '''
    

    Parameters
    ----------
    img : liste
        image sous forme de tableau numpy 
        converti en niveau de gris
    seuil : int
        seuil de couleur à partir duquel on prend en compte un pixel
        pour en faire la moyenne des coordonnées

    Returns
    -------
    xG : int
        coordonnée x du barycentre de l'objet'
    yG : int
        coordonnée y du barycentre de l'objet'

    '''
    ytaille,xtaille=img.shape[0:2]
    coord=[]
    for x in range(xtaille):
        for y in range(ytaille):
            #on élimine les pixels en dessous du seuil
            if img[y,x]<=seuil:
                coord.append([y,x])
    #cas où on a mis un seuil trop élevé et que aucun pixel n'est pris en compte            
    if len(coord)==0:
        return 'aucun points',0
    #moyenne des coordonnées:
    yG=sum([coord[i][0] for i in range(len(coord))])//len(coord)
    xG=sum([coord[i][1] for i in range(len(coord))])//len(coord)
    return xG,yG


def recadrer(img,x,y,taille):
    '''
    

    Parameters
    ----------
    img : liste
        image sous forme de tableau numpy
    x : int
        coordonnée x du milieu de l'image recadrée
    y : TYPE
        coordonnée y du milieu de l'image recadrée
    taille : int
        largeur et hauteur ajoutée de chaque coté à partir du milieu

    Returns
    -------
    im_crop : liste
        image recadrée sous forme de tableau numpy
    
    la fonction peut être optimiser car on utilise des if pour chaque cas où le recadrage demandé déborde

    '''
    ytaille,xtaille=img.shape[0:2]
    if y-taille<0 and y+taille> ytaille:
        return 'les parametres sont pas bons chefs'
    elif y-taille<0 and y+taille> ytaille:
        return 'les parametres sont pas bons chefs' 
    elif y-taille<0 and x-taille<0:
        im_crop=img[0: y+taille, 0:x+taille]
        return im_crop
    elif y-taille<0 and x+taille>xtaille:
        im_crop=img[0: y+taille, x-taille:xtaille]
        return im_crop
    elif y+taille>ytaille and x-taille<0:
        im_crop=img[y-taille: ytaille, 0:x+taille]
        return im_crop
    elif y+taille>ytaille and x+taille>xtaille:
        im_crop=img[y-taille: ytaille, x-taille:xtaille]
        return im_crop
    elif y+taille>ytaille:
            im_crop=img[y-taille: ytaille, x-taille:x+taille]
            return im_crop
    elif x+taille>xtaille:
        im_crop=img[y-taille: y+taille, x-taille:xtaille]
        return im_crop
    elif x-taille<0:
        im_crop=img[y-taille: y+taille, 0:x+taille]
        return im_crop
    elif y-taille<0:
        im_crop=img[0: y+taille, x-taille:x+taille]
        return im_crop

    else:  
        im_crop=img[y-taille: y+taille, x-taille:x+taille]
        return im_crop
    

def tracking(video):
    '''
    

    Parameters
    ----------
    video : str
        le titre ou le chemin si la vidéo n'est
        pas au même endroit que le fichier python

    Retourne
    -------
    X : liste
        liste des coordonnée selon x 
    Y : liste
        liste des coordonnée selon y
    t : liste
        liste regroupant le temps à chaque image
    nombre_erreur: int 
        nombre de fois où la fonction barycentre n'a rien renvoyé car aucun pixel n'est pris en compte

    '''
    video_path =video  
    cap = cv2.VideoCapture(video_path)

    # Vérifier si la vidéo est ouverte correctement
    if not cap.isOpened():
        return "Erreur lors de l'ouverture de la vidéo"

    # Lire la première image pour initialiser la zone de suivi
    ret, frame = cap.read()

    if not ret:
        return "Impossible de lire la vidéo"
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    #selection du point de départ et de la taille du recadrage avec une interface
    point, taille = selectionner_zone(gray_frame)
    x, y = point
    #on recadre sur la zone selectionnée
    image_recadree=recadrer(gray_frame,x,y,taille)
    
    #selection du seuil avec une interface
    seuil = choisir_seuil(image_recadree)[1]
    
    nombre_erreur=0
    fps = cap.get(cv2.CAP_PROP_FPS)
    T=1/fps
    X=[]
    Y=[]
    t=[]
    temps=0
    #on fait tourner la vidéo
    while True:
        # Lire une frame de la vidéo
        ret, frame = cap.read()  
        
        
        if not ret:
            print("Fin de la vidéo ou erreur de lecture")
            break  # Si la vidéo e(st terminée ou si une erreur se produit, sortir de la boucle
        
        #conversion de l'image en niveau de gris
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #on recadre l'image là où était la particule à la frame d'avant
        image_recadree=recadrer(gray_frame,x,y,taille)
        #on récupère les coordonnées du barycentre de la particule sur l'image recadrée
        xG,yG=barycentre(image_recadree,seuil)
        #si jamais le seuil est trop foncé et que l'image recadrée est vue comme entièrement blanche par le programme on ajoute une erreur et on passe ce point
        if xG=='aucun points':
            nombre_erreur+=1
            continue
        #coordonées du barycentre de la particule sur l'image entière non recadrée: 
        x=x-taille+xG
        y=y-taille+yG
        X.append(x)
        Y.append(y)
        #a chaque frame on ajoute le temps qu'il passe entre 2 frames
        temps+=T
        t.append(round(temps,3))
        # Affichage en direct pour que l'utilisateur puisse verifier que le tracking se passe bien
        frame_copy = frame.copy()
        
        cv2.circle(frame_copy, (x, y), 3, (0, 255, 0), -1)
        cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) 
        cv2.imshow("Tracking", frame_copy)
        
        # Attendre 1 milliseconde pour qu'une touche soit pressée
        # Si la touche 'q' est pressée, arrêter la lecture
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #tracé de la trajectoire de la particule
    plt.plot(X,Y)
    
    #inverser l'axe des ordonnées car les axes sont inversés quand on lit la vidéo
    plt.gca().invert_yaxis()
    plt.xlabel('x (pixel)')
    plt.ylabel('y (pixel)')
    plt.title(f'Trajectoire de la particule pour seuil={seuil}')
    plt.show()

    #fin de la vidéo
    cap.release()
    cv2.destroyAllWindows()
    #pour avoir une idée du nombre de points sautés dans le tracking
    print(f"nombre d'erreur : {nombre_erreur}")
    return X, Y, t



def extraction_des_donnees(X, Y,taille_pixel):
    '''
    

    Parameters
    ----------
    X : liste
        liste des coordonnée selon x 
    Y : liste
        liste des coordonnée selon y
    t : liste
        liste regroupant le temps à chaque image

    Returns
    -------
    dico_coord: dictionnaire
        le dictionnaire dont le contenu est expliqué ci dessous 

    '''
    #on met les coordonnées à l'echelle réelle 
    X1=[X[i]*taille_pixel for i in range(len(X))]
    Y1=[Y[i]*taille_pixel for i in range(len(Y))]
    n = len(X1)
    #on créé un dictionnaire qui aura comme clés i : le nombre d'image entre les déplacements, et comme valeurs les déplacements en 1D entre les i images 
    dico_coord=dict()
    for i in range(1,n-1):
        dico_coord[i]=[]
        dico_coord[i]+=[X1[k+i]-X1[k] for k in range(n-i)]
        dico_coord[i]+=[Y1[k+i]-Y1[k] for k in range(n-i)]
    return dico_coord
        

def fonction_finale(video):
    '''
    assemble les fonction tracking et extraction_des_donnees pour aller plus vite lors de l'analyse des vidéos'

    Parameters
    ----------
    video : str
        le titre ou le chemin si la vidéo n'est
        pas au même endroit que le fichier python

    Returns
    -------
    dico_coord: dictionnaire
        le dictionnaire dont le contenu est expliqué dans la fonction extraction_des_donnees

    '''
    X,Y,t=tracking(video)[0:3]
    dico_coord=extraction_des_donnees(X, Y,taille_pixel)
    return dico_coord
    
    
    
    
#partie du code pas entièrement maîtrisée : interface




# Variables globales
point_selectionne = None
taille_carre = 20  # Taille initiale du carré
image_originale = None
selection_terminee = False
scale_affichage = 1.0 
coordonnees = []

def on_mouse_click(event, x, y, flags, param):
    global point_selectionne
    if event == cv2.EVENT_LBUTTONDOWN:
        orig_x = int(x)  # ✅ conversion
        orig_y = int(y)
        point_selectionne = (orig_x, orig_y)
        afficher_image()

def on_trackbar(val):
    """ Met à jour la taille du carré et rafraîchit l'affichage. """
    global taille_carre
    taille_carre = max(5, val)  # Empêche d'avoir une taille de 0
    afficher_image()

def afficher_image():
    global scale_affichage  # ✅ pour réutiliser le scale calculé
    if image_originale is None:
        return

    image_copy = image_originale.copy()
    
    if point_selectionne:
        x, y = point_selectionne
        cv2.rectangle(image_copy, (x - taille_carre, y - taille_carre),
                      (x + taille_carre, y + taille_carre), (0, 255, 0), 2)
    cv2.namedWindow("Sélection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Sélection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) 
    cv2.imshow("Sélection", image_copy)
 

def selectionner_zone(image):
    global image_originale, taille_carre, point_selectionne, selection_terminee
    image_originale = image.copy()
    selection_terminee = False

    cv2.namedWindow("Sélection",  cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Sélection", on_mouse_click)  # ✅ pas besoin de param ici
    cv2.createTrackbar("Taille", "Sélection", taille_carre, 300, on_trackbar)

    afficher_image()

    while not selection_terminee:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and point_selectionne:
            selection_terminee = True

    cv2.destroyAllWindows()
    return point_selectionne, taille_carre

def choisir_seuil(image_gris):
    """
    Convertit une image couleur en niveaux de gris et permet de choisir un seuil pour binariser l'image.
    
    - image_couleur : Image en couleur (3 canaux)
    - Retourne : L'image binarisée et le seuil choisi
    """


    def on_trackbar(val):
        """ Met à jour l'affichage avec le nouveau seuil. """
        _, image_seuil = cv2.threshold(image_gris, val, 255, cv2.THRESH_BINARY)
        
        # Concaténer les images pour affichage côte à côte
        image_affichee = np.hstack((image_gris, image_seuil))
        cv2.imshow("Comparaison", image_affichee)

    # Création de la fenêtre et de la barre de défilement
    cv2.namedWindow("Comparaison", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Comparaison', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) 
    cv2.createTrackbar("Seuil", "Comparaison", 127, 255, on_trackbar)

    # Affichage initial
    on_trackbar(127)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Touche "Entrée" pour valider
            break

    # Récupération du seuil final sélectionné
    seuil_final = cv2.getTrackbarPos("Seuil", "Comparaison")
    _, image_binaire = cv2.threshold(image_gris, seuil_final, 255, cv2.THRESH_BINARY)

    cv2.destroyAllWindows()
    return image_binaire, seuil_final





#partie sauvegarde des données :
    
    
    
    
    
fichier = "coordonnees.txt"

def sauvegarder_dico(dico, fichier):
    '''
    permet de sauvegarder le dictionnaire avec les données extraites dans un fichier txt

    Parameters
    ----------
    dico : dictionnaire
        le dictionnaire avec les données extraites
    fichier : 'str'
        nom du fichier txt dans lequel on enregistre les données

    Returns
    -------
    None.

    '''
    with open(fichier, 'w') as f:
        for cle, valeurs in dico.items():
            ligne = f"{cle}: {', '.join(map(str, valeurs))}\n"
            f.write(ligne)


def charger_dico(fichier):
    '''
    Permet de charger le dictionnaire enregistré dans le fichier txt

    Parameters
    ----------
    fichier : 'str'
        nom du fichier txt dans lequel on enregistre les données.

    Returns
    -------
    dico : dictionnaire
        le dictionnaire avec les données extraites.

    '''
    dico = {}
    try:
        with open(fichier, 'r') as f:
            for ligne in f:
                cle, valeurs = ligne.strip().split(":")
                dico[int(cle)] = [float(v.strip()) for v in valeurs.split(',') if v.strip()]
    except FileNotFoundError:
        pass  # Si le fichier n'existe pas encore
    return dico



#algorithme utilisé pour tracker les nombreuses particules, il assemble les fonctions d'au dessus et se lance quand on lance le programme (pour l'utiliser il faut enlever les ''' avant et après)
'''
#Calculer nouveau dico en suivnt une particule
nouveau_dico = fonction_finale(video)
#le programme demande à l'utilisateur si il veut sauvegarder les données'
reponse=int(input('Sauvegarde des données ? 1 pour oui autre chose pour non:'))
if reponse==1:
    print('données sauvegardées')
    #Charger l'existant
    dico_existant = charger_dico(fichier)
    #Mettre à jour
    for k, v in nouveau_dico.items():
        if k in dico_existant:
            dico_existant[k] += v
        else:
            dico_existant[k] = v
    #Sauvegarder
    sauvegarder_dico(dico_existant, fichier)
else:
    print('pas sauvegardé')
'''




#partie création des graphiques: 

    


#des paramètres que nous avons utilisé
D1=8*4.32e-13
D=3.921632263462439e-12
T=0.25
fichier2='donnees_r_carre.txt'

def ecart_quadratique_moyen(X1, Y1,t):
    '''
    fonction traçant <r^2> en fonction du temps selon la première méthode statistique et effuctuant une régression linéaire afin d'extraire le coefficient de diffusion

    Parameters
    ----------
    X : liste
        liste des coordonnée de la particule traquée selon x 
    Y : liste
        liste des coordonnée de la particule traquée selon y
    t : liste
        liste regroupant le temps à chaque image


    Returns
    -------
    None.

    '''
    ecart=[]
    X=[X1[i]*3.07e-7 for i in range(len(X1))]
    Y=[Y1[i]*3.07e-7 for i in range(len(Y1))]
    s = 0
    n = len(X)
    x0 , y0 = X[0] , Y[0]
    for i in range(n):
        x1= X[i]
        y1= Y[i]
        s += (x1 - x0) ** 2 + (y1 - y0) ** 2 
        v = s / (i+1)
        ecart.append(v)
    x = np.array(t)  # Valeurs de x
    y = np.array(ecart)  # Valeurs de y

    coeffs = np.polyfit(x, y, 1)  # Ajustement d'un polynôme de degré 1 (y = ax + b)
    slope, intercept = coeffs

    # --- Génération de la droite de régression ---
    y_fit = slope * x + intercept

    # --- Affichage du graphique ---
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, label="Données expérimentales", color="blue")  # Points expérimentaux
    plt.plot(x, y_fit, 'r--', label=f"Régression linéaire (y = {slope}x + {intercept})")  # Ajustement
    plt.xlabel("<r²>")
    plt.ylabel("t")
    plt.title("Régression Linéaire avec NumPy")
    plt.legend()
    plt.grid(True)
    plt.show()
    coef_diffusion=slope/4
    print(f'coefficient de diffusion : {coef_diffusion}')

def transformation_fichier(fichier2):
    '''
    fonction permettant de transformer le fichier contenant le déplacement x entre i image en l'écart quadratique moyen entre i images

    Parameters
    ----------
    fichier2 : fichier que l'on va modifier contenant le déplacement x entre chaque image.

    Returns
    -------
    None.

    '''
    dico = charger_dico(fichier2)
    nouveau_dico=dict()
    #le mouvement étant aléatoire il suffit de prendre deux coordonnées au hasard dans la liste et d'additionner leur carrés pour avoir l'écart quadratique moyen (sans prendre deux fois la même valeur bien sûr)
    for k in dico.keys():
        liste=dico[k]
        nouveau_dico[k]=[]
        assert len(liste)%2==0
        for i in range(len(liste)-1):
            nouveau_dico[k].append(liste[i]**2+liste[i+1]**2)
    sauvegarder_dico(nouveau_dico, fichier2)
    
def coeff_diffusion(fichier2,T):
    '''
    trace <r^2> en fonction du temps effectue une régression linéaire et renvoie le coefficient de diffusion 

    Parameters
    ----------
    fichier2 : str
        fichier contenant l'écart quadratique moyen entre i images
    T : float
        periode entre chaque image

    Returns
    -------
    coef_diffusion : float
        le coefficient de diffusion des particules

    '''
    dico = charger_dico(fichier2)
    R=[]
    t=[]
    #on fait la moyenne sur tout les ecarts quadratiques moyens entre k images et on l'enregistre dans une liste, on enregistre le temps entre les k images dans une autre liste 
    for k in dico.keys():
        t.append(k*T)
        liste=dico[k]
        moyenne=sum(liste)/len(liste)
        R.append(moyenne)
    #on trace <r^2> en fonction du temps
    x = np.array(t)  # Valeurs de x
    y = np.array(R)  # Valeurs de y

    coeffs = np.polyfit(x, y, 1)  # Ajustement d'un polynôme de degré 1 (y = ax + b)
    slope, intercept = coeffs

    # --- Génération de la droite de régression ---
    y_fit = slope * x + intercept

    # --- Affichage du graphique ---
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, label="Données expérimentales", color="blue")  # Points expérimentaux
    plt.plot(x, y_fit, 'r--', label=f"Régression linéaire (y = {round(slope,13)}x + {round(intercept,13)})")  # Ajustement
    plt.xlabel("t (s)")
    plt.ylabel("<r²> (m²)")
    plt.title("Extraction du coefficient de diffusion")
    plt.legend()
    plt.grid(True)
    plt.show()
    #calcul du coeff de diffusion
    coef_diffusion=slope/4
    return coef_diffusion
    
def graphique(fichier, i, D, T):
    '''
    créé le graphique de l'histogramme des valeurs expérimentales superposé à la courbe des valeurs théoriques pour les déplacements de particules entre i images

    Parameters
    ----------
    fichier : 'str'
        nom du fichier txt dans lequel on enregistre les données
    i : int
        expliqué juste au dessus
    D : float
        coefficient de diffusion des particules
    T : float
        periode entre chaque image

    Returns
    -------
    None.

    '''
    #on charge la liste de données que l'on va mettre dans l'histogramme
    dico = charger_dico(fichier)
    data = dico[i]
    

    # Histogramme normalisé
    plt.hist(data, bins=50, density=True, alpha=0.5, label='données expérimentales')

    # Fonction gaussienne
    f = lambda x: (1 /(np.sqrt(4 * np.pi * D * i * T))) * np.exp(-x**2 /((4 * D * i * T)))
    
    #on délimite manuellement la taille du cadre pour qu'il soit fixe et qu'on puisse distinguer l'applatissement de la gaussienne au cours du temps
    x_min = -2.6e-5
    x_max = 2.6e-5
    x_values = np.linspace(x_min, x_max, 400)
    y_values = f(x_values)

    #on créé le graphique avec la légende
    plt.plot(x_values, y_values, 'r-', lw=2, label='courbe théorique')
    plt.ylim(0, 220000)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Densité")
    plt.title(f"Histogramme + Fonction Gaussienne à t={T*i}s")
    plt.show()




                    
            


