import cv2                                            # Importation de la bibliothèque OpenCV pour la capture vidéo et la détection d'objets
import numpy as np                                    # Importation de la bibliothèque NumPy pour la gestion des tableaux
from keras.models import load_model                   # Importation de la fonction pour charger un modèle pré-entraîné
from keras.preprocessing.image import img_to_array    # Importation de la fonction pour convertir une image en tableau
import pygame                                         # Importation de Pygame pour jouer des sons
from threading import Thread                          # Importation de la classe Thread pour exécuter des tâches simultanées
import time
from datetime import datetime

# Chargement des modèles et des cascades de détection
classes = ['Closed', 'Open']                                                        # Définir les classes d'yeux : fermés et ouverts
face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")    # Charger la cascade pour la détection des visages
left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")    # Charger la cascade pour l'œil gauche
right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")  # Charger la cascade pour l'œil droit
mouth_cascade = cv2.CascadeClassifier("data/haarcascade_mcs_mouth.xml")             # Charger la cascade pour la détection de la bouche
 
cap = cv2.VideoCapture(0)                # Ouvrir la caméra pour capturer des images en temps réel 
model = load_model("ds_project.h5")      # Charger le modèle de détection de somnolence pré-entraîné
count = 0                                # Compteur pour compter les frames avec les yeux fermés
alarm_on = False                         # Indicateur pour savoir si l'alarme est activée
alarm_sound = "data/wake_up_alarm.mp3"   # Fichier sonore de l'alarme
msg_sound = "data/msg.mp3"               # Fichier sonore pour le bâillement
wel_sound="data/welcome.mp3"             # Fichier sonore welcome 
status1 = ''                             # Statut de l'œil gauche (fermé ou ouvert)
status2 = ''                             # Statut de l'œil droit (fermé ou ouvert)
welcome_played = False
yawn_count = 0                           # Compteur de bâillements détectés     
yawn_alerted = False                     # Indicateur pour éviter des alertes répétées pour le bâillement

def start_alarm(sound, play_on_loop=True):
    pygame.mixer.init()
    pygame.mixer.music.load(sound)
    if play_on_loop:
        pygame.mixer.music.play(-1)  
    else:
        pygame.mixer.music.play()        # Lancer la lecture du son

def stop_alarm():
    pygame.mixer.music.stop()
    time.sleep(2)
    play_msg_once(msg_sound,"Il est conseillé de prendre une pause")

def play_msg_once(sound,msg):
    pygame.mixer.init()
    pygame.mixer.music.load(sound)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        cv2.putText(frame, f"{msg}", (7, height-30), cv2.FONT_ITALIC, 0.4, (255, 255, 255), 1)
        continue


while True:
    ab, frame = cap.read()                              # Capturer une image depuis la caméra
    height, width, ab = frame.shape                      # Récupérer la hauteur et la largeur de l'image

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    
    cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f"{current_time}", (400, 20), cv2.FONT_ITALIC, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, "Systeme de detection de somnolence", (10, 20), cv2.FONT_ITALIC, 0.6, (255, 255, 255), 1)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        right_eye = right_eye_cascade.detectMultiScale(roi_gray)

        if not welcome_played:
            time.sleep(1)
            txt = "Salut ! Je suis votre compagnon de route et je veillerai sur vous pendant que vous conduisez."
            t = Thread(target=play_msg_once, args=(wel_sound, txt))
            t.daemon = True
            welcome_played = True   
            t.start()
             

        for (x1, y1, w1, h1) in left_eye:
            cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
            eye1 = roi_color[y1:y1+h1, x1:x1+w1]
            eye1 = cv2.resize(eye1, (145, 145))
            eye1 = eye1.astype('float') / 255.0
            eye1 = img_to_array(eye1)
            eye1 = np.expand_dims(eye1, axis=0)
            pred1 = model.predict(eye1)
            status1=np.argmax(pred1)
            break

        for (x2, y2, w2, h2) in right_eye:
            cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
            eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
            eye2 = cv2.resize(eye2, (145, 145))
            eye2 = eye2.astype('float') / 255.0
            eye2 = img_to_array(eye2)
            eye2 = np.expand_dims(eye2, axis=0)
            pred2 = model.predict(eye2)
            status2=np.argmax(pred2)
            break
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)                    # Détecte la bouche dans la ROI en niveaux de gris
        for (mx, my, mw, mh) in mouth:                                               # Si une bouche est détectée
            yawn_ratio = mh / mw                                                     # Calcule le ratio hauteur/largeur de la bouche
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 1)   # Dessine un rectangle autour de la bouche
            if yawn_ratio > 0.6:                                                     # Si la bouche est ouverte au-delà d'un seuil (bâillement)
                yawn_count += 1                                                      # Incrémente le compteur de bâillements
                cv2.putText(frame, "Detection de baillement!", (10, height - 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)    # Affiche un message de bâillement
                if not yawn_alerted:       # Si l'alerte de bâillement n'a pas encore été donnée
                    yawn_alerted = True    # Lance un thread pour jouer le son du bâillement
                    t = Thread(target=start_alarm, args=(msg_sound))  
                    t.daemon = True
                    t.start()
                break
            else:
                yawn_count = 0         # Réinitialise le compteur si aucun bâillement n'est détecté
                yawn_alerted = False   # Réinitialise l'alerte de bâillement

            # Si la bouche est ouverte, afficher un message
            if mh < 40:    # Si la hauteur de la bouche est supérieure à 40 pixels, cela indique que la bouche est ouverte
                # Affiche "Bouche ouverte" sur l'image à la position (10, height - 100), en rouge (0, 0, 255), avec une taille de police de 1 et une épaisseur de 2
                cv2.putText(frame, "Bouche fermer", (10, height - 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)  

        
        if status1 == 2 and status2 == 2:
            count += 1
            cv2.putText(frame, "Yeux fermes, Nb: " + str(count), (10, 48), cv2.FONT_ITALIC, 0.9, (0, 0, 255), 2)
            if count >= 4:
                cv2.putText(frame, "Avertissement : Somnolence detectee!", (80, height-20), cv2.FONT_ITALIC, 0.5, (0, 0, 255), 2)
                if not alarm_on:
                    alarm_on = True
                    t = Thread(target=start_alarm, args=(alarm_sound, True))
                    t.daemon = True
                    t.start()
        else:
            cv2.putText(frame, "Yeux ouverts", (10, 48), cv2.FONT_ITALIC, 0.9, (0, 255, 0), 2)
            count = 0
            if alarm_on:
                alarm_on = False

                t = Thread(target=stop_alarm)
                t.daemon = True
                t.start()
    cv2.imshow("projet fournit par Rihem , Amal , montahe & Islem", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
