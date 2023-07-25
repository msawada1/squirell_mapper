import sys
import numpy as np
from PIL import Image
import keras
import json

import heatmap

import os

# Récupérer le chemin absolu du fichier actuel
chemin_absolu = os.path.abspath(__file__)

# Récupérer le répertoire parent du fichier actuel
rep_parent = os.path.dirname(chemin_absolu)


img_path= os.path.join(rep_parent,'image\image.jpg')


input_file_path = os.path.join(rep_parent,'python_input.json')
output_file_path = os.path.join(rep_parent,'python_output.txt')


# # Chargement des données depuis le fichier d'entrée JSON
with open(input_file_path, 'r') as input_file:
    data = json.load(input_file)

objet = data['objet']

heatmap.tout


# Ouverture de l'image
image = Image.open(img_path)

# Obtention de la taille de l'image
image_size = image.size

# Affichage de la taille de l'image
print("Taille de l'image :", image_size)


# Prétraitement de l'image
image = image.resize((224, 224))
image = np.array(image)
image = image / 255.0
image = np.expand_dims(image, axis=0)

nom_MODEL = r"C:\Users\lpari053\OneDrive - University of Ottawa\Desktop\flickr_squirell\denseflickrsucces098.keras"

model = keras.models.load_model(nom_MODEL)

prediction = model.predict(image)
print(prediction)


# # Enregistrement de la prédiction dans un fichier de sortie
with open(output_file_path, 'w') as output_file:
    #output_file.write(str(sys.argv))
    if prediction>0.5:
        output_file.write('Le modele a predit que il y a un SQUIRELL dans la photo')
    else:
        output_file.write("Le modele a predit que il n'y a PAS SQUIRELL dans la photo")
        

# Débogage - Afficher les valeurs et les étapes intermédiaires

print("Image path :", img_path)
print("Input file path :", input_file_path)
print("Output file path :", output_file_path)
print("Prediction :", prediction)
