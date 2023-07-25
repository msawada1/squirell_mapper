import sys
import numpy as np
from PIL import Image
import keras
import json
from tensorflow.keras.preprocessing import image
from keras.applications.densenet import preprocess_input
import heatmap

import os

# Récupérer le chemin absolu du fichier actuel
chemin_absolu = os.path.abspath(__file__)

# Récupérer le répertoire parent du fichier actuel
rep_parent = os.path.dirname(chemin_absolu)


img_path= os.path.join(rep_parent,'image\image.jpg')


input_file_path = os.path.join(rep_parent,'python_input_morph.json')
output_file_path = os.path.join(rep_parent,'python_output_morph.txt')


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
image = np.expand_dims(image, axis=0)
images = preprocess_input(image)

model = keras.models.load_model(r"C:\Users\lpari053\OneDrive - University of Ottawa\Desktop\SQUIRELL-GRAY-BLACK3\python2\val9813.keras")


prediction_morph = model.predict(images)

print(prediction_morph[0])

if prediction_morph[0][0]==max(prediction_morph[0]):
    
    morph='Black'
    
if prediction_morph[0][1]==max(prediction_morph[0]):
    
    morph='Gray'
    
if prediction_morph[0][2]==max(prediction_morph[0]):
    
    morph='Other'

# # Enregistrement de la prédiction dans un fichier de sortie
with open(output_file_path, 'w') as output_file:
    output_file.write(f"L'image contient un {morph} Squirell")

