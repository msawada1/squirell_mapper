#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import pathlib
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import regularizers
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten,Dropout,Conv2D,SeparableConv2D,MaxPooling2D,Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics


# In[2]:

###########################################################################################################
chemin_image ='C:/Users\lpari053\JupyterNotebook\squirell_or-not-squirell\SQUIRELL_STAGE\catdog_squirell'
##########################################################################################################



nom_MODEL='squirell_street_cat_dog.keras'


# In[3]:


image_size = (224, 224)

batch_size = 52

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)


# In[4]:


#JEU D'ENTRAINEMENT 
train_dataset = image_generator.flow_from_directory(
    os.path.join(chemin_image, 'train'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

#JEU DE VALIDATION
validation_dataset = image_dataset_from_directory(
    os.path.join(chemin_image, 'validation'),
    labels='inferred',
    label_mode='binary',
    image_size=image_size,
    batch_size=batch_size
)

#JEU DE TEST
test_dataset = image_dataset_from_directory(
    os.path.join(chemin_image, 'test'),
    labels='inferred',
    label_mode='binary',
    image_size=image_size,
    batch_size=batch_size
)


# In[5]:


model = VGG16(weights='imagenet', 
                   include_top=False, 
                   input_shape=(image_size[0],image_size[1], 3))

for layer in model.layers:
    layer.trainable = False


x = model.output

x=Flatten()(x)

x = Dense(128,activation='relu',name='dense')(x)

x = Dense(128,activation='relu',name='d1')(x)

x = Dense(64,activation='relu',name='d2')(x)

x = Dense(32,activation='relu',name='d3')(x)

predictions = Dense(1,activation='sigmoid',name='fin')(x)


model = Model(inputs=model.input, outputs=predictions)  #mise en place finale du modele


# In[6]:


model.summary()


# In[7]:


learning_rate = 0.01                                      #learning rate de l'optimizer 

early_stopping = EarlyStopping(monitor='val_loss', patience=20)   
#arrete l'entrainement lorsque le monitor n'evalue plus apres patience epoch


# In[8]:


model.compile(loss="binary_crossentropy",       #binary car seulelemtn deux classe finale
              optimizer=Adam(learning_rate),        #definition optimizer avec son learning rate specifique
              metrics=["accuracy"])             #evalutation du modele avec accuracy nombre valeur juste predite

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=nom_MODEL,           #enregistrement du meilleure modele selon le monitor
        save_best_only=True,
        monitor="val_loss")                
]


# In[9]:


history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=callbacks)


# In[10]:


import matplotlib.pyplot as plt
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()


# ### Apres les 50 epochs

# In[11]:


test_acc = model.evaluate(test_dataset)[1]
print(f"Test accuracy: {test_acc:.3f}")


# #### Meilleure model 

# In[13]:


modelie = keras.models.load_model(nom_MODEL)
test_acc = modelie.evaluate(test_dataset)[1]
print(f"Test accuracy: {test_acc:.3f}")


# In[27]:


from PIL import Image
import numpy as np
# Chemin du dossier contenant les images à prédire
folder_path='C:/Users\lpari053\JupyterNotebook\squirell_or-not-squirell\SQUIRELL_STAGE/prediction'

# Obtention de la liste des chemins complets des images dans le dossier
image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.jpg')]


# Prédiction des images
predictions = []
for path in image_paths:
    # Charger l'image
    image = Image.open(path)

    # Prétraitement de l'image
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Effectuer la prédiction
    prediction = modelie.predict(image)
    predictions.append(prediction)


# In[58]:


import matplotlib.pyplot as plt
import os

chemin_destination='C:/Users\lpari053\JupyterNotebook\squirell_or-not-squirell\SQUIRELL_STAGE/prediction_faites'
if not os.path.exists(chemin_destination):
    os.makedirs(chemin_destination)


# Chemin vers le dossier contenant les images
chemin_images = 'C:/Users\lpari053\JupyterNotebook\squirell_or-not-squirell\SQUIRELL_STAGE/prediction'

# Liste des noms de fichiers des images
noms_images = [os.path.join(folder_path, filename) for filename in os.listdir(chemin_images) if filename.endswith('.jpg')]

# Calcul du nombre de lignes et de colonnes nécessaires
nb_images_par_ligne = 5
nb_images_total = len(noms_images)
nb_lignes = (nb_images_total - 1) // nb_images_par_ligne + 1
nb_colonnes = min(nb_images_total, nb_images_par_ligne)

# Configuration de la taille de la figure
fig, axes = plt.subplots(nb_lignes, nb_colonnes, figsize=(16,12))

# Parcourir les noms des fichiers d'images
for i, nom_image in enumerate(noms_images):
    # Chemin complet de l'image
    chemin_image = os.path.join(chemin_images, nom_image)
    
    # Charger et afficher l'image
    image = plt.imread(chemin_image)
    
    # Sélectionner le sous-graphique approprié
    if nb_lignes == 1:
        ax = axes[i % nb_colonnes]
    else:
        ax = axes[i // nb_colonnes, i % nb_colonnes]
    
    ax.imshow(image)
    ax.axis('off')
    
    if predictions[i]<0.5:
        a='OTHER'
        ax.set_title('OTHER')
    else:
        a='SQUIRELL'
        ax.set_title('SQUIRELL')
    
    label = a
    plt.tight_layout()
    nom_fichier = f"image_complet_{i+1}_{label}.jpg"
    
    if os.path.exists(chemin_destination_image):
        os.remove(chemin_destination_image)
    
    chemin_destination_image = os.path.join(chemin_destination, nom_fichier)
    plt.savefig(chemin_destination_image)
        
    

# Supprimer les sous-graphiques vides
for j in range(len(noms_images), nb_lignes * nb_colonnes):
    if nb_lignes == 1:
        axes[j].axis('off')
    else:
        axes[j // nb_colonnes, j % nb_colonnes].axis('off')
        

# Afficher la figure contenant les images
plt.tight_layout()
plt.show()


# In[63]:


import matplotlib.pyplot as plt
import os

# Chemin vers le dossier contenant les images
chemin_images = 'C:/Users/lpari053/JupyterNotebook/squirell_or-not-squirell/SQUIRELL_STAGE/prediction'

# Liste des noms de fichiers des images
noms_images = [filename for filename in os.listdir(chemin_images) if filename.endswith('.jpg')]

# Configuration de la taille de la figure
fig, axes = plt.subplots(len(noms_images), 1, figsize=(10, 5*len(noms_images)))

# Parcourir les noms des fichiers d'images
for i, nom_image in enumerate(noms_images):
    # Chemin complet de l'image
    chemin_image = os.path.join(chemin_images, nom_image)
    
    # Charger et afficher l'image
    image = plt.imread(chemin_image)
    
    # Afficher l'image sur le sous-graphique correspondant
    ax = axes[i]
    ax.imshow(image)
    ax.axis('off')
    
    if predictions[i] < 0.5:
        a = 'OTHER'
        ax.set_title('OTHER')
    else:
        a = 'SQUIRELL'
        ax.set_title('SQUIRELL')
        
    label = a
    plt.tight_layout()
    nom_fichier = f"image_{i+1}_{label}.jpg"
    
    if os.path.exists(chemin_destination_image):
        os.remove(chemin_destination_image)
    
    chemin_destination_image = os.path.join(chemin_destination, nom_fichier)
    plt.savefig(chemin_destination_image)
        

# Ajuster la taille des sous-graphiques pour agrandir les images
for ax in axes.flat:
    ax.margins(0.2)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

# Afficher la figure contenant les images
plt.tight_layout()
plt.show()


# In[ ]:




