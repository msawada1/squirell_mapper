#!/usr/bin/env python
# coding: utf-8

# # Modele qui classifie entre SQUIREEL et STREET

# Importation des librairie et package utilise

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
from tensorflow.keras.layers import Dense, Flatten,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping


# Les images se trouvent dans un dossier qui contient 3 sous dossier : test validation et train qui eux meme chacun a deux dossier SQUIRELL et STREET

# In[2]:

################################################################################################################
chemin_image ='C:/Users\lpari053\JupyterNotebook\squirell_or-not-squirell\SQUIRELL_STAGE\squirell_vs_not_squirell'
################################################################################################################




nom_MODEL='model_squirell_street.keras'


# Definition de la normalisation des images (image size et rescale)pour les normaliser et de la taille des batch=echantillions

# In[3]:


image_size = (224, 224)

batch_size = 54 

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)


# ## Importation dataset

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


# Utilisation du modele pre entrainer VGG16 sans fine tune = donc avec des weigth pas entrainer

# In[4]:


base_model = VGG16(weights='imagenet', 
                   include_top=False, 
                   input_shape=(image_size[0],image_size[1], 3))

for layer in base_model.layers:
    layer.trainable = False


# Defintion d'un neural network qui vq diversifier nos images permettant invariance a la rotation et translation

# In[5]:


data_augmentation = keras.Sequential(
 [
 layers.RandomFlip("horizontal"),
 layers.RandomRotation(0.1),
 layers.RandomZoom(0.2),
 ]
)


# ### Definition du modele

# In[6]:


x = base_model.output     #recuperation de la sortie de couhe du modele pre entrainer


x = data_augmentation(x)  #data augmentaion sur nos donnees pour invariance

x = Flatten()(x)       #transforme la sortie de couche qui sont en 2D pour les mettre en vecteur 1D

x = Dense(128, activation='relu')(x)      
x = Dense(64, activation='relu')(x)

predictions = Dense(1, activation='sigmoid')(x)  #dernier couche de neural network dense qui utilise activation sigmoid 
                                 #donne en sortie en float entre 0 et 1 ainsi si plus proche de 0
                         #nous somme dans la classe ici SQUIRELL etplus on est proche de 1 classe STREET


# In[7]:


model = Model(inputs=base_model.input, outputs=predictions)  #mise en place finale du modele


# In[8]:


model.summary()


# In[9]:


learning_rate = 0.05                                         #learning rate de l'optimizer 

early_stopping = EarlyStopping(monitor='val_loss', patience=35)   
#arrete l'entrainement lorsque le monitor n'evalue plus apres patience epoch


# In[10]:


model.compile(loss="binary_crossentropy",       #binary car seulelemtn deux classe finale
              optimizer=Adam(learning_rate),        #definition optimizer avec son learning rate specifique
              metrics=["accuracy"])             #evalutation du modele avec accuracy nombre valeur juste predite

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=nom_MODEL,           #enregistrement du meilleure modele selon le monitor
        save_best_only=True,
        monitor="val_loss"),
    early_stopping                    
]


# In[11]:


history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=callbacks)


# In[12]:


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


# In[13]:


modelie = keras.models.load_model(nom_MODEL)
test_loss, test_acc = modelie.evaluate(test_dataset) 
print(f"Test accuracy: {test_acc:.3f}")


# In[14]:


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

# Afficher les images et les prédictions
fig, axs = plt.subplots(len(image_paths),1, figsize=(30,30))
for i, path in enumerate(image_paths):
    # Charger l'image
    image = Image.open(path)

    # Afficher l'image
    axs[i].imshow(image)
    axs[i].axis('off')

    # Interpréter la prédiction
    if predictions[i] > 0.5:
        prediction_label = 'STREET'
    else:
        prediction_label = 'SQUIRELL'

    # Afficher la prédiction
    axs[i].set_title(f'Prédiction : {prediction_label}')

plt.show()

