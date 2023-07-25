import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16,VGG19
from tensorflow.keras.optimizers import Adam,SGD

image_width, image_height = 224, 224

chemin_image=r'C:\Users\lpari053\OneDrive - University of Ottawa\Desktop\flickr_squirell'
nom_MODEL='flirckr11.keras'
# Définir les hyperparamètres du modèle
batch_size = 512
epochs = 50



def dataset(train_dir,validation_dir,test_dir):
    
    # Prétraitement des données
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary')
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary')
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='binary')
    
    return(train_generator,validation_generator,test_generator
           )


def modelie(new_learning_rate=0.01):

    data_augmentation = keras.Sequential(
     [
      keras.layers.Input((image_width, image_height,3)),
     keras.layers.RandomFlip("horizontal"),
     keras.layers.RandomRotation(0.1),
     keras.layers.RandomZoom(0.2),
     ]
    )
    
    model = keras.models.Sequential([
        
        keras.layers.Input((image_width, image_height,3)),
        
        data_augmentation,
        
        keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same',input_shape=(image_width, image_height, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.Conv2D(16, (3, 3), padding='same',activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.Conv2D(128, (3, 3), padding='same',activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        
        
        keras.layers.Flatten(),
        
        keras.layers.Dense(256, activation='relu'),

        keras.layers.Dense(128, activation='relu'),
    
        keras.layers.Dense(64, activation='relu'),
        
        keras.layers.Dense(32, activation='relu'),
    
        
        keras.layers.Dense(16, activation='relu'),
        
        keras.layers.Dense(8, activation='relu'),
        
        keras.layers.Dense(1, activation='sigmoid')
        
        
        
    ])
    
    
    
    
    
    
    # Compiler le modèle
    model.compile(optimizer="adam",
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    return(model)





def training():
    
    model=modelie()
    
    train_generator,validation_generator,test_dataset=dataset( 
        os.path.join(chemin_image,'train'),
        os.path.join(chemin_image,'validation'),
        os.path.join(chemin_image,'test')
        )
        
        
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=nom_MODEL,           #enregistrement du meilleure modele selon le monitor
            save_best_only=True,
            monitor="val_accuracy")                
    ]

    # Entraîner le modèle
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // batch_size,
        callbacks=callbacks)
    
    test_model(model,test_dataset,nom_MODEL=nom_MODEL)
    
    
    
        


def test_model(model,test_dataset,nom_MODEL=nom_MODEL):
    
    model.evaluate(test_dataset)
    
    best_model=keras.models.load_model(nom_MODEL)

    
    best_model.evaluate(test_dataset)






