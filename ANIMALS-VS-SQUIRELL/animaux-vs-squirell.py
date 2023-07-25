
import tensorflow as tf
import os
import pathlib
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import regularizers
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam,SGD
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16,VGG19
from tensorflow.keras.layers import Dense, Flatten,Dropout,Conv2D,SeparableConv2D,MaxPooling2D,Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import metrics


chemin_image = r'C:\Users\lpari053\JupyterNotebook\squirell_or-not-squirell\SQUIRELL_STAGE\animaux-vs_squirell'

nom_MODEL='animaux_model_vgg19-fine-tune_23_juillet.keras'

image_size = (224, 224)

batch_size = 16

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)


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


base_model = VGG19(weights='imagenet', 
                   include_top=False, 
                   input_shape=(image_size[0],image_size[1], 3))
base_model.trainable=False

for layer in base_model.layers[:-5]:
    layer.trainable = True


# Créer une nouvelle instance de modèle avec les couches tronquées
#truncated_model = keras.models.Model(inputs=model.input, outputs=model.output)

x = base_model.output

x=Flatten()(x)

x = Dense(128,activation='relu',name='d1')(x)

#x=Dropout(0.5)(x)

x = Dense(96,activation='relu',name='d2')(x)

x = Dense(64,activation='relu',name='d3')(x)

#x=Dropout(0.5)(x)

x = Dense(32,activation='relu',name='d4')(x)

predictions = Dense(1,activation='sigmoid',name='fin')(x)


model = Model(inputs=base_model.input, outputs=predictions)  #mise en place finale du modele


model.summary()

learning_rate = 0.01                                    #learning rate de l'optimizer 

early_stopping = EarlyStopping(monitor='val_accuracy', patience=20)   
#arrete l'entrainement lorsque le monitor n'evalue plus apres patience epoch


model.compile(loss="binary_crossentropy",       #binary car seulelemtn deux classe finale
              optimizer=SGD(learning_rate=learning_rate),        #definition optimizer avec son learning rate specifique
              metrics=["accuracy"])             #evalutation du modele avec accuracy nombre valeur juste predite

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=nom_MODEL,           #enregistrement du meilleure modele selon le monitor
        save_best_only=True,
        monitor="val_accuracy")                
]



history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=validation_dataset,
    callbacks=callbacks)



def affichage(history):


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
    plt.savefig('console1_1.jpg')
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()


def test9(test_dataset,nom_MODEL):
    modelie = keras.models.load_model(nom_MODEL)
    test_loss, test_acc = modelie.evaluate(test_dataset) 
    print(f"Test accuracy: {test_acc:.3f}")


