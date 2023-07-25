
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input

# Définir les paramètres
num_classes = 3
input_shape = (224, 224, 3)
num_folds = 3
batch_size = 112
epochs = 10

# Charger le modèle MobileNet pré-entraîné (poids sur ImageNet sans la couche de classification)
base_model =MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)



for layer in base_model.layers:
    
    if layer in base_model.layers:
        layer.trainable = False
    else:
        layer.trainable = True
        print(layer.name)
 
    
    
x = base_model.layers[-2].output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)

predictions = Dense(num_classes, activation='softmax')(x)

    

model = Model(inputs=base_model.input, outputs=predictions)


model.summary()
# Geler les poids du modèle M

# Compiler le modèle
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

# Définir le générateur de données avec augmentation des données
data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Préparer le répertoire des données
chemin_image = r'C:\Users\lpari053\OneDrive - University of Ottawa\Desktop\SQUIRELL-GRAY-BLACK3\morph'


validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    


# Créer un objet KFold pour la validation croisée
kfold = KFold(n_splits=num_folds, shuffle=True)

# Itérer sur les plis de la validation croisée
fold = 1
for train_indices, val_indices in kfold.split(range(num_classes)):

    print(f"Training Fold {fold}...")
    print("")

    # Créer les flux de répertoires pour les données d'entraînement et de validation
    train_data = data_generator.flow_from_directory(
        os.path.join(chemin_image,'train'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

    val_data = validation_datagen.flow_from_directory(
        os.path.join(chemin_image,'validation'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
    
    callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='unititled9.keras',           #enregistrement du meilleure modele selon le monitor
        save_best_only=True,
        monitor="val_accuracy")                
    ]

    # Entraîner le modèle sur les données d'entraînement
    model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=callbacks
    )

    fold += 1
    
    
    
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_dataset = test_datagen.flow_from_directory(
os.path.join(chemin_image,'test'),
target_size=(224, 224),
batch_size=batch_size,
class_mode='categorical',
shuffle=True)


best_model=tf.keras.models.load_model('unititled9.keras')


best_model.evaluate(test_dataset)