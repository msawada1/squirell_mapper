import sys
from PIL import Image

image_path = sys.argv[1]
image = Image.open(image_path)

# Exemple d'opération : afficher la taille de l'image
largeur, hauteur = image.size
print(f"La taille de l'image est : {largeur} x {hauteur}")