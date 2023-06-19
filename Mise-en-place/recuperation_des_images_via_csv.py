import pandas as pd
import requests
import os

### Recuperation et telechargement des images a partir d'url se trouuvant dans un CSV

##CSV :   Le CSV doit contenir des URL qui amene a une image 

#Egalement il est possible de recuperer d'autre information contenu dans le CSV ici on utiliise :
    
    #   inat_id est l'identifiant pour retrouver l'origine de l'image sur INaturaliste
    #   morph_class est la couleur identifie par des professionels de l'ecureuil
    #   latitude et longitude sont les coordonnees WGS84 de ou a ete pris la photo
    
    
#Prealablement creer un dossier qui va contenir les images



# Chemin vers le fichier CSV et vers le doosier de destination des images
chemin_fichier = "database_squirel.csv"
chemin_destination = "image/"



#Creation d'un dataframe pandas contenant le CSV complet
dataframe = pd.read_csv(chemin_fichier)
    

def creation_dossier_image(chemin_fichier,chemin_destination,dataframe=dataframe):

    
    url_liste = dataframe['url'].tolist()                   #Recuperation de tout les URL en liste
    inat_id_liste = dataframe['inat_id'].tolist()           #Recuperation des id unique par image
    morph_class_liste=dataframe['morph_class'].tolist()     #Recuperation de la couleur de l'ecureuil
    lat_class_liste=dataframe['latitude'].tolist()          #Recuperation des coordonnee longitude et latitude
    long_class_liste=dataframe['longitude'].tolist()
    

    #On parcour la liste contenant les informations pour chaque ecureuil ainsi un index represente un ecureuil
    for index in range(0,len(url_liste)):   
        
        url=url_liste[index]                   #URL de l'image
        
        inat_id=inat_id_liste[index]           #id naturaliste de l'image
        
        morph_class=morph_class_liste[index]   #Couleur de l'ecureuil sur l'image
        
        latitude=lat_class_liste[index]        #Latitude de la position ou a ete pris l'image
        
        longitude=long_class_liste[index]      # Longitude de la position ou a ete pris l'image
        
        
    
        # Requete pour recuperer l'image
        response = requests.get(url)
        
        # Reponse de si la requete a reussi si est egale a 200 requete reussi
        if response.status_code == 200:
            
            #etablissement du nom de l'image avec ces informations personnelles
            nom_image = f"image_{inat_id}_{morph_class}_lat_{latitude}_long_{longitude}.jpg"
            
            #Chemin de destination comportant le chemin du dossier et le nom final de l'image dans le dossier
            chemin_destination2 = os.path.join(chemin_destination, nom_image)
            
            
            #Enregistrement de l'image dans le dossier voulu avec son nouveau nom
            with open(chemin_destination2, 'wb') as f:
                f.write(response.content)
            print("L'image a été téléchargée et sauvegardée avec succès.  ",index)
        else:
            print("La requête pour télécharger l'image a échoué.   ",index)