
###Ce programme sert a deplacer des fichiers d'un dossier vers un autre dossier

import os
import shutil



dossier_source = r'C:\Users\lpari053\Downloads\archive\train\train_sans_squirell'


dossier_cible = r"C:\Users\lpari053\JupyterNotebook\squirell_or-not-squirell\SQUIRELL_STAGE\animaux-vs_squirell - Copy - Copy\train\OTHER"



for nom_fichier in os.listdir(f'{dossier_source}'):
    
   
    
    chemin_fichier_source = os.path.join(f'{dossier_source}', nom_fichier)
    
 
    
    chemin_fichier_cible = os.path.join(dossier_cible,nom_fichier)
    

    if os.path.isfile(chemin_fichier_source):
        
        
        # Copier le fichier
        shutil.copy(chemin_fichier_source, chemin_fichier_cible)
            
