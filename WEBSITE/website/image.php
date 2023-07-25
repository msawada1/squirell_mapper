<?php
if ($_FILES['image']['error'] === UPLOAD_ERR_OK) {
  $nomTemporaire = $_FILES['image']['tmp_name'];
  $destination = 'image/image.jpg' ;

  if (move_uploaded_file($nomTemporaire, $destination)) {
    echo 'Image téléchargée avec succès.';

  
  } else {
    echo 'Une erreur s\'est produite lors du téléchargement de l\'image.';
  }
}
?>
