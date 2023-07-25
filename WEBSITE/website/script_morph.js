
var model;
var img;
var texte;
var formData
var imagePath='image/image.jpg'



///////////////////////////////////////////////

var inter_python="C:\ Users\lpari053\AppData\Local\anaconda3\python.exe";

var bouton_texte= document.getElementById('texte_bouton')

bouton_texte.addEventListener('click',interpeter_python)

///////////////////////////////////////////////////
function interpeter_python(){
  var texte = document.getElementById("monTexte").value;
  inter_python=texte
  console.log(inter_python);

}


//////////////////////////////////////////

var bouton_1=document.getElementById('bouton_morph')
bouton_1.addEventListener('click',sendRequest)


// Fonction pour envoyer une requête AJAX au script PHP
function sendRequest() {

  console.log('Chargement de la couleur')

  $('#loading-indicator').show();

  
    const xhr = new XMLHttpRequest();
  
    xhr.onreadystatechange = function () {
      if (xhr.readyState === XMLHttpRequest.DONE) {
        if (xhr.status === 200) {
          const response = xhr.responseText;
          showResult(response);
        } else {
          console.error('Erreur lors de la requête : ' + xhr.status);
        }
        $('#loading-indicator').hide();
      }
    };
  
    const phpScriptPath = 'script_morph.php';
    const imageFilePath = imagePath;
  
    const requestData = {
      objet: 'objectt',
      imagePath: imageFilePath,
      python_inter : inter_python
    };
  
    const jsonString = JSON.stringify(requestData);
  
    xhr.open('POST', phpScriptPath, true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(jsonString);
  }
  
  // Afficher le résultat dans un élément HTML avec l'id "result"
  function showResult(result) {
    const resultElement = document.getElementById('result_morph');
    resultElement.textContent = result;
    console.log(result)



  }
  
  


  