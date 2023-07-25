
var model;
var img;
var texte;
var formData
var imagePath='image/image.jpg'

// document.getElementById('choisirImage').addEventListener('change', function(e) {
//   var reader = new FileReader();
//   reader.onload = function(e) {
//     document.getElementById('imageAffichee').setAttribute('src', e.target.result);
//     console.log("Chemin de l'image :", e.target.result);
//   }
//   reader.readAsDataURL(e.target.files[0]);
// });



// function form_data(){
// var input = document.getElementById("imageAffichee");
// console.log(input)
// // var file = input.files[0];
// // var formData = new FormData();
// // formData.append("image", file);


// }


///////////////////////////////////////////////

var inter_python="C:\ Users\lpari053\AppData\Local\anaconda3\python.exe";

var bouton_texte= document.getElementById('texte_bouton')

bouton_texte.addEventListener('click',interpeter_python)

///////////////////////////////////////////////////
function interpeter_python(){
  var texte = document.getElementById("monTexte").value;
  inter_python=texte
  console.log(inter_python);

// imagePath=texte;
}


//////////////////////////////////////////

var bouton_1=document.getElementById('bouton1')
bouton1.addEventListener('click',sendRequest)

bouton1.addEventListener('click',function(event){

b_heatmpap=document.getElementById('bouton_HEATMAP')
b_heatmpap.style.display='none'

heatmpap=document.getElementById('heatmap')
heatmpap.src='#'

})

var bouton_envoyer=document.getElementById('envoyer')

bouton_envoyer.addEventListener('click',function(event){

b_heatmpap=document.getElementById('bouton_HEATMAP')
b_heatmpap.style.display='none'

heatmpap=document.getElementById('heatmap')
heatmpap.src='#'

})



// Fonction pour envoyer une requête AJAX au script PHP
function sendRequest() {

  console.log('execute-python')

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
  
    const phpScriptPath = 'script.php';
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
    const resultElement = document.getElementById('result');
    resultElement.textContent = result;
    console.log(result);

    if (result=='Le modele a predit que il y a un SQUIRELL dans la photo'){
   
      var ajout_squirell=document.getElementById("ajout_squirell");
      ajout_squirell.style.display='block';

      var bouton_morph=document.getElementById("bouton_morph");
      bouton_morph.style.display='block';


    }

    b_heatmpap=document.getElementById('bouton_HEATMAP')
    b_heatmpap.style.display='block'

    b_heatmpap.addEventListener('click',function(event){

      var image = document.getElementById('heatmap');
      image.src = 'image/cam2.jpg';


    })


  }
  
  // Appel de la fonction pour envoyer la requête
  //sendRequest();
  


  