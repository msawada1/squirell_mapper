var data = {objet:'object'};

var lati;
var longi;
var cityselect;
var ville_option;



  var jsonData = JSON.stringify(data);
    
function getMySQLTable() {
    var xhttp = new XMLHttpRequest();
    
    xhttp.onreadystatechange = function() {
      if (this.readyState === 4 && this.status === 200) {
        var response = JSON.parse(this.responseText);


        var selectElement = document.getElementById('villeSelect');


        ville_option=response;


        response.forEach(function(ville) {
            var option = document.createElement('option');
            option.value = ville.city;
            option.textContent = ville.city;
            selectElement.appendChild(option);
            
          }


      )}
    };
    
    xhttp.open("GET", "cities.php", true);
    xhttp.send();

    
  }

  getMySQLTable() 


 var city_bouton= document.getElementById('bouton_city')

city_bouton.addEventListener('click',function(event){

  var selectElement = document.getElementById('villeSelect');

// Utilisation de la fonction pour chercher une ville spécifique
var villeRecherchee = chercherVille(selectElement.value);

if (villeRecherchee) {
  console.log('Ville trouvée :', villeRecherchee);
  map.setView([villeRecherchee.lat,villeRecherchee.lng])

} else {
  console.log('Ville non trouvée');
}

}
)




// Fonction de recherche de la ville dans le tableau JSON
function chercherVille(nomVille) {
  var villeTrouvee = ville_option.find(function(ville) {
    return ville.city === nomVille;
  });

  return villeTrouvee;
}







