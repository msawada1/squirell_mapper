var map = L.map('map').setView([45.424721, -75.695000], 13);

L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}).addTo(map);

var point;

var cont_gray=0;
var cont_black=0;
var cont_other=0;

var bounds = map.getBounds();

var sudOuest = bounds.getSouthWest();
var sudEst = bounds.getSouthEast();
var nordOuest = bounds.getNorthWest();
var nordEst = bounds.getNorthEast();

// Créer un objet contenant les coordonnées des coins
var data = {
    sudOuest: {
      latitude: sudOuest.lat,
      longitude: sudOuest.lng
    },
    sudEst: {
      latitude: sudEst.lat,
      longitude: sudEst.lng
    },
    nordOuest: {
      latitude: nordOuest.lat,
      longitude: nordOuest.lng
    },
    nordEst: {
      latitude: nordEst.lat,
      longitude: nordEst.lng
    }
  };



  
  // Convertir l'objet en une chaîne JSON
  var jsonData = JSON.stringify(data);
    
function getMySQLTable() {

    var xhttp = new XMLHttpRequest();
    
    xhttp.onreadystatechange = function() {
      if (this.readyState === 4 && this.status === 200) {
        var response = JSON.parse(this.responseText);

        metttre_point_map(response);
        var nb=document.getElementById('nombre_squirell')
        nb.innerText='Nombre de squirell present :'+response.length
    
    
      }
    };
    
    xhttp.open("GET", "map.php?bounds=" + encodeURIComponent(jsonData), true);
    xhttp.send();

    
  }
  
  getMySQLTable() 



function metttre_point_map(point){


    var cont=0;

    cont_gray=0;
    cont_black=0;
    cont_other=0;


    point.forEach(function(objet) {

        if (cont!=0 ){
        var longitude = objet.longitude;
        var latitude = objet.latitude;
        var imageURL = objet.url;
        

        var morph=objet.morph_class;

        if (morph=='gray'){

          cont_gray=cont_gray+1;


          var marker= L.marker([latitude, longitude],{icon: grayIcon}).addTo(map);



        }


        else if (morph=='black'){

          cont_black=cont_black+1;


          var marker= L.marker([latitude, longitude],{icon: blackIcon}).addTo(map);


          


        }


        
        else if (morph=='other'){

          cont_other=cont_other+1;


          var marker =  L.marker([latitude, longitude],{icon: otherIcon}).addTo(map);


        }

    
    }

    
    cont=cont+1;
    
      })

 charting(cont_black,cont_gray,cont_other)
    }


// Ajouter un gestionnaire d'événements pour tout événement de zoom
map.on('moveend', function(event) {


  boundie();

  jsonData = JSON.stringify(data);



  var xhttp = new XMLHttpRequest();
    
    xhttp.onreadystatechange = function() {
      if (this.readyState === 4 && this.status === 200) {
        var response = JSON.parse(this.responseText);

        metttre_point_map(response);
       
        
        var nb=document.getElementById('nombre_squirell')
        nb.innerText='Nombre de squirell present :'+response.length+ '\n\n Black :'+cont_black+ '\n Gray :'+cont_gray+ '\nOther :'+cont_other
   
      }
    };
    
    xhttp.open("GET", "map.php?bounds=" + encodeURIComponent(jsonData), true);
    xhttp.send();



});


function boundie(){

bounds = map.getBounds();

sudOuest = bounds.getSouthWest();
 sudEst = bounds.getSouthEast();
nordOuest = bounds.getNorthWest();
nordEst = bounds.getNorthEast();

data = {
    sudOuest: {
      latitude: sudOuest.lat,
      longitude: sudOuest.lng
    },
    sudEst: {
      latitude: sudEst.lat,
      longitude: sudEst.lng
    },
    nordOuest: {
      latitude: nordOuest.lat,
      longitude: nordOuest.lng
    },
    nordEst: {
      latitude: nordEst.lat,
      longitude: nordEst.lng
    }
  }


}



var grayIcon = L.icon({
  iconUrl: 'icon/gray.png',  
  iconSize: [25, 41], 
  iconAnchor: [12, 41] 
});


var blackIcon = L.icon({
  iconUrl: 'icon/black.png', 
  iconSize: [25, 41], 
  iconAnchor: [12, 41] 
});


var otherIcon = L.icon({
  iconUrl: 'icon/other.png', 
  iconSize: [25, 41], 
  iconAnchor: [12, 41] 
});



var lat_text;

var lon_text;

var coord_bouton=document.getElementById('coord_bouton');


var pas_lat_long=document.getElementById('pas_lat_long');


function coord_entrer(){


lat_text=document.getElementById('lat_text').value;

lon_text=document.getElementById('lon_text').value;


if (lat_text=='' || lon_text=='' ){


pas_lat_long.innerText='PAS DE LATITUDE OU LONGITUDE ENTREE'

}

else{
  pas_lat_long.innerText=''


  if(estLongitude(lon_text)!=false || estLatitude(lat_text)!=false){

  map.setView([lat_text,lon_text],13)

}

  else{

    pas_lat_long.innerText=' LATITUDE OU LONGITUDE ENTREE NON VALID'

  }

}


}


coord_bouton.addEventListener('click',coord_entrer);

function estLongitude(entrée) {
  // Expression régulière pour vérifier le format de la longitude
  var regexLongitude = /^-?((1[0-7]|[1-9])?\d(\.\d+)?|180(\.0+)?)$/;
  return regexLongitude.test(entrée);
}

function estLatitude(entrée) {
  // Expression régulière pour vérifier le format de la latitude
  var regexLatitude = /^-?(([0-8]?\d)(\.\d+)?|90(\.0+)?)$/;
  return regexLatitude.test(entrée);
}


new_york=document.getElementById('coord_bouton_new_york');
boston=document.getElementById('coord_bouton_boston');
ottawa=document.getElementById('coord_bouton_ottawa');
brevard=document.getElementById('coord_bouton_brevard');


ottawa.addEventListener('click',function(event){

  lat_text='45.424721'

  lon_text='-75.695000'

  map.setView([lat_text,lon_text],13)


})

boston.addEventListener('click',function(event){

  lat_text='42.361145'

  lon_text='-71.057083'

  map.setView([lat_text,lon_text],13)


})

new_york.addEventListener('click',function(event){

  lat_text='40.71291923539871'

  lon_text='-74.00604915688747'

  map.setView([lat_text,lon_text],13)


})


brevard.addEventListener('click',function(event){

  lat_text='35.23345'

  lon_text='-82.73429'

  map.setView([lat_text,lon_text],13)


})






map.on("click", function(event) {
  // Retrieve the latitude and longitude from the event object
  var latitude = event.latlng.lat;
  var longitude = event.latlng.lng;

  // Perform any necessary actions with the latitude and longitude values here
  console.log("Latitude: " + latitude);
  console.log("Longitude: " + longitude);
});



function charting(cont_black,cont_gray,cont_other){


  // Données du diagramme
  var data = {
    labels: ['BLACK', 'GRAY', 'OTHER'],
    datasets: [{
      label: 'Squirell Morph CLass',
      data: [cont_black,cont_gray,cont_other],
      backgroundColor: ['black', 'gray', 'green']
    }]
  };

  // Création du diagramme
  var ctx = document.getElementById('myChart').getContext('2d');
  var myChart = new Chart(ctx, {
    type: 'doughnut',
    data: data,
    options: {
      responsive: true,
    }
  });



}



var ajout_squirell=document.getElementById("coord_ajout")
ajout_squirell=addEventListener('click',coord_ajout)


function coord_ajout(){


  lat_text=document.getElementById('lat_AJOUT').value;
  
  lon_text=document.getElementById('lon_AJOUT').value;
  
  
  if (lat_text=='' || lon_text=='' ){
  
  
  pas_lat_long.innerText='PAS DE LATITUDE OU LONGITUDE ENTREE'
  
  }
  
  else{
    pas_lat_long.innerText=''
  
  
    if(estLongitude(lon_text)!=false || estLatitude(lat_text)!=false){
  
    map.setView([lat_text,lon_text],13)
    var marker =  L.marker([lat_text, lon_text]).addTo(map);

    
  }
  
    else{
  
      pas_lat_long.innerText=' LATITUDE OU LONGITUDE ENTREE NON VALID'
  
    }
  
  }
  
  
  }