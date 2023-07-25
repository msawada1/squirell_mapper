

<?php


$bounds = json_decode($_GET['bounds'], true);

// Accéder aux coordonnées des coins
$sudOuestLatitude = $bounds['sudOuest']['latitude'];
$sudOuestLongitude = $bounds['sudOuest']['longitude'];

$sudEstLatitude = $bounds['sudEst']['latitude'];
$sudEstLongitude = $bounds['sudEst']['longitude'];

$nordOuestLatitude = $bounds['nordOuest']['latitude'];
$nordOuestLongitude = $bounds['nordOuest']['longitude'];

$nordEstLatitude = $bounds['nordEst']['latitude'];
$nordEstLongitude = $bounds['nordEst']['longitude'];










// Configuration de la connexion à la base de données MySQL
$servername = "localhost";
$username = "root";
$password = "root";
$dbname = "squirell";

// Créez une connexion à la base de données
$conn = new mysqli($servername, $username, $password, $dbname);

// Vérifiez la connexion
if ($conn->connect_error) {
  die("Connexion échouée: " . $conn->connect_error);
}

// Requête SQL pour récupérer la table
$sql = "SELECT * FROM point_squirell WHERE (latitude BETWEEN $sudOuestLatitude AND $nordOuestLatitude) AND (longitude BETWEEN $sudOuestLongitude AND $sudEstLongitude)

AND (morph_class IN ('gray', 'black', 'other')); ";
$result = $conn->query($sql);

$tableData = array();

// Parcourez les résultats et stockez-les dans un tableau
if ($result->num_rows > 0) {
  while ($row = $result->fetch_assoc()) {
    $tableData[] = $row;
  }
}

// Fermez la connexion à la base de données
$conn->close();

// Renvoyer les données de la table au format JSON
echo json_encode($tableData);
?>
