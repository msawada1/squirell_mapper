

<?php

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
$sql = "SELECT city,lat,lng FROM `canadacities` ORDER BY city";
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
