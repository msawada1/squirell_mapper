<?php

$content = trim(file_get_contents("php://input"));
$data = json_decode($content, true);
$objet = $data['objet'];
$imageFilePath = $data['imagePath'];

$pythonInterpreter = 'C:\Program Files\Python311\python.exe';

$pythonInputFilePath = 'python_input.json';
$interprete = $data['python_inter'];

$pythonInputData = [
    'objet' => $objet,
    'imagePath' => $imageFilePath,
    'interpeter' => $interprete
];

file_put_contents($pythonInputFilePath, json_encode($pythonInputData));

$pythonOutputFilePath = 'python_output.txt';

file_put_contents($pythonOutputFilePath,'');

$chemin_absolu = realpath('script.py');


$pythonScriptPath = $chemin_absolu;


$command = '"' . $pythonInterpreter . '" "' . $pythonScriptPath . '" "' . $pythonInputFilePath . '" "' . $pythonOutputFilePath . '"';


$output = shell_exec($command);

// Récupération de la sortie du script Python
$pythonOutput = file_get_contents($pythonOutputFilePath);


echo $pythonOutput;

// Suppression des fichiers intermédiaires
unlink($pythonInputFilePath);
unlink($pythonOutputFilePath);

?>
