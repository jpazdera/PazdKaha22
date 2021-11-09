<?php
// the $_POST[] array will contain the passed in filename and data
// the directory "data" is writable by the server (chmod 777)
$filename = $_POST['filename'];
$data = $_POST['filedata'];
// write the file to disk
$fh = fopen($filename, 'w') or die("Unable to open file!");
fwrite($fh, $data);
fclose($fh);
?>
