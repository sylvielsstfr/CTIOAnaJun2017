# README.md

- memorize how to convert movies in images using ffmeg


Jean Cartier’s Blog :
https://www.jcartier.net/spip.php?article36

ffmpeg -i film_2h00.mov -vf format=gray image%d.tiff

ffmpeg -i film_2h00.mov image%d.tif

ffmpeg -i film_2h00.mov image%d.png

ffmpeg -i film_2h00.mov

