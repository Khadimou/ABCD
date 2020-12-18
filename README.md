# CNN_ABCD

Bienvenue dans la branche master du projet ABCD (réseau de neurones de détection de cancer du sein)

# convertir les images pgm en ppm avec ffmpeg
for im in *.pgm; do echo "Converting: "$im; ffmpeg -i $im $( basename $im pgm)ppm; done

# Appliquer un filtre à un ensemble d'images
for f in *.pgm; do echo $f; ../filtre -sobel $f name_folder/$f; done
