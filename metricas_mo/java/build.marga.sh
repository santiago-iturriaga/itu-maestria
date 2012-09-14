ruta=$(pwd)

mkdir bin
rm -rf bin/*
javac -classpath ${ruta}/src -d ${ruta}/bin $(find . -name "*.java")

