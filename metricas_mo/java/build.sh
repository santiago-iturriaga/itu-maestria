mkdir /home/siturria/AE/metricas_mo/java/bin
rm -rf /home/siturria/AE/metricas_mo/java/bin/*
/usr/java/jdk1.6.0_03/bin/javac -classpath /home/siturria/AE/metricas_mo/java/src -d /home/siturria/AE/metricas_mo/java/bin $(find . -name "*.java")

