find ./src -name *.java > sources_list.txt

/usr/java/jdk1.6.0_03/bin/javac -classpath /home/siturria/AprendAut/AAI/src:/home/siturria/AprendAut/AAI/lib/mallet-deps.jar \
	-Xlint:deprecation,unchecked -d bin @sources_list.txt
