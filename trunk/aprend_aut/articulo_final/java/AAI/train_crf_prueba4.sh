#rm model_crf/*.model

java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
	-Xmx3000M AII.FullCRFTrainSimilSVM
