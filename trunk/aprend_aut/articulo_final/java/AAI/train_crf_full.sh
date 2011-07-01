#rm model_crf/*.model

for i in {0..9}
do
	java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
		-Xmx3072M AII.FullCRFTrain ${i}
done