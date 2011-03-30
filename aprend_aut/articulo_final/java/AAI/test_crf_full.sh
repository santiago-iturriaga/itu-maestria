for i in {0..9}
do
	java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
		AII.FullCRFTest model_crf/f_${i}.model corpus/test_${i}.txt > model_crf/f_${i}_result.txt	

done
