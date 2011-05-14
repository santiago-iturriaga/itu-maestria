for i in {0..9}
do
	java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
	    -Xmx1024M AII.CompararResultadoCRF corpus/test_full_${i}.txt model_crf/f_${i}_result.txt > model_crf/f_${i}_eval.txt
done
