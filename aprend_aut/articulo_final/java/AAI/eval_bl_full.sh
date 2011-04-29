for i in {0..9}
do
	java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
	    -Xmx1024M AII.CompararResultadoCRF corpus/test_full_${i}.txt model_bl/result_${i}.txt > model_bl/eval_${i}.txt
done
