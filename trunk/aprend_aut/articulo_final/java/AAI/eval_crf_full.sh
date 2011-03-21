for i in {0..9}
do
	java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
	    AII.CompararResultado corpus/test_full_${i}.txt model_crf/result_${i}.txt > model_crf/eval_${i}.txt
done
