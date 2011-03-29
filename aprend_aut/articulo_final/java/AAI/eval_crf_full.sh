#for i in {0..9}
#do
#	java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
#	    AII.CompararResultado corpus/test_full_${i}.txt model_crf/result_${i}.txt > model_crf/eval_${i}.txt
#done

for t in {0..2}
do
	for p in {0..2}
	do
		for i in {0..8}
		do
			java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
			    AII.CompararResultado corpus/test_full_2.txt model_crf/t1_crf_${i}_${p}_${t}.result.txt > model_crf/t1_crf_${i}_${p}_${t}.eval.txt
		done
	done
done 
   
java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
    AII.CompararResultado corpus/test_full_2.txt model_crf/t2_crf_2_5.result.txt > model_crf/t2_crf_2_5.eval.txt

java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
    AII.CompararResultado corpus/test_full_2.txt model_crf/t2_crf_2_15.result.txt > model_crf/t2_crf_2_15.eval.txt

java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
    AII.CompararResultado corpus/test_full_2.txt model_crf/t2_crf_2_20.result.txt > model_crf/t2_crf_2_20.eval.txt

java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
    AII.CompararResultado corpus/test_full_2.txt  model_crf/t2_crf_2_10.result.txt > model_crf/t2_crf_2_10.eval.txt

java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
    AII.CompararResultado corpus/test_full_2.txt  model_crf/induce_crf_2.result.txt > model_crf/induce_crf_2.eval.txt

java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
    AII.CompararResultado corpus/test_full_2.txt model_crf/svm_result_2.txt > model_crf/svm_crf_2.eval.txt
    