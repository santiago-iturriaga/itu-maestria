for t in {0..3}
do
	for p in {0..3}
	do
		for i in {0..9}
		do
			java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
			    AII.CompararResultado corpus/test_full_2.txt corpus/crf_${i}_${p}_${t}.result.txt > corpus/crf_${i}_${p}_${t}.diff.txt
		done
	done
done