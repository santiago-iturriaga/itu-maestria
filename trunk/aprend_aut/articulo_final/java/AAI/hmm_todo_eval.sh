java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
	AII.HMMTest corpus/hmm_0.model > corpus/hmm_0.result.txt 
./comparar_resultado.sh corpus/test_full_1.txt corpus/hmm_0.result.txt 

java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
	AII.HMMTest corpus/hmm_1.model > corpus/hmm_1.result.txt 
./comparar_resultado.sh corpus/test_full_1.txt corpus/hmm_1.result.txt

java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
	AII.HMMTest corpus/hmm_2.model > corpus/hmm_2.result.txt 
./comparar_resultado.sh corpus/test_full_1.txt corpus/hmm_2.result.txt

java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
	AII.HMMTest corpus/hmm_3.model > corpus/hmm_3.result.txt 
./comparar_resultado.sh corpus/test_full_1.txt corpus/hmm_3.result.txt

java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
	AII.HMMTest corpus/hmm_4.model > corpus/hmm_4.result.txt 
./comparar_resultado.sh corpus/test_full_1.txt corpus/hmm_4.result.txt

java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
	AII.HMMTest corpus/hmm_5.model > corpus/hmm_5.result.txt 
./comparar_resultado.sh corpus/test_full_1.txt corpus/hmm_5.result.txt

java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
	AII.HMMTest corpus/hmm_6.model > corpus/hmm_6.result.txt 
./comparar_resultado.sh corpus/test_full_1.txt corpus/hmm_6.result.txt

java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
	AII.HMMTest corpus/hmm_7.model > corpus/hmm_7.result.txt 
./comparar_resultado.sh corpus/test_full_1.txt corpus/hmm_7.result.txt
