java -classpath /home/santiago/eclipse/java-workspace/AAI/bin:/home/santiago/eclipse/java-workspace/AAI/lib/mallet-deps.jar \
	cc.mallet.fst.SimpleTagger --train true --default-label O --model-file corpus/2.model corpus/train_2.txt

