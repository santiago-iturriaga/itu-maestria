java -classpath "/home/santiago/bin/mallet-2.0.6/class:/home/santiago/bin/mallet-2.0.6/lib/mallet-deps.jar" \
	cc.mallet.fst.SimpleTagger --train true --iterations 1000 \
	--default-label O --weights sparse \
	--model-file corpus/3.model corpus/train_3.txt

