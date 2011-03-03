java -classpath "/home/santiago/bin/mallet-2.0.6/class:/home/santiago/bin/mallet-2.0.6/lib/mallet-deps.jar" \
	cc.mallet.fst.SimpleTagger --train true --iterations 1000 --training-proportion 0.9 \
	--random-seed 1 --default-label O --weights sparse --model-file tildes.model corpus.txt

