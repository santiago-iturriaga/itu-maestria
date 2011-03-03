java -classpath "/home/santiago/bin/mallet-2.0.6/class:/home/santiago/bin/mallet-2.0.6/lib/mallet-deps.jar" \
	cc.mallet.fst.SimpleTagger --random-seed 1 --model-file tildes.model --training-proportion 0.9 \
	--include-input false corpus.txt $1