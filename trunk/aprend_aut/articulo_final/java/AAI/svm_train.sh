#/home/santiago/bin/svm_hmm_linux/svm_hmm_learn -c 3 -e 0.5 \
#	/home/santiago/eclipse/java-workspace/AAI/corpus_svm/train_2.txt \
#	/home/santiago/eclipse/java-workspace/AAI/corpus_svm/train_2.model

export path=$path:/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin
export PERL5LIB=/home/santiago/Facultad/AAI/SVMTool-1.3.1/lib:$PERL5LIB

/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTlearn /home/santiago/eclipse/java-workspace/AAI/train_config.svmt
