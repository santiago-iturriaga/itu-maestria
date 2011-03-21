#/home/santiago/Desktop/svm/svm_hmm_linux/svm_hmm_classify \
#	/home/santiago/eclipse/java-workspace/AAI/corpus_svm/test_0.txt \
#	/home/santiago/eclipse/java-workspace/AAI/corpus_svm/train_0.model \
#	/home/santiago/eclipse/java-workspace/AAI/corpus_svm/result_0.txt

export path=$path:/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin
export PERL5LIB=/home/santiago/Facultad/AAI/SVMTool-1.3.1/lib:$PERL5LIB

#/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval /home/santiago/eclipse/java-workspace/AAI/test_config.svmt
#/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTagger -T 0 -S LR -V 2 svmtool_c2 \
#	/home/santiago/eclipse/java-workspace/AAI/corpus/test_2.txt /home/santiago/prueba.txt
#/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTlearn -V 2 /home/santiago/eclipse/java-workspace/AAI/eval_config.svmt
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 svmt_c2 corpus/test_full_2.txt prueba.txt
