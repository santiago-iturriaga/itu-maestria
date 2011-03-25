export path=$path:/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin
export PERL5LIB=/home/santiago/Facultad/AAI/SVMTool-1.3.1/lib:$PERL5LIB

cd model_svm

for i in {2..2}
do
	#/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTagger -T 0 -S LR -V 2 svm_t${i} < ../corpus/test_${i}.txt > result_${i}.txt
	/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTagger -T 5 -S LRL -V 2 svm_t${i} < ../corpus/test_${i}.txt > result_${i}_2.txt
done

cd ..