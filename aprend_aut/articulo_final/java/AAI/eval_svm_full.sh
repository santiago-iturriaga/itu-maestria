export path=$path:/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin
export PERL5LIB=/home/santiago/Facultad/AAI/SVMTool-1.3.1/lib:$PERL5LIB

cd model_svm

for i in {0..9}
do
	/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 svm_t${i} ../corpus/test_full_${i}.txt result_${i}.txt > eval_${i}.txt
done

cd ..