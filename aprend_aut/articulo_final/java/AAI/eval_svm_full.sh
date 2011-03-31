export path=$path:/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin
export PERL5LIB=/home/santiago/Facultad/AAI/SVMTool-1.3.1/lib:$PERL5LIB

cd model_svm

for i in {0..9}
do
	/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 f${i}_svm ../corpus/test_full_${i}.txt f${i}_result.txt > f${i}_eval.txt
done

cd ..
