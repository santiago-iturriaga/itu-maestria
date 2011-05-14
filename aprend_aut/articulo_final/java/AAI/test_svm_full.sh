export path=$path:/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin
export PERL5LIB=/home/santiago/Facultad/AAI/SVMTool-1.3.1/lib:$PERL5LIB

cd model_svm

for i in {2..2}
do
	/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTagger -T 5 -S GLRL f${i}_svm < ../corpus/test_${i}.txt > f${i}_result.txt
done


cd ..
