export path=$path:/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin
export PERL5LIB=/home/santiago/Facultad/AAI/SVMTool-1.3.1/lib:$PERL5LIB

cd model_svm

/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTagger -T 5 -S LRL final_svm < ../corpus/test_2.txt > final_svm_2.txt

cd ..