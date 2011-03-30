export path=$path:/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin
export PERL5LIB=/home/santiago/Facultad/AAI/SVMTool-1.3.1/lib:$PERL5LIB

cd model_svm

/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTagger -T 5 -S LRL final_svm < /home/santiago/eclipse/java-workspace/AAI/SVMTestFinal.log > final_svm_2.txt

cd ..