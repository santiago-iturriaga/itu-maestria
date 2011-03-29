export path=$path:/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin
export PERL5LIB=/home/santiago/Facultad/AAI/SVMTool-1.3.1/lib:$PERL5LIB

cd model_svm

i=2

# M0
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTagger -T 0 -S LRL -V 2 full_a_${i}_0 < ../corpus/test_${i}.txt > result_a_${i}_0_0.txt
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTagger -T 2 -S LRL -V 2 full_a_${i}_0 < ../corpus/test_${i}.txt > result_a_${i}_0_2.txt
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTagger -T 5 -S LRL -V 2 full_a_${i}_0 < ../corpus/test_${i}.txt > result_a_${i}_0_5.txt

# M4
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTagger -T 4 -S LRL -V 2 full_a_${i}_4 < ../corpus/test_${i}.txt > result_a_${i}_4_6.txt
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTagger -T 6 -S LRL -V 2 full_a_${i}_4 < ../corpus/test_${i}.txt > result_a_${i}_4_4.txt

cd ..