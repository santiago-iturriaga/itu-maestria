export path=$path:/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin
export PERL5LIB=/home/santiago/Facultad/AAI/SVMTool-1.3.1/lib:$PERL5LIB

cd model_svm

#for i in {0..9}
#do
#	/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 svm_t${i} ../corpus/test_full_${i}.txt result_${i}.txt > eval_${i}.txt
#done

/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 svm_t2 ../corpus/test_full_2.txt result_2_0.txt > eval_2_0.txt
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 svm_t2 ../corpus/test_full_2.txt result_2_2.txt > eval_2_2.txt
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 svm_t2 ../corpus/test_full_2.txt result_2_5.txt > eval_2_5.txt

/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 full_c_2_0 ../corpus/test_full_2.txt result_c_2_0_0.txt > eval_c_2_0_0.txt
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 full_c_2_0 ../corpus/test_full_2.txt result_c_2_0_2.txt > eval_c_2_0_2.txt
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 full_c_2_0 ../corpus/test_full_2.txt result_c_2_0_5.txt > eval_c_2_0_5.txt

/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 full_c_2_4 ../corpus/test_full_2.txt result_c_2_4_6.txt > eval_c_2_4_6.txt
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 full_c_2_4 ../corpus/test_full_2.txt result_c_2_4_4.txt > eval_c_2_4_4.txt

/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 full2_c_2_0 ../corpus/test_full_2.txt result_c2_2_0_0.txt > eval_c2_2_0_0.txt
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 full2_c_2_0 ../corpus/test_full_2.txt result_c2_2_0_2.txt > eval_c2_2_0_2.txt
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 full2_c_2_0 ../corpus/test_full_2.txt result_c2_2_0_5.txt > eval_c2_2_0_5.txt

/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 full2_c_2_4 ../corpus/test_full_2.txt result_c2_2_4_6.txt > eval_c2_2_4_6.txt
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 full2_c_2_4 ../corpus/test_full_2.txt result_c2_2_4_4.txt > eval_c2_2_4_4.txt

/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 full_a_2_0 ../corpus/test_full_2.txt result_a_2_0_0.txt > eval_a_2_0_0.txt
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 full_a_2_0 ../corpus/test_full_2.txt result_a_2_0_2.txt > eval_a_2_0_2.txt
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 full_a_2_0 ../corpus/test_full_2.txt result_a_2_0_5.txt > eval_a_2_0_5.txt

/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 full_a_2_4 ../corpus/test_full_2.txt result_a_2_4_6.txt > eval_a_2_4_6.txt
/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTeval 0 full_a_2_4 ../corpus/test_full_2.txt result_a_2_4_4.txt > eval_a_2_4_4.txt

cd ..