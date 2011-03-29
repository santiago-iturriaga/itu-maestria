export path=$path:/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin
export PERL5LIB=/home/santiago/Facultad/AAI/SVMTool-1.3.1/lib:$PERL5LIB

cd model_svm

i=2

for m in {0..4}
do
	echo "NAME = full_c_${i}_${m}" > t_full_c_${i}_${m}.svmt
	echo "TRAINSET = /home/santiago/eclipse/java-workspace/AAI/corpus/train_${i}.txt" >> t_full_c_${i}_${m}.svmt
	echo "VALSET = /home/santiago/eclipse/java-workspace/AAI/corpus/test_full_${i}.txt" >> t_full_c_${i}_${m}.svmt
	echo "SVMDIR = /home/santiago/Facultad/AAI/svm_light/" >> t_full_c_${i}_${m}.svmt
	echo "REMOVE_FILES = 1" >> t_full_c_${i}_${m}.svmt
	echo "do M${m} LRL CK:0.01:10:3:10:log CU:0.01 T" >> t_full_c_${i}_${m}.svmt

	/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTlearn t_full_c_${i}_${m}.svmt
done

for m in {0..4}
do
	echo "NAME = full2_c_${i}_${m}" > t_full2_c_${i}_${m}.svmt
	echo "TRAINSET = /home/santiago/eclipse/java-workspace/AAI/corpus/train_${i}.txt" >> t_full2_c_${i}_${m}.svmt
	echo "VALSET = /home/santiago/eclipse/java-workspace/AAI/corpus/test_full_${i}.txt" >> t_full2_c_${i}_${m}.svmt
	echo "SVMDIR = /home/santiago/Facultad/AAI/svm_light/" >> t_full2_c_${i}_${m}.svmt
	echo "REMOVE_FILES = 1" >> t_full2_c_${i}_${m}.svmt
	echo "do M${m} LRL CK:0.01:10:3:10:log CU:0.07 T" >> t_full2_c_${i}_${m}.svmt

	/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTlearn t_full2_c_${i}_${m}.svmt
done


cd ..
