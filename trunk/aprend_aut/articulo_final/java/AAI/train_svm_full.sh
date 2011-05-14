export path=$path:/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin
export PERL5LIB=/home/santiago/Facultad/AAI/SVMTool-1.3.1/lib:$PERL5LIB

cd model_svm

for i in {2..2}
do
	echo "NAME = f${i}_svm" > f${i}.svmt
	echo "TRAINSET = /home/santiago/eclipse/java-workspace/AAI/corpus/train_${i}.txt" >> f${i}.svmt
	echo "TESTSET = /home/santiago/eclipse/java-workspace/AAI/corpus/test_full_${i}.txt" >> f${i}.svmt
	echo "SVMDIR = /home/santiago/Facultad/AAI/svm_light/" >> f${i}.svmt
	echo "REMOVE_FILES = 1" >> f${i}.svmt
	echo "F = 1 200000" >> f${i}.svmt
	echo "X = 10" >> f${i}.svmt
	echo "W = 5 2" >> f${i}.svmt
	echo "Kfilter = 0" >> f${i}.svmt
	echo "Ufilter = 0" >> f${i}.svmt
	echo "A0k = w(-2) w(-1) w(0) w(1) w(2) w(-2,-1) w(-1,0) w(0,1) w(-1,1) w(1,2) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) w(-1,1,2) w(0,1,2) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) SA aa" >> f${i}.svmt
	echo "A0u = w(-2) w(-1) w(0) w(1) w(2) w(-2,-1) w(-1,0) w(0,1) w(-1,1) w(1,2) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) w(-1,1,2) w(0,1,2) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) SA aa" >> f${i}.svmt
	echo "do M0 LRL" >> f${i}.svmt
	
	/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTlearn f${i}.svmt
done

cd ..
