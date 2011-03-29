export path=$path:/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin
export PERL5LIB=/home/santiago/Facultad/AAI/SVMTool-1.3.1/lib:$PERL5LIB

cd model_svm

i=2
m=0

echo "NAME = full_a_${i}_${m}" > t_full_a_${i}_${m}.svmt
echo "TRAINSET = /home/santiago/eclipse/java-workspace/AAI/corpus/train_${i}.txt" >> t_full_a_${i}_${m}.svmt
echo "TESTSET = /home/santiago/eclipse/java-workspace/AAI/corpus/test_full_${i}.txt" >> t_full_a_${i}_${m}.svmt
echo "SVMDIR = /home/santiago/Facultad/AAI/svm_light/" >> t_full_a_${i}_${m}.svmt
echo "REMOVE_FILES = 1" >> t_full_a_${i}_${m}.svmt
echo "A0k = w(-2) w(-1) w(0) w(1) w(2) w(-2,-1) w(-1,0) w(0,1) w(-1,1) w(1,2) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) w(-1,1,2) w(0,1,2) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) a(0) a(1) a(2) m(0) m(1) m(2)" >> t_full_a_${i}_${m}.svmt
echo "A0u = w(-2) w(-1) w(0) w(1) w(2) w(-2,-1) w(-1,0) w(0,1) w(-1,1) w(1,2) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) w(-1,1,2) w(0,1,2) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) a(0) a(1) a(2) m(0) m(1) m(2) a(2) a(3) a(4) z(2) z(3) z(4) ca(1) cz(1)" >> t_full_a_${i}_${m}.svmt
echo "do M${m} LRL" >> t_full_a_${i}_${m}.svmt

/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTlearn t_full_a_${i}_${m}.svmt

cd ..
