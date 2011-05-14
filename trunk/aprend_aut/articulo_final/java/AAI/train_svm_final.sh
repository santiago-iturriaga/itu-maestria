export path=$path:/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin
export PERL5LIB=/home/santiago/Facultad/AAI/SVMTool-1.3.1/lib:$PERL5LIB

cd model_svm

echo "NAME = final_svm" > final_svm.svmt
#echo "TRAINSET = /home/santiago/eclipse/java-workspace/AAI/corpus/train_2.txt" >> final_svm.svmt
#echo "TESTSET = /home/santiago/eclipse/java-workspace/AAI/corpus/test_full_2.txt" >> final_svm.svmt
echo "TRAINSET = /home/santiago/eclipse/java-workspace/AAI/SVMTrainFinal.log" >> final_svm.svmt
echo "TESTSET = /home/santiago/eclipse/java-workspace/AAI/SVMTestFinal.log" >> final_svm.svmt
echo "SVMDIR = /home/santiago/Facultad/AAI/svm_light/" >> final_svm.svmt
echo "REMOVE_FILES = 1" >> final_svm.svmt
#echo "F = 1 200000" >> final_svm.svmt
#echo "F = 3 200000" >> final_svm.svmt (13/65)
#echo "F = 2 200000" >> final_svm.svmt (16/65)
#echo "F = 1 200000" >> final_svm.svmt (17/65)
#echo "F = 0 200000" >> final_svm.svmt #(?/65)
#echo "X = 10" >> final_svm.svmt
#echo "W = 5 2" >> final_svm.svmt
#echo "Kfilter = 0" >> final_svm.svmt
#echo "Ufilter = 0" >> final_svm.svmt

#DEFAULT (19/65)
#echo "A0k = w(-2) w(-1) w(0) w(1) w(2) w(-2,-1) w(-1,0) w(0,1) w(-1,1) w(1,2) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) w(-1,1,2) w(0,1,2) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) a(0) a(1) a(2) m(0) m(1) m(2)" >> final_svm.svmt
#echo "A0u = w(-2) w(-1) w(0) w(1) w(2) w(-2,-1) w(-1,0) w(0,1) w(-1,1) w(1,2) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) w(-1,1,2) w(0,1,2) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) a(0) a(1) a(2) m(0) m(1) m(2) a(2) a(3) a(4) z(2) z(3) z(4) ca(1) cz(1) L SA AA SN CA CAA CP CC CN MW" >> final_svm.svmt

#MOD1 (19/65)
#echo "A0k = w(-2) w(-1) w(0) w(1) w(2) w(-2,-1) w(-1,0) w(0,1) w(-1,1) w(1,2) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) w(-1,1,2) w(0,1,2) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) a(0) a(1) a(2) m(0) m(1) m(2)" >> final_svm.svmt
#echo "A0u = w(-2) w(-1) w(0) w(1) w(2) w(-2,-1) w(-1,0) w(0,1) w(-1,1) w(1,2) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) w(-1,1,2) w(0,1,2) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) a(0) a(1) a(2) m(0) m(1) m(2) a(2) a(3) a(4) z(2) z(3) z(4) ca(1) cz(1)" >> final_svm.svmt

#MOD2 (13/65)
#echo "A0k = w(-2) w(-1) w(0) w(1) w(2) w(-2,-1) w(-1,0) w(0,1) w(-1,1) w(1,2) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) w(-1,1,2) w(0,1,2) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) a(0) a(1) a(2) m(0) m(1) m(2) a(2) a(3) a(4) z(2) z(3) z(4) ca(1) cz(1) L MW CC CN CP SN AA aa CA SA sa" >> final_svm.svmt
#echo "A0u = w(-2) w(-1) w(0) w(1) w(2) w(-2,-1) w(-1,0) w(0,1) w(-1,1) w(1,2) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) w(-1,1,2) w(0,1,2) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) a(0) a(1) a(2) m(0) m(1) m(2) a(2) a(3) a(4) z(2) z(3) z(4) ca(1) cz(1) L MW CC CN CP SN AA aa CA SA sa" >> final_svm.svmt

#MOD3 (14/65)
#echo "A0k = w(-2) w(-1) w(0) w(1) w(2) w(-2,-1) w(-1,0) w(0,1) w(-1,1) w(1,2) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) w(-1,1,2) w(0,1,2) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) a(0) a(1) a(2) m(0) m(1) m(2) a(2) a(3) a(4) z(2) z(3) z(4) ca(1) cz(1) L SA AA SN CA CAA CP CC CN MW" >> final_svm.svmt
#echo "A0u = w(-2) w(-1) w(0) w(1) w(2) w(-2,-1) w(-1,0) w(0,1) w(-1,1) w(1,2) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) w(-1,1,2) w(0,1,2) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) a(0) a(1) a(2) m(0) m(1) m(2) a(2) a(3) a(4) z(2) z(3) z(4) ca(1) cz(1) L SA AA SN CA CAA CP CC CN MW" >> final_svm.svmt

#MOD4 (17/65)
#echo "A0k = w(-2) w(-1) w(0) w(1) w(-2,-1) w(-1,0) w(0,1) w(-1,1) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) p(-2,-1) p(-1,1) p(1,2) a(0) m(0) z(2) z(3) ca(1) cz(1) L CC CP SA sa" >> final_svm.svmt
#echo "A0u = w(-2) w(-1) w(0) w(1) w(-2,-1) w(-1,0) w(0,1) w(-1,1) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) p(-2,-1) p(-1,1) p(1,2) a(0) m(0) z(2) z(3) ca(1) cz(1) L CC CP SA sa" >> final_svm.svmt

#MOD5
#echo "A0k = w(-2) aw(-1) w(0) w(1) w(2) w(-2,-1) w(-1,0) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) a(0) a(1) a(2) m(0) m(1) m(2)" >> final_svm.svmt
#echo "A0u = w(-2) w(-1) w(0) w(1) w(2) w(-2,-1) w(-1,0) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) a(0) a(1) a(2) m(0) m(1) m(2) a(2) a(3) a(4) z(2) z(3) z(4) ca(1) cz(1)" >> final_svm.svmt

#MOD6 19
#echo "A0k = w(-3) w(-2) w(-1) w(0) w(1) w(2) w(3) w(-3,-2) w(-2,-1) w(-1,0) w(0,1) w(1,2) w(2,3) w(-1,1) w(-3,-2,-1) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) w(-1,1,2) w(0,1,2) w(1,2,3) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) a(0) a(1) a(2) m(0) m(1) m(2)" >> final_svm.svmt
#echo "A0u = w(-3) w(-2) w(-1) w(0) w(1) w(2) w(3) w(-3,-2) w(-2,-1) w(-1,0) w(0,1) w(1,2) w(2,3) w(-1,1) w(-3,-2,-1) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) w(-1,1,2) w(0,1,2) w(1,2,3) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) a(0) a(1) a(2) m(0) m(1) m(2) a(2) a(3) a(4) z(2) z(3) z(4) ca(1) cz(1) L SA AA SN CA CAA CP CC CN MW" >> final_svm.svmt

#MOD7
#echo "A0k = w(-4) w(-3) w(-2) w(-1) w(0) w(1) w(2) w(3) w(4) w(-4,-3) w(-3,-2) w(-2,-1) w(-1,0) w(0,1) w(1,2) w(2,3) w(3,4) w(-1,1) w(-4,-3,-2) w(-3,-2,-1) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) w(-1,1,2) w(0,1,2) w(1,2,3) w(2,3,4) p(-4) p(-3) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) p(3) p(4) a(0) a(1) a(2) m(0) m(1) m(2)" >> final_svm.svmt
#echo "A0u = w(-4) w(-3) w(-2) w(-1) w(0) w(1) w(2) w(3) w(4) w(-4,-3) w(-3,-2) w(-2,-1) w(-1,0) w(0,1) w(1,2) w(2,3) w(3,4) w(-1,1) w(-4,-3,-2) w(-3,-2,-1) w(-2,-1,0) w(-2,-1,1) w(-1,0,1) w(-1,1,2) w(0,1,2) w(1,2,3) w(2,3,4) p(-4) p(-3) p(-2) p(-1) p(-2,-1) p(-1,1) p(1,2) p(-2,-1,1) p(-1,1,2) p(3) p(4) a(0) a(1) a(2) m(0) m(1) m(2) a(2) a(3) a(4) z(2) z(3) z(4) ca(1) cz(1) L SA AA SN CA CAA CP CC CN MW" >> final_svm.svmt

echo "do M0 LRL" >> final_svm.svmt

/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTlearn final_svm.svmt

cd ..
