export path=$path:/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin
export PERL5LIB=/home/santiago/Facultad/AAI/SVMTool-1.3.1/lib:$PERL5LIB

cd model_svm

for i in {0..9}
do
	# -------------------------------------------------------------
	#SVMTool configuration file for English on the whole WSJ corpus
	# -------------------------------------------------------------
	#prefix of the model files which will be created
	echo "NAME = svm_t${i}" > model_svm/train_${i}.svmt
	# -------------------------------------------------------------
	#location of the training set
	echo "TRAINSET = /home/santiago/eclipse/java-workspace/AAI/corpus/train_${i}.txt" >> train_${i}.svmt
	# -------------------------------------------------------------
	#location of the Joachims svmlight software
	echo "SVMDIR = /home/santiago/Facultad/AAI/svm_light/" >> train_${i}.svmt
	# -------------------------------------------------------------
	#action items
	# -------------------------------------------------------------
	echo "do M0 LR" >> train_${i}.svmt
	# -------------------------------------------------------------

	/home/santiago/Facultad/AAI/SVMTool-1.3.1/bin/SVMTlearn train_${i}.svmt
done

cd ..
