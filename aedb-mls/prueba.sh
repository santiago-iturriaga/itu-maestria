MOEA="/home/santiago/Dropbox/Facultad/Publicaciones/EvoComnetAEDB/IJAHUC/NIDISC/comparison/data"
CLASSPATH="/home/santiago/google-hosting/itu-maestria/svn/trunk/metricas_mo/java/bin"

./convert.py best-pf-d100-c40.pf 10 > aux_best.pf
./convert2.py mls-eval-d100-c40/mls-eval-d100-c10.0.out 10 > aux_approx.pf
java -classpath ${CLASSPATH} jmetal.qualityIndicator.Spread aux_approx.pf aux_best.pf 3
java -classpath ${CLASSPATH} jmetal.qualityIndicator.Epsilon aux_approx.pf aux_best.pf 3
java -classpath ${CLASSPATH} jmetal.qualityIndicator.InvertedGenerationalDistance aux_approx.pf aux_best.pf 3
java -classpath ${CLASSPATH} jmetal.qualityIndicator.Hypervolume aux_approx.pf aux_best.pf 3

./convert3.py ${MOEA}/CellDE/100dev/FUN.0 10 > aux_cellde.pf
java -classpath ${CLASSPATH} jmetal.qualityIndicator.Epsilon aux_cellde.pf aux_best.pf 3
./convert3.py ${MOEA}/NSGAII/100dev/FUN.0 10 > aux_nsgaii.pf
java -classpath ${CLASSPATH} jmetal.qualityIndicator.Epsilon aux_nsgaii.pf aux_best.pf 3
