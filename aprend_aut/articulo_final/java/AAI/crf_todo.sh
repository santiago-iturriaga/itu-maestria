./train_crf_prueba.sh
./test_crf_prueba.sh > corpus/crf_2.result.txt 
./comparar_resultado.sh corpus/test_full_2.txt corpus/crf_2.result.txt 
