#!/bin/bash

# Nombre del trabajo
#PBS -N aai_svm_hmm3

# Requerimientos
#PBS -l nodes=1,walltime=48:00:00

# Cola
#PBS -q publica

# Working dir
#PBS -d /home/siturria/AprendAut/AAI/corpus_svm/

# Correo electronico
#PBS -M siturria@fing.edu.uy

# Email
#PBS -m abe
# n: no mail will be sent.
# a: mail is sent when the job is aborted by the batch system.
# b: mail is sent when the job begins execution.
# e: mail is sent when the job terminates.

# Output path
#PBS -e /home/siturria/AprendAut/AAI/corpus_svm/
#PBS -o /home/siturria/AprendAut/AAI/corpus_svm/

#PBS -V

echo Job Name: $PBS_JOBNAME
echo Working directory: $PBS_O_WORKDIR
echo Queue: $PBS_QUEUE
echo Cantidad de tasks: $PBS_TASKNUM
echo Home: $PBS_O_HOME
echo Puerto del MOM: $PBS_MOMPORT
echo Nombre del usuario: $PBS_O_LOGNAME
echo Idioma: $PBS_O_LANG
echo Cookie: $PBS_JOBCOOKIE
echo Offset de numero de nodos: $PBS_NODENUM
echo Shell: $PBS_O_SHELL
#echo JobID: $PBS_O_JOBID
echo Host: $PBS_O_HOST
echo Cola de ejecucion: $PBS_QUEUE
echo Archivo de nodos: $PBS_NODEFILE
echo Path: $PBS_O_PATH

echo
cd $PBS_O_WORKDIR
echo Current path: 
pwd
echo
echo Nodos:
cat $PBS_NODEFILE
echo
# Define number of processors
echo Cantidad de nodos:
NPROCS=`wc -l < $PBS_NODEFILE`
echo $NPROCS
echo

/home/siturria/AprendAut/svm_hmm_linux/svm_hmm_learn -c 3 -e 0.5 \
	/home/siturria/AprendAut/AAI/corpus_svm/train_2.txt \
	/home/siturria/AprendAut/AAI/corpus_svm/train_c3_2.model
