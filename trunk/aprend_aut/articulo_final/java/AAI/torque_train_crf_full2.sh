#!/bin/bash

# Nombre del trabajo
#PBS -N train_crf2

# Requerimientos
#PBS -l nodes=1:ram24,walltime=48:00:00

# Cola
#PBS -q publica

# Working dir
#PBS -d /home/siturria/AprendAut/AAI/

# Correo electronico
###PBS -M siturria@fing.edu.uy

# Email
#PBS -m abe
# n: no mail will be sent.
# a: mail is sent when the job is aborted by the batch system.
# b: mail is sent when the job begins execution.
# e: mail is sent when the job terminates.

# Output path
#PBS -e /home/siturria/AprendAut/AAI/
#PBS -o /home/siturria/AprendAut/AAI/

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

rm model_crf/*.model

/usr/java/jdk1.6.0_03/bin/java -classpath /home/siturria/AprendAut/AAI/bin:/home/siturria/AprendAut/AAI/lib/mallet-deps.jar \
	-Xmx12000M AII.FullCRFTrainSimilSVM
	