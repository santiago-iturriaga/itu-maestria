// Verificador de soluciones para instancias de Braun et al.
// Parametros: <archivo_instancia> <archivo_sol>
// El archivo de instancias DEBE llevar el cabezal con datos (NT, NM).
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INFT 9999999999.0
#define NO_ASIG -1
#define SIZE_NOM_ARCH 80

#define DEBUG 0

int main(int argc, char *argv[]){

if (argc < 5){
	printf("Sintaxis: %s <archivo_instancias> <archivo_sol> <#tasks> <#machines>\n", argv[0]);
	exit(1);
}

int NT, NM;

FILE *fi, *fs;

int i,j;

NT=atoi(argv[3]);
NM=atoi(argv[4]);

//fscanf(fi,"%d %d",&NT,&NM);
//printf("NT: %d, NM: %d\n",NT,NM);

float **ETC = (float **) malloc(sizeof(float *)*NT);

if (ETC == NULL){
	printf("Error al reservar memoria para ETC, dimensiones %dx%d\n",NT,NM);
	exit(2);
}

for (i=0;i<NT;i++){
	ETC[i] = (float *) malloc(sizeof(float)*NM);
	if (ETC[i] == NULL){
		printf("Error al reservar memoria para fila %d de ETC\n",i);
		exit(2);
	}
}

if((fi=fopen(argv[1], "r"))==NULL){
	printf("No se puede leer archivo de instancia %s\n",argv[1]);
	exit(1);
}

int max_etc=0;

for (i=0;i<NT;i++){
	for (j=0;j<NM;j++){
		fscanf(fi,"%f",&ETC[i][j]);
	}
}

close(fi);

// Array de maquinas, almacena el MAT
float *mach = (float *) malloc(sizeof(float)*NM);
for (j=0;j<NM;j++){
	mach[j]=0.0;
}

// Array de asignaciones
int *asig= (int*) malloc(sizeof(float)*NT);
int nro_asig=0;
for (i=0;i<NT;i++){
	asig[i]=NO_ASIG;
}

if((fs=fopen(argv[2], "r"))==NULL){
	printf("No se puede leer archivo de solucion %s\n", argv[2]);
	exit(1);
}

for (i=0;i<NT;i++){
	fscanf(fs,"%d ",&asig[i]);
}

fclose(fs);

for (i=0;i<NT;i++) {
    if (asig[i]==NO_ASIG) {
        printf("ERROR!!! tarea %d no asignada!!!\n", i);
        exit(1);
    }
}

int maq;

for(i=0;i<NT;i++){
	maq = asig[i];
	mach[maq] = mach[maq]+ETC[i][maq];
}

float makespan = 0.0;
for (j=0;j<NM;j++){
	if (mach[j]>makespan){
		makespan = mach[j];
	}
}

printf("%f\n",makespan);

}
