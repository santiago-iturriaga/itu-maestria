// Min-min scheduler.
// Parameters : <instance_ETC_file> <num_tasks> <num_machines>
//	

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#define NO_ASIG -1
#define SIZE_NOM_ARCH 180

#define DEBUG 0

int main(int argc, char *argv[]){

if (argc < 4){
        printf("Sintaxis: %s <archivo_instancias> <#tasks> <#machines>\n", argv[0]);
        exit(1);
}

int NT, NM;

FILE *fi;

char *arch_inst;
arch_inst = (char *)malloc(sizeof(char)*120);

strcpy(arch_inst,argv[1]);
fprintf(stdout,"Archivo: %s\n",arch_inst);

if((fi=fopen(arch_inst, "r"))==NULL){
    fprintf(stderr,"No se puede leer archivo de instancia %s\n",arch_inst);
    exit(1);
}

NT = atoi(argv[2]);
NM = atoi(argv[3]);
//fprintf(stdout,"NT: %d, NM: %d\n",NT,NM);

// Read input file, store ETC matrix

int i,j;

float **ETC = (float **) malloc(sizeof(float *)*NT);

if (ETC == NULL){
	fprintf(stderr,"Error in malloc for ETC matrix, dimensions %dx%d\n",NT,NM);
	exit(2);
}

for (i=0;i<NT;i++){
	ETC[i] = (float *) malloc(sizeof(float)*NM);
	if (ETC[i] == NULL){
		fprintf(stderr,"Error in malloc, row %d in ETC\n",i);
		exit(2);
	}
}

float max_etc = 0.0;

for (i=0;i<NT;i++){
	for (j=0;j<NM;j++){
        fscanf(fi,"%f",&ETC[i][j]);
	}
}

close(fi);

// Machine array, stores the MET.
float *mach = (float *) malloc(sizeof(float)*NM);
if (mach == NULL){
	fprintf(stderr,"Error in malloc (machine array), dimension %d\n",NT);
	exit(2);
}

for (j=0;j<NM;j++){
	mach[j]=0.0;
}

// Assigned tasks array
int *asig= (int*) malloc(sizeof(float)*NT);
if (asig == NULL){
	fprintf(stderr,"Error in malloc (assigned tasks array), dimension %d\n",NT);
	exit(2);
}

int nro_asig=0;
for (i=0;i<NT;i++){
	asig[i]=NO_ASIG;
}

float min_ct;
float min_ct_task;

int best_machine, best_mach_task;
int best_task;

while (nro_asig < NT){
	// Select non-assigned tasks with minimum completion time. 
	best_task = -1;
	best_machine = -1;
	min_ct = FLT_MAX;

	// Loop in tasks.
	for (i=0;i<NT;i++){
		min_ct_task = FLT_MAX;
		best_mach_task = -1;
		if (asig[i] == NO_ASIG){
			float ct=0.0;
			// Loop in machines
			for (j=0;j<NM;j++){
				ct = mach[j]+ETC[i][j];
				if (ct < min_ct_task){
					min_ct_task = ct; 
					best_mach_task = j;
				}
			}
		}
 
		if (min_ct_task < min_ct){
			min_ct = min_ct_task;
			best_task = i;
			best_machine = best_mach_task;
		}	
	}
	mach[best_machine]+=ETC[best_task][best_machine];
	asig[best_task]=best_machine;
	nro_asig++;
}

//float makespan=0.0;
//for (j=0;j<NM;j++){
//	if (mach[j]>makespan){
//		makespan = mach[j];
//	}
//}

//fprintf(stdout,"Makespan: %f\n",makespan);

//fprintf(stdout,"[");
for (i=0;i<NT;i++){
    fprintf(stdout,"%d\n",asig[i]);
}
//fprintf(stdout,"]\n");

}
