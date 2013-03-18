// Min-Min energy-aware scheduler.
// Phase 1: Pair with minimum  ETC
// Phase 2: Minimum ETC.
// Parameters : <instance_ETC_file> <num_tasks> <num_machines> 
//	

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#define NO_ASIG -1
#define SIZE_NOM_ARCH 180

#define DEBUG 0

float energy_mach(float *mach, int num_cores, float E_I, float E_M, int napp){

float curr_energy = 0.0;
int h;
float E_h;

#if DEBUG
printf("function energy_m start\n--------\n");
printf("M: [");
for (h=0;h<num_cores;h++){
	printf("%f ",mach[h]);
}
printf("]\n");
#endif
// Calculate the energy consumption for a given assignment in machine j
if (napp == 0){
	return 0.0;
}
if (napp >= num_cores){
	curr_energy = E_M*mach[0];
#if DEBUG
printf("napp >= ncores. napp: %d, num_cores: %d, E_I: %f, E_M: %f, mach[0]=%f, curr energy: %f\n",napp,num_cores,E_I,E_M,mach[0],curr_energy);

#endif
	for(h=1;h<num_cores;h++){
		E_h = E_I+((E_M-E_I)*(num_cores-h)/num_cores);
		curr_energy = curr_energy + E_h*(mach[h]-mach[h-1]);
#if DEBUG
printf("for (1): napp: %d, num_cores: %d, E_I: %f, E_M: %f, E(%d): %f, mach[%d]=%f, mach[%d]=%f, curr energy: %f\n",napp,num_cores,E_I,E_M,num_cores-h,E_h,h,mach[h],h-1,mach[h-1],curr_energy);
#endif
	}
} else {
	E_h = E_I+((E_M-E_I)*napp/num_cores);
	curr_energy = E_h * mach[num_cores-napp];
#if DEBUG
printf("napp < ncores. napp: %d, num_cores: %d, E_I: %f, E_M: %f, E(%d): %f, mach[%d]=%f, mach[%d]=%f, curr energy: %f\n",napp,num_cores,E_I,E_M,napp,E_h,h,mach[h],h-1,mach[h-1],curr_energy);
#endif
	for(h=1;h<napp;h++){
		curr_energy = curr_energy + (mach[num_cores-napp+h]-mach[num_cores-napp+h-1])*(E_I+((E_M-E_I)*(napp-h)/num_cores));
#if DEBUG
printf("for: napp: %d, num_cores: %d, E_I: %f, E_M: %f, mach[%d]=%f, mach[%d]=%f, curr energy: %f\n",napp,num_cores,E_I,E_M,h,mach[h],h-1,mach[h-1],curr_energy);
#endif
	}
}

#if DEBUG
printf("napp: %d, num_cores: %d, E_I: %f, E_M: %f, energy: %f\n",napp,num_cores,E_I,E_M,curr_energy);
printf("function energy_m end\n--------\n");
#endif

return curr_energy;

}

int main(int argc, char *argv[]){

if (argc < 5){
        //printf("Sintaxis: %s <workload index> <scenario index> <num_tasks> <num_machines> <etc_model: 0-Zomaya, 1-Braun>\n", argv[0]);
        printf("Sintaxis: %s <workload index> <scenario index> <num_tasks> <num_machines> \n", argv[0]);
        exit(1);
}

int NT, NM;
FILE *fi, *fp;
char *arch_inst, *arch_proc;

NT = atoi(argv[3]);
NM = atoi(argv[4]);

arch_inst = (char *)malloc(sizeof(char)*120);
strcpy(arch_inst,"instances/workload.");
arch_inst = strcat(arch_inst,argv[1]);

arch_proc = (char *)malloc(sizeof(char)*120);
strcpy(arch_proc,"instances/scenario.");

arch_proc = strcat(arch_proc,argv[2]);

fprintf(stdout,"NT: %d, NM: %d, arch_ETC: %s, arch_proc: %s\n",NT,NM,arch_inst,arch_proc);

float *E_IDLE = (float *) malloc(sizeof(float)*NM);
if (E_IDLE == NULL){
    fprintf(stderr,"Error in malloc for E_IDLE matrix, dimension %d\n",NM);
    exit(2);
}

float *E_MAX = (float *) malloc(sizeof(float)*NM);
if (E_MAX == NULL){
    fprintf(stderr,"Error in malloc for E_MAX matrix, dimension %d\n",NM);
    exit(2);
}

int *cores = (int *) malloc(sizeof(int)*NM);
if (cores == NULL){
    fprintf(stderr,"Error in malloc for cores matrix, dimension %d\n",NM);
    exit(2);
}

float *GFLOPS = (float *) malloc(sizeof(float)*NM);
if (GFLOPS == NULL){
    fprintf(stderr,"Error in malloc for GFLOPS matrix, dimension %d\n",NM);
    exit(2);
}

float **ETC = (float **) malloc(sizeof(float *)*NT);
if (ETC == NULL){
    fprintf(stderr,"Error in malloc for ETC matrix, dimensions %dx%d\n",NT,NM);
    exit(2);
}

int i,j,h,k;

for (i=0;i<NT;i++){
    ETC[i] = (float *) malloc(sizeof(float)*NM);
    if (ETC[i] == NULL){
        fprintf(stderr,"Error in malloc, row %d in ETC\n",i);
        exit(2);
    }
}

// Machine array, stores the MET.
float **mach = (float **) malloc(sizeof(float *)*NM);
if (mach == NULL){
	fprintf(stderr,"Error in malloc (machine array), dimension %d\n",NM);
	exit(2);
}

// Read input files, store ETC matrix and proc. info

if((fp=fopen(arch_proc, "r"))==NULL){
    fprintf(stderr,"Can't read processor file: %s\n",arch_inst);
    exit(1);
}

for (j=0;j<NM;j++){
    fscanf(fp,"%d %f %f %f\n",&cores[j],&GFLOPS[j],&E_IDLE[j],&E_MAX[j]);
}

for (j=0;j<NM;j++){
    mach[j] = (float *) malloc(sizeof(float)*cores[j]);
    if (mach[j] == NULL){
        fprintf(stderr,"Error in malloc, row %d in mach.\n",j);
        exit(2);
    }
	for(h=0;h<cores[j];h++){
		mach[j][h] = 0.0;
	}
}

close(fp);

if((fi=fopen(arch_inst, "r"))==NULL){
    fprintf(stderr,"Can't read instance file: %s\n",arch_inst);
    exit(1);
}

for (i=0;i<NT;i++){
    for (j=0;j<NM;j++){
        fscanf(fi,"%f",&ETC[i][j]);
        ETC[i][j] = ETC[i][j]/(GFLOPS[j]/(1000.0*cores[j]));
        ETC[i][j] = ETC[i][j]/1000.0;
    }
}

close(fi);

// Number of applications array
int *napp = (int*) malloc(sizeof(float)*NM);
if (napp == NULL){
    fprintf(stderr,"Error in malloc (number of applications array), dimension %d\n",NM);
    exit(2);
}

for (j=0;j<NM;j++){
    napp[j] = 0;
}

// Assigned tasks array

int *asig= (int*) malloc(sizeof(float)*NT);
if (asig == NULL){
	fprintf(stderr,"Error in malloc (assigned tasks array), dimension %d\n",NT);
	exit(2);
}

int nro_asig = 0;
for (i=0;i<NT;i++){
	asig[i]=NO_ASIG;
}

float mct_i_j;
float min_ct, min_ct_task;

int best_machine, best_mach_task;
int best_task;

float et, new_et;

while (nro_asig < NT){
	// Select non-assigned tasks with maximun robustness radio - minimum completion time. 
	best_task = -1;
	best_machine = -1;
	min_ct = FLT_MAX;

	// Loop on tasks.
	for (i=0;i<NT;i++){
		best_mach_task = -1;
		min_ct_task = FLT_MAX;

		if (asig[i] == NO_ASIG){
			// Loop on machines
			for (j=0;j<NM;j++){
				// Evaluate MCT of (ti, mj)
				// mach[j][0] has the min local makespan for machine j.
				et = mach[j][0]+ETC[i][j];
				if (et < min_ct_task){
					min_ct_task = et;	
					best_mach_task = j;
				}
			}
		//mct_i_j = mach[best_mach_task][0]+ETC[i][best_mach_task];

			if (min_ct_task <= min_ct){
				min_ct = min_ct_task; 
				best_task = i;
				best_machine = best_mach_task;
			}		
		}
	}

#if DEBUG
printf("********************* Assign task %d to machine %d\n",best_task,best_machine);
#endif

	// Ordered insertion.
	new_et = mach[best_machine][0]+ETC[best_task][best_machine];

#if DEBUG
printf("new_et: %f\n",new_et);
printf("mach[%d]: [",best_machine);
for(k=0;k<cores[best_machine];k++){
printf("%f ",mach[best_machine][k]);
}
printf("] (SHIFT) -> ");
#endif

	h = 1;
    if ( new_et < mach[best_machine][h]) {
        h = 0;
    } else {
		while((h<cores[best_machine]) && (new_et > mach[best_machine][h])){
			mach[best_machine][h-1] = mach[best_machine][h];
			if (h<cores[best_machine]){
				h++;
			}
		}
		//if (h == cores[best_machine]){
			h--;
		//}
	}

#if DEBUG
printf("mach[%d]: [",best_machine);
for(k=0;k<cores[best_machine];k++){
printf("%f ",mach[best_machine][k]);
}
printf("] -> ETC[%d,%d]=%f, inserto en pos %d valor %f\n",best_task,best_machine,ETC[best_task][best_machine],h,new_et);
#endif

	mach[best_machine][h] = new_et;

#if DEBUG
printf("mach[%d]: [",best_machine);
for(k=0;k<cores[best_machine];k++){
printf("%f ",mach[best_machine][k]);
}
#endif

	asig[best_task] = best_machine;
	//energy_mach[best_machine] += energy(best_task,best_machine,mach[best_machine],cores[best_machine],E_IDLE[best_machine],E_MAX[best_machine],et,napp[best_machine]);
    napp[best_machine]++;
	nro_asig++;

#if DEBUG
printf("] napp: %d\n***************************\n",napp[best_machine]);
#endif
}

float makespan = 0.0;
int heavy = -1;
float mak_local = 0.0;
float total_energy = 0.0;
float energy_m;

for (j=0;j<NM;j++){
	mak_local = mach[j][cores[j]-1];
    if (mak_local > makespan){
        makespan = mak_local;
heavy = j;
    }
}

for (j=0;j<NM;j++){
	energy_m = energy_mach(mach[j],cores[j],E_IDLE[j],E_MAX[j],napp[j]) + E_IDLE[j]*(makespan-mach[j][cores[j]-1]);
    total_energy += energy_m;
printf("M[%d]: mak_local %f energy %f, total energy %f (%d tasks). %d cores, %.2f GFLOPS, E:%.2f-%.2f\n",j,mach[j][cores[j]-1],energy_m,total_energy,napp[j],cores[j],GFLOPS[j],E_IDLE[j],E_MAX[j]);
}

//fprintf(stdout,"heavy: %d\n",heavy);
//for (j=0;j<NM;j++){
//printf("M[%d]: mak_local %f energy consumption %f (%d tareas)\n",j,mach[j],energy_mach[j],napp[j]);
//printf("M[%d]: energy consumption %f (%d tareas)\n",j,energy_mach[j],napp[j]);
//}
//*/

fprintf(stdout,"Makespan: %f energy consumption: %f heavy: %d\n",makespan,total_energy,heavy);

#if 0
printf("[");
for (i=0;i<NT;i++){
    printf("%d ",asig[i]);
}
printf("]\n");

#endif

printf("**********\n");

exit(0);
}

