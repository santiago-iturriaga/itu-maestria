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

int main(int argc, char *argv[])
{

	if (argc < 5)
	{
		//printf("Sintaxis: %s <workload index> <scenario index> <num_tasks> <num_machines> <etc_model: 0-Zomaya, 1-Braun>\n", argv[0]);
		printf("Sintaxis: %s <workload index> <scenario index> <num_tasks> <num_machines> \n", argv[0]);
		exit(1);
	}

	int NT, NM;
	FILE *fi, *fp;
	char *arch_inst, *arch_proc;

	NT = atoi(argv[3]);
	NM = atoi(argv[4]);

	arch_inst = argv[1];
	arch_proc = argv[2];

	#if DEBUG
	fprintf(stdout,"NT: %d, NM: %d, arch_ETC: %s, arch_proc: %s\n",NT,NM,arch_inst,arch_proc);
	#endif

	float *E_IDLE = (float *) malloc(sizeof(float)*NM);
	if (E_IDLE == NULL)
	{
		fprintf(stderr,"Error in malloc for E_IDLE matrix, dimension %d\n",NM);
		exit(2);
	}

	float *E_MAX = (float *) malloc(sizeof(float)*NM);
	if (E_MAX == NULL)
	{
		fprintf(stderr,"Error in malloc for E_MAX matrix, dimension %d\n",NM);
		exit(2);
	}

	float *GFLOPS = (float *) malloc(sizeof(float)*NM);
	if (GFLOPS == NULL)
	{
		fprintf(stderr,"Error in malloc for GFLOPS matrix, dimension %d\n",NM);
		exit(2);
	}

	float **ETC = (float **) malloc(sizeof(float *)*NT);
	if (ETC == NULL)
	{
		fprintf(stderr,"Error in malloc for ETC matrix, dimensions %dx%d\n",NT,NM);
		exit(2);
	}

	int i,j,h,k;

	for (i=0;i<NT;i++)
	{
		ETC[i] = (float *) malloc(sizeof(float)*NM);
		if (ETC[i] == NULL)
		{
			fprintf(stderr,"Error in malloc, row %d in ETC\n",i);
			exit(2);
		}
	}

	// Machine array, stores the MET.
	float *mach = (float *) malloc(sizeof(float)*NM);
	if (mach == NULL)
	{
		fprintf(stderr,"Error in malloc (machine array), dimension %d\n",NM);
		exit(2);
	}

	// Read input files, store ETC matrix and proc. info

	if((fp=fopen(arch_proc, "r"))==NULL)
	{
		fprintf(stderr,"Can't read processor file: %s\n",arch_inst);
		exit(1);
	}

	int cores;
	for (j=0;j<NM;j++)
	{
		fscanf(fp,"%d %f %f %f\n",&cores,&GFLOPS[j],&E_IDLE[j],&E_MAX[j]);
	}

	for (j=0;j<NM;j++)
	{
		mach[j] = 0.0;
	}

	close(fp);

	if((fi=fopen(arch_inst, "r"))==NULL)
	{
		fprintf(stderr,"Can't read instance file: %s\n",arch_inst);
		exit(1);
	}

	for (i=0;i<NT;i++)
	{
		for (j=0;j<NM;j++)
		{
			fscanf(fi,"%f",&ETC[i][j]);
			ETC[i][j] = ETC[i][j] / GFLOPS[j];
		}
	}

	close(fi);

	// Number of applications array
	int *napp = (int*) malloc(sizeof(float)*NM);
	if (napp == NULL)
	{
		fprintf(stderr,"Error in malloc (number of applications array), dimension %d\n",NM);
		exit(2);
	}

	for (j=0;j<NM;j++)
	{
		napp[j] = 0;
	}

	// Assigned tasks array

	int *asig= (int*) malloc(sizeof(float)*NT);
	if (asig == NULL)
	{
		fprintf(stderr,"Error in malloc (assigned tasks array), dimension %d\n",NT);
		exit(2);
	}

	int nro_asig = 0;
	for (i=0;i<NT;i++)
	{
		asig[i]=NO_ASIG;
	}

	float mct_i_j;
	float min_ct, min_ct_task;

	int best_machine, best_mach_task;
	int best_task;

	float et, new_et;

	while (nro_asig < NT)
	{
		// Select non-assigned tasks with maximun robustness radio - minimum completion time.
		best_task = -1;
		best_machine = -1;
		min_ct = FLT_MAX;

		// Loop on tasks.
		for (i=0;i<NT;i++)
		{
			best_mach_task = -1;
			min_ct_task = FLT_MAX;

			if (asig[i] == NO_ASIG)
			{
				// Loop on machines
				for (j=0;j<NM;j++)
				{
					// Evaluate MCT of (ti, mj)
					// mach[j][0] has the min local makespan for machine j.
					et = mach[j]+ETC[i][j];
					if (et < min_ct_task)
					{
						min_ct_task = et;
						best_mach_task = j;
					}
				}
				//mct_i_j = mach[best_mach_task][0]+ETC[i][best_mach_task];

				if (min_ct_task <= min_ct)
				{
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
		new_et = mach[best_machine]+ETC[best_task][best_machine];

		#if DEBUG
		printf("new_et: %f\n",new_et);
		#endif

		mach[best_machine] = new_et;

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

	for (j=0;j<NM;j++)
	{
		mak_local = mach[j];
		if (mak_local > makespan)
		{
			makespan = mak_local;
			heavy = j;
		}
	}

	for (j=0;j<NM;j++)
	{
		energy_m = (mach[j] * E_MAX[j]) + (E_IDLE[j] * (makespan-mach[j]));
		total_energy += energy_m;
		#if DEBUG
		printf("M[%d]: mak_local %f energy %f, total energy %f (%d tasks). %.2f GFLOPS, E:%.2f-%.2f\n",
			j,mach[j],energy_m,total_energy,napp[j],GFLOPS[j],E_IDLE[j],E_MAX[j]);
		#endif
	}

	//fprintf(stdout,"heavy: %d\n",heavy);
	//for (j=0;j<NM;j++){
	//printf("M[%d]: mak_local %f energy consumption %f (%d tareas)\n",j,mach[j],energy_mach[j],napp[j]);
	//printf("M[%d]: energy consumption %f (%d tareas)\n",j,energy_mach[j],napp[j]);
	//}
	//*/

	#if DEBUG
	fprintf(stdout,"Makespan: %f energy consumption: %f heavy: %d\n",makespan,total_energy,heavy);
	#endif

	fprintf(stdout,"%f|%f\n",makespan,total_energy);

	#if 0
	printf("[");
	for (i=0;i<NT;i++)
	{
		printf("%d ",asig[i]);
	}
	printf("]\n");
	#endif

	#if DEBUG
	printf("**********\n");
	#endif

	exit(0);
}
