// Min-MIN energy-aware scheduler.
// Phase 1: Pair with minimum energy consumption
// Phase 2: Minimum energy.
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
		printf("Sintaxis: %s <workload> <scenario> <num_tasks> <num_machines> \n", argv[0]);
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

	float *energy_mach = (float*) malloc(sizeof(float)*NM);
	if (energy_mach == NULL)
	{
		fprintf(stderr,"Error in malloc (energy_mach), dimension %d\n",NM);
		exit(2);
	}

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
		energy_mach[j] = 0.0;
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

	int best_machine, best_mach_task;
	int best_task;

	float ct, delta_energy, min_energy, min_energy_task, best_delta_energy, et, new_et;

	float mak_temp = 0.0;

	while (nro_asig < NT)
	{
		// Select non-assigned tasks with maximun robustness radio - minimum completion time.
		best_task = -1;
		best_machine = -1;
		min_energy = FLT_MAX;

		// Loop on tasks.
		for (i=0;i<NT;i++)
		{
			best_mach_task = -1;
			min_energy_task = FLT_MAX;

			#if DEBUG
			printf("Task %d\n",i);
			#endif
			if (asig[i] == NO_ASIG)
			{
				// Loop on machines
				for (j=0;j<NM;j++)
				{
					// Evaluate delta energy of (ti, mj)
					// mach[j] has the makespan for machine j.
					et = mach[j]+ETC[i][j];
					delta_energy = ETC[i][j] * E_MAX[j];
					
					#if DEBUG
					printf("energy_mach[%d]: %f, delta energy (%d,%d):%f -",j,energy_mach[j],i,j,delta_energy);
					#endif
					
					if (delta_energy < min_energy_task)
					{
						min_energy_task = delta_energy;
						best_mach_task = j;
						#if DEBUG
						printf("delta energy best pair(%d,%d): %f\n",i,best_mach_task,delta_energy);
						#endif
					}
				}

				if (min_energy_task <= min_energy)
				{
					min_energy = min_energy_task;
					best_task = i;
					best_machine = best_mach_task;
					best_delta_energy = min_energy_task;
					#if DEBUG
					printf("min_energy (%d,%d): %f, best_delta_energy: %f\n",best_task,best_machine,min_energy,best_delta_energy);
					#endif
				}
			}
		}

		#if DEBUG
		printf("********************* Assign task %d to machine %d\n",best_task,best_machine);
		#endif

		// Ordered insertion.
		new_et = mach[best_machine]+ETC[best_task][best_machine];
		mach[best_machine] = new_et;

		asig[best_task]=best_machine;
		energy_mach[best_machine]+=best_delta_energy;
		napp[best_machine]++;
		nro_asig++;

		#if DEBUG
		printf("] napp: %d, energy: %f\n***************************\n",napp[best_machine],energy_mach[best_machine]);
		#endif

		/* Update energy */
		float delta_mak = 0.0;

		if (mach[best_machine] > mak_temp)
		{
			delta_mak = mach[best_machine] - mak_temp;
			mak_temp = mach[best_machine];

			for (j=0;j<NM;j++)
			{
				if ( j != best_machine )
				{
					energy_mach[j] = energy_mach[j]+(E_IDLE[j]*delta_mak);
				}
			}
		}

	}

	float makespan = 0.0;
	int heavy = -1;
	float mak_local = 0.0;
	float total_energy = 0.0;

	for (j=0;j<NM;j++)
	{
		mak_local = mach[j];
		if (mak_local > makespan)
		{
			makespan = mak_local;
			heavy = j;
		}
		total_energy += energy_mach[j];
		#if DEBUG
		printf("M[%d]: mak_local %f energy %f, total energy %f (%d tasks). %.2f GFLOPS, E:%.2f-%.2f\n",j,mak_local,energy_mach[j],total_energy,napp[j],GFLOPS[j],E_IDLE[j],E_MAX[j]);
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

	#if DEBUG
	fprintf(stdout,"[");
	for (i=0;i<NT;i++)
	{
		fprintf(stdout,"%d ",asig[i]);
	}
	fprintf(stdout,"]\n");
	#endif
}
