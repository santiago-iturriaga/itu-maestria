// MIN-Min energy-aware scheduler.
// Phase 1: Pair with minimum completion time
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

int
main (int argc, char *argv[])
{

	if (argc < 5)
	{
		//printf("Sintaxis: %s <workload index> <scenario index> <num_tasks> <num_machines> <etc_model: 0-Zomaya, 1-Braun>\n", argv[0]);
		printf
			("Sintaxis: %s <workload> <scenario> <num_tasks> <num_machines> \n",
			argv[0]);
		exit (1);
	}

	int NT, NM;
	FILE *fi, *fp;
	char *arch_inst, *arch_proc;

	NT = atoi (argv[3]);
	NM = atoi (argv[4]);

	arch_inst = argv[1];
	arch_proc = argv[2];

	#if DEBUG
	fprintf (stdout, "NT: %d, NM: %d, arch_ETC: %s, arch_proc: %s\n", NT, NM,
		arch_inst, arch_proc);
	#endif

	float *E_IDLE = (float *) malloc (sizeof (float) * NM);
	if (E_IDLE == NULL)
	{
		fprintf (stderr, "Error in malloc for E_IDLE matrix, dimension %d\n",
			NM);
		exit (2);
	}

	float *E_MAX = (float *) malloc (sizeof (float) * NM);
	if (E_MAX == NULL)
	{
		fprintf (stderr, "Error in malloc for E_MAX matrix, dimension %d\n",
			NM);
		exit (2);
	}

	float *GFLOPS = (float *) malloc (sizeof (float) * NM);
	if (GFLOPS == NULL)
	{
		fprintf (stderr, "Error in malloc for GFLOPS matrix, dimension %d\n",
			NM);
		exit (2);
	}

	int *CORES = (int *) malloc(sizeof(int)*NM);
	if (CORES == NULL)
	{
		fprintf(stderr,"Error in malloc for CORES matrix, dimension %d\n",NM);
		exit(2);
	}

	float **ETC = (float **) malloc (sizeof (float *) * NT);
	if (ETC == NULL)
	{
		fprintf (stderr, "Error in malloc for ETC matrix, dimensions %dx%d\n",
			NT, NM);
		exit (2);
	}

	int i, j, h, k;

	for (i = 0; i < NT; i++)
	{
		ETC[i] = (float *) malloc (sizeof (float) * NM);
		if (ETC[i] == NULL)
		{
			fprintf (stderr, "Error in malloc, row %d in ETC\n", i);
			exit (2);
		}
	}

	// Machine array, stores the MET.
	float *mach = (float *) malloc (sizeof (float) * NM);
	if (mach == NULL)
	{
		fprintf (stderr, "Error in malloc (machine array), dimension %d\n", NM);
		exit (2);
	}

	// Read input files, store ETC matrix and proc. info

	if ((fp = fopen (arch_proc, "r")) == NULL)
	{
		fprintf (stderr, "Can't read processor file: %s\n", arch_inst);
		exit (1);
	}

	for (j=0;j<NM;j++)
	{
		fscanf(fp,"%d %f %f %f\n",&CORES[j],&GFLOPS[j],&E_IDLE[j],&E_MAX[j]);
	}

	for (j = 0; j < NM; j++)
	{
		mach[j] = 0.0;
	}

	fclose (fp);

	if ((fi = fopen (arch_inst, "r")) == NULL)
	{
		fprintf (stderr, "Can't read instance file: %s\n", arch_inst);
		exit (1);
	}

	for (i=0;i<NT;i++)
	{
		for (j=0;j<NM;j++)
		{
			fscanf(fi,"%f",&ETC[i][j]);
			ETC[i][j] = ETC[i][j] / (GFLOPS[j] / (CORES[j] * 1000));
		}
	}

	fclose (fi);

	float *energy_mach = (float *) malloc (sizeof (float) * NM);
	if (energy_mach == NULL)
	{
		fprintf (stderr, "Error in malloc (energy_mach), dimension %d\n", NM);
		exit (2);
	}

	// Number of applications array
	int *napp = (int *) malloc (sizeof (float) * NM);
	if (napp == NULL)
	{
		fprintf (stderr,
			"Error in malloc (number of applications array), dimension %d\n",
			NM);
		exit (2);
	}

	for (j = 0; j < NM; j++)
	{
		napp[j] = 0;
		energy_mach[j] = 0.0;
	}

	// Assigned tasks array

	int *asig = (int *) malloc (sizeof (float) * NT);
	if (asig == NULL)
	{
		fprintf (stderr,
			"Error in malloc (assigned tasks array), dimension %d\n", NT);
		exit (2);
	}

	int nro_asig = 0;
	for (i = 0; i < NT; i++)
	{
		asig[i] = NO_ASIG;
	}

	float mct_i_j;
	float min_ct;

	int best_machine, best_mach_task;
	int best_task;

	float ct, delta_energy, min_energy, new_et;

	float mak_temp = 0.0;

	while (nro_asig < NT)
	{
		// Select non-assigned tasks with maximun robustness radio - minimum completion time.
		best_task = -1;
		best_machine = -1;
		min_energy = FLT_MAX;

		// Loop on tasks.
		for (i = 0; i < NT; i++)
		{
			best_mach_task = -1;
			min_ct = FLT_MAX;

			if (asig[i] == NO_ASIG)
			{
				// Loop on machines
				for (j = 0; j < NM; j++)
				{
					// Evaluate MCT of (ti, mj)
					// mach[j] has the makespan for machine j.
					mct_i_j = mach[j] + ETC[i][j];
					#if DEBUG
					printf ("mct[%d,%d]:%f -", i, j, mct_i_j);
					#endif
					if (mct_i_j <= min_ct)
					{
						min_ct = mct_i_j;
						best_mach_task = j;
						#if DEBUG
						printf ("MCT best pair(%d,%d): %f\n", i, best_mach_task,
							min_ct);
						#endif
					}
				}
				#if DEBUG
				printf ("End loop machines: MCT best pair(%d,%d):%f\n", i,
					best_mach_task, min_ct);
				#endif
			
				if (mct_i_j <= mak_temp) {
				    delta_energy = (E_MAX[best_mach_task] - E_IDLE[best_mach_task]) * ETC[i][best_mach_task];
				} else {
				    delta_energy = 0.0;
			        float mak_diff = mct_i_j - mak_temp;
			        
				    for (int aux_j = 0; aux_j < NM; aux_j++) {
				        if (aux_j == best_mach_task) {
                            delta_energy += (ETC[i][best_mach_task] * E_MAX[best_mach_task]) - (mak_diff * E_IDLE[best_mach_task]);
				        } else {
                            delta_energy += (mak_diff * E_IDLE[aux_j]);
				        }
				    }
                }

				#if DEBUG
				printf ("delta energy (%d,%d): %f\n", i, best_mach_task,
					delta_energy);
				#endif
				if (delta_energy <= min_energy)
				{
					min_energy = delta_energy;
					best_task = i;
					best_machine = best_mach_task;
					// new_et = min_ct;
					#if DEBUG
					printf ("delta energy best ! (%d,%d): %f\n", best_task,
						best_machine, delta_energy);
					#endif
				}
			}
		}

		#if DEBUG
		printf
			("********************* Assign task %d to machine %d (energy: %f)\n",
			best_task, best_machine, min_energy);
		#endif

		// Ordered insertion.
		new_et = mach[best_machine] + ETC[best_task][best_machine];

		#if DEBUG
		printf ("new_et: %f\n", new_et);
		#endif

		mach[best_machine] = new_et;

		asig[best_task] = best_machine;
		energy_mach[best_machine] = energy_mach[best_machine] + min_energy;
		napp[best_machine] += 1;
		nro_asig++;
		
		#if DEBUG
		printf ("energy[%d]:%f, napp[%d]:%d***************************\n",
			best_machine, energy_mach[best_machine], best_machine,
			napp[best_machine]);
		#endif

		/* Update energy */
		float delta_mak = 0.0;

		if (mach[best_machine] > mak_temp)
		{
			delta_mak = mach[best_machine] - mak_temp;
			#if DEBUG
			printf ("Delta mak: %f\n", delta_mak);
			printf ("Update mak_temp: %f ", mak_temp);
			#endif
			mak_temp = mach[best_machine];
			#if DEBUG
			printf ("-> %f\n", mak_temp);
			#endif
			for (j = 0; j < NM; j++)
			{
				#if DEBUG
				printf ("Update energy[%d]:%f ", j, energy_mach[j]);
				#endif
				//energy_mach[j]=energy_mach[j]+E_IDLE[j]*(mak_temp-mach[j][cores[j]-1]);
				if (j != best_machine)
				{
					energy_mach[j] = energy_mach[j] + E_IDLE[j] * delta_mak;
				}
				#if DEBUG
				printf ("%f\n", j, energy_mach[j]);
				#endif
			}
		}

	}

	float makespan = 0.0;
	int heavy = -1;
	float mak_local = 0.0;
	float total_energy = 0.0;

	for (j = 0; j < NM; j++)
	{
		mak_local = mach[j];
		if (mak_local > makespan)
		{
			makespan = mak_local;
			heavy = j;
		}
	}

	for (j = 0; j < NM; j++)
	{
		total_energy =
			total_energy + (mach[j] * E_MAX[j]) + ((makespan - mach[j]) * E_IDLE[j]);

		#if DEBUG
		printf
			("M[%d]: mak_local %f energy %f, total energy %f (%d tasks). %.2f GFLOPS, E:%.2f-%.2f\n",
			j, mach[j][cores[j] - 1], energy_mach[j], total_energy, napp[j],
			GFLOPS[j], E_IDLE[j], E_MAX[j]);
		#endif

	}

	#if DEBUG
	fprintf (stdout, "Makespan: %f energy consumption: %f heavy: %d\n",
		makespan, total_energy, heavy);
	#endif

	fprintf (stdout, "%f %f\n", makespan, total_energy);

	#if DEBUG
	fprintf (stdout, "[");
	for (i = 0; i < NT; i++)
	{
		fprintf (stdout, "%d ", asig[i]);
	}
	fprintf (stdout, "]\n");
	#endif
}
