// PALS for the HCSP (static).
// Parameters : <instance_ETC_file> <TOP_T1> <TOP_T2> <MAX_PALS_STEPS> <THREADS> <SINC_POINTS>
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <pthread.h>
#include <semaphore.h>

#define NO_ASIG -1
#define SIZE_NOM_ARCH 180

#define NT 2048
#define NM 64

#define DEBUG 0
#define DEBUG_T 0
#define P_HEAVY 1.0

// Initialization
#define MCT 0
#define PORC_MACH 0.4

// Global variables 
float ETC[NT][NM];
int   globalRandomMachine[NM];
float global_l_mak[NM];
int   global_sol[NM][NT];
int   global_ntasks[NM];
float global_makespan = FLT_MAX;
float global_heavy_mach = 0.0;

int TOP_T1, TOP_T2, TOP_T3, MAX_STEPS; 
int end = 0;
int best_found = 0;

sem_t S1, S2, S3;

void * PALS();

int main(int argc, char *argv[]) {

if (argc != 8){
        printf("Sintaxis: %s <archivo_instancias> <TOP_T1> <TOP_T2> <TOP_T3> <MAX_PALS_STEPS> <THREADS> <SINC_POINTS>\n", argv[0]);
        exit(1);
}

FILE *fi;
char *arch_inst;

arch_inst = (char *)malloc(sizeof(char)*120);
strcpy(arch_inst,argv[1]);
if((fi=fopen(arch_inst, "r"))==NULL){
    fprintf(stderr,"No se puede leer archivo de instancia %s\n",arch_inst);
    exit(1);
}

TOP_T1 = atoi(argv[2]);
TOP_T2 = atoi(argv[3]);
TOP_T3 = atoi(argv[4]);
MAX_STEPS = atoi(argv[5]);
int numberOfThreads = atoi(argv[6]);
int sinchronizationPoints = atoi(argv[7]);

unsigned int seed=time(NULL);

clock_t t_start, t_end;
double dif2;

t_start = clock();

// Semaphores: 
//		S1: wait (workers).
//		S2: wait (master).
//		S3: for updating global best solution.
sem_init(&S1, 0, 0);
sem_init(&S2, 0, 0);
sem_init(&S3, 0, 1);

// Read input file, store ETC matrix
int i,j,k,h;
for (i=0;i<NT;i++){
        for (j=0;j<NM;j++) {
        	fscanf(fi,"%f",&ETC[i][j]);
        }
}
close(fi);

// Local makespan array, stores the MET for each machine.
for (j=0;j<NM;j++) {
	global_l_mak[j] = 0.0;
	global_ntasks[j] = 0;
}

//generate a rand_rom permutation of the machines
int pos, tmp;
for (j = 0; j < NM; j++)
	globalRandomMachine[j] = j;

for (j = NM-1; j > 0; j--) {
	//pick a rand_rom element [0..j-1] to swap
	pos = rand_r(&seed) % j;
	//swap
	tmp = globalRandomMachine[pos];
	globalRandomMachine[pos] = globalRandomMachine[j];
	globalRandomMachine[j]   = tmp;
}

pthread_t threads[numberOfThreads];
int rc;

//printf("Inicio: TOP_T1: %d, TOP_T2: %d, TOP_T3: %d, MAX_STEPS: %d, numberOfThreads: %d, sinchronizationPoints: %d\n",TOP_T1,TOP_T2,TOP_T3,MAX_STEPS,numberOfThreads,sinchronizationPoints);

// Create the thread pool
for (i = 0; i < numberOfThreads; i ++) {
	rc = pthread_create(&threads[i], NULL, PALS, NULL);
	if (rc) {
		printf("ERROR; return code from pthread_create() is %d\n", rc);
	  exit(-1);
	}
}

for(i=0;i<sinchronizationPoints;i++) {

	// Active model: master performs its own search (to be implemented).

#if DEBUG
printf("step: %d\n",i);
#endif

	// Wait for synchronization
	for (k=0;k<numberOfThreads;k++) {
#if DEBUG
printf("Synchronization, waiting (%d) ....\n",k);
#endif
		sem_wait(&S2);
	}	

#if DEBUG
printf("Best (%d) = (%f)\n", i, global_makespan);
#endif
	
//printf("broadcast c\n");
// pthread_cond_broadcast(&c);
	// Resume work
	for (k=0;k<numberOfThreads;k++) {
#if DEBUG
printf("Continuation signal (%d) ....\n",k);
#endif
		sem_post(&S1);
	}	
} 

t_end = clock();
dif2 = (double)(t_end-t_start)/numberOfThreads;

// Display best solution
int tot_t = 0;
for (j=0; j<NM;j++) {
	#if DEBUG
	printf("%d ",global_ntasks[j]);
	for (i=0;i<global_ntasks[j];i++) {
		printf("%d ",global_sol[j][i]);
	}
	//printf(" (%d tasks) -> local makespan: %f\n",global_ntasks[j],global_l_mak[j]);
	#endif
	tot_t += global_ntasks[j];
	//if (global_l_mak[j] > global_makespan){
	//	global_makespan = global_l_mak[j];
	//}
}

//fprintf(stdout,"(END)->Makespan: %f best_found: %d Time: %f, tot_t: %d\n",global_makespan,best_found,dif2,tot_t);

end = 1;
for (k=0;k<numberOfThreads;k++) {
#if DEBUG
printf("Continuation signal (%d) ....\n",k);
#endif
	sem_post(&S1);
}	

    int task_assignment[NT];
    for (i = 0; i < NM; i++)
    {
        for (j = 0; j < global_ntasks[i]; j++)
        {
            task_assignment[global_sol[i][j]] = i;
        }
    }
    
    fprintf(stdout, "1\n");
    for (i = 0; i < NT; i++) {
        fprintf(stdout, "%d\n", task_assignment[i]);
    }

pthread_exit(NULL);

} // fin main


/********************************************************************************/

void * PALS() {

int sol[NM][NT];
float l_mak[NM];
int ntasks[NM];
float makespan = 0.0;

int i,j,h;

pthread_t id_thread = pthread_self();
unsigned int seed_t = time(NULL)+(unsigned)id_thread;

#if DEBUG
printf("init thread %d: TOP_T1: %d, TOP_T2: %d, TOP_T3: %d, MAX_STEPS: %d\n",(int) id_thread,TOP_T1,TOP_T2,TOP_T3,MAX_STEPS);
#endif

// Local makespan array, stores the MET for each machine.
for (j=0;j<NM;j++) {
	l_mak[j]  = 0.0;
	ntasks[j] = 0;
}

// Initialization
int t, m, best_machine, heavy_mach;

#if MCT

float ct, min_ct;

//Task assignment
for (t = 0; t < NT; t++) {
	best_machine = -1;
	min_ct       = FLT_MAX;
	ct           = 0.0; 

	//look for the best machine
	for (m = 0; m < NM; m++) {
		ct = l_mak[m] + ETC[t][m];
		if (ct < min_ct) {
			min_ct = ct;
			best_machine = m;
		}
	}
	
	//assign tast "t" to machine "best_machine"
	// 1.- Update the local makespan of machine "best_machine"
	l_mak[best_machine] += ETC[t][best_machine];
	// 2.- Retrieve and update the first free slot of "best_machine"
	int freeSlot = ntasks[best_machine];
	ntasks[best_machine] ++;
	// 3.- Deploy the assignment
	sol[best_machine][freeSlot] = t;
}

#else
// MCT rand
int startingRandomMachine;
float ct, min_ct;

//Task assignment
for (t = 0; t < NT; t++) {
	best_machine = -1;
	min_ct       = FLT_MAX;
	ct           = 0.0; 
  
	//look for the best machine
	startingRandomMachine = rand_r(&seed_t) % NM;
	for (j = 0; j < (int)NM*PORC_MACH; j++) {
		m = globalRandomMachine[(startingRandomMachine + j) % NM];
		ct = l_mak[m] + ETC[t][m];
		if (ct < min_ct) {
			min_ct = ct;
			best_machine = m;
		}
	}
	
	//assign tast "t" to machine "best_machine"
	// 1.- Update the local makespan of machine "best_machine"
	l_mak[best_machine] += ETC[t][best_machine];
	// 2.- Retrieve and update the first free slot of "best_machine"
	int freeSlot = ntasks[best_machine];
	ntasks[best_machine] ++;
	// 3.- Deploy the assignment
	sol[best_machine][freeSlot] = t;
}

// for (j = 0; j < NM; j++) printf("m[%d] = %d\n", j, ntasks[j]);
#endif

//generate a random permutation of the machines within the thread 
int randomMachine[NM];
int pos, tmp;
for (j = 0; j < NM; j++) {
	randomMachine[j] = j;
}

int init = 1;

do {

	for (j = NM-1; j > 0; j--) {
		//pick a random element [0..j-1] to swap
		pos = rand_r(&seed_t) % j;
		//swap
		tmp = randomMachine[pos];
		randomMachine[pos] = randomMachine[j];
		randomMachine[j]   = tmp;
	}

	if (init == 0){

        //pthread_mutex_lock(&mut);

sem_wait(&S3);

#if DEBUG_T
int tot_t=0;
#endif
		//copy the global makespan and numberOfAssigned tasks into the local one
		for (j=0;j<NM;j++) {
			l_mak[j] = global_l_mak[j];
			ntasks[j] = global_ntasks[j];
#if DEBUG_T
printf("Leida sol global: ntasks[%d] = %d, l_mak = %f\n",j,ntasks[j],l_mak[j]);
tot_t+=ntasks[j];
#endif
		}
#if DEBUG_T
printf("ntasks: %d\n",tot_t);
#endif

		// retrieve the global solution
		for (i = 0; i < NM; i++) {
			for (j = 0; j < global_ntasks[i]; j++) {
				sol[i][j] = global_sol[i][j];
			}
		}

		makespan = global_makespan;
		heavy_mach = global_heavy_mach;

#if DEBUG_T
printf("Leida sol global: makespan = %f\n",makespan);
#endif

sem_post(&S3);

        //pthread_mutex_unlock(&mut);

	} else {
		for (j=0;j<NM;j++) {
			if (l_mak[j] > makespan){
				makespan = l_mak[j];
				heavy_mach = j;
			}
#if DEBUG_T
printf("Sol inicial: ntasks[%d] = %d, l_mak = %f\n",j,ntasks[j],l_mak[j]);
#endif
		}
#if DEBUG_T
printf("Sol inicial: makespan = %f\n",makespan);
#endif
#if DEBUG
printf("thread %d: makespan inicial: %f\n",(int) id_thread,makespan);
#endif
	}

	init = 0;

	int best_machine, best_task, best_mach_task; //, best_found = 0;
	
	int best_task_swap, best_swap,best_source_m;

	float delta_best, new_mak_target_m, new_mak_source_m;
	float delta_task, best_delta_task, best_machine_swap, operator_type;
	int temp;

	int steps = 0, indexRandomMachine = 0, move = 0;
	int randomT1, randomT2, randomT3;
	int targetM, sourceM, targetT, sourceT;
	float p;

	int accept = 0;

	float old_makespan = 0.0;

	do {
        delta_best = makespan;
        best_task = -1;
        operator_type = -1;
        
        // Randomize source machine
        p = (float) rand_r(&seed_t)/(1.0 + RAND_MAX);
        if ( p < P_HEAVY) {
        	do {
         		sourceM = rand_r(&seed_t)%NM;
         	} while (ntasks[sourceM] == 0);
        } 
        else {
          sourceM = heavy_mach;
        }
      
      	randomT1 = rand_r(&seed_t) % TOP_T1;

        for(h=0;h<randomT1; h++) {
			// Iterate on machine "sourceM" from task "sourceT"
			//sourceT = rand() % (numberOfAssignedTasks[sourceM]/2);
#if DEBUG
//printf("numberOfAssignedTasks[%d] = %d\n", sourceM, ntasks[sourceM]);
#endif

			if (h == 0){
				sourceT = rand_r(&seed_t) % ntasks[sourceM];
			} else {
				sourceT = (sourceT + 1) % ntasks[sourceM];
			}
                
			best_delta_task = makespan;
			best_task_swap = -1;
                
			//randomly choose the target machine
			do {
				targetM = randomMachine[indexRandomMachine];
				indexRandomMachine =  (indexRandomMachine + 1) % NM;
			} while ((targetM == sourceM) || (ntasks[targetM] == 0));
                                
//printf("it %d -> procesando máquina %d, tarea %d; probando tareas de la máquina %d\n", h, sourceM, sourceT, targetM);
                
			//randomly choose task "targetT" on machine "targetM" to start iterating
			targetT = rand_r(&seed_t) % ntasks[targetM];
			int index;
			int taskIndex = 0;
                
			p = (float) rand_r(&seed_t)/(1.0 + RAND_MAX);
			if ( p < 0.75) {
				move = 0;
			} else {
				move = 1;
			}
         		    	
			if (move == 0) { 
				//TASK SWAPPING
				randomT2 = rand_r(&seed_t) % TOP_T2;
				for (index = 0; index < randomT2; index++) {
					//evaluate makespan variation
					int t = (targetT+taskIndex);
		            //look for next available tasks
		            while (t == ntasks[targetM]) {
						//choose a new targetM
						do {
							targetM = randomMachine[rand_r(&seed_t) % NM];
						} while (ntasks[targetM] == 0);
		              		
		              	//choose a new targetT
		              	targetT = rand_r(&seed_t) % ntasks[targetM];
		              	taskIndex = 0;
		              	t = (targetT+taskIndex);
					}
		              	
		            new_mak_source_m = l_mak[sourceM] - ETC[sol[sourceM][sourceT]][sourceM] + ETC[sol[targetM][t]][sourceM];
		            new_mak_target_m = l_mak[targetM] - ETC[sol[targetM][t]][targetM] + ETC[sol[sourceM][sourceT]][targetM] ;
		              	                   
		            //check for global_makespan
		            if ( (new_mak_target_m < new_mak_source_m) && ( new_mak_source_m < best_delta_task) ) {
						best_task_swap    = t;
						best_machine_swap = targetM;
						best_delta_task   = new_mak_source_m;
#if DEBUG
printf("\t Swap: improvement on source (%d, %d) -> %f\n", targetM, t, best_delta_task);
#endif
					} else {
						if ( (new_mak_source_m < new_mak_target_m) && ( new_mak_target_m < best_delta_task ) ) {
							best_task_swap    = t;
							best_machine_swap = targetM;
							best_delta_task = new_mak_target_m;
#if DEBUG
printf("\t Swap: improvement on target (%d, %d) -> %f\n", targetM, t, best_delta_task);
#endif
		              	}
					}
					taskIndex++;
				}

				if (best_delta_task < delta_best) {
		           	delta_best   = best_delta_task;
		            best_task    = sourceT;
		            best_source_m = sourceM;
		            best_swap    = best_task_swap;
		            best_machine = best_machine_swap;
		            operator_type = 0;
#if DEBUG
printf("\tSwap -> best_machine = %d; best_swap = %d\n", best_machine, best_swap);
#endif
				}
			} else { 
				//TASK MOVEMENT
				randomT3 = rand_r(&seed_t) % TOP_T3;
				for (index = 0; index < randomT3; index++) {
		           	//evaluate makespan variation
		          	int t = (targetT+taskIndex);
		              	
		           	//look for next available tasks
		           	while (t == ntasks[targetM]) {
		           		//choose a new targetM
		           		do {
		           		  targetM = randomMachine[rand_r(&seed_t) % NM];
		           		} while (ntasks[targetM] == 0);
		              		
		           		//choose a new targetT
		           		targetT = rand_r(&seed_t) % ntasks[targetM];
		           		taskIndex = 0;
		           		t = (targetT+taskIndex);
		           	}
		              	
		           	new_mak_source_m = l_mak[sourceM] + ETC[sol[targetM][t]][sourceM]; 
					new_mak_target_m = l_mak[targetM] - ETC[sol[targetM][t]][targetM];
		              	                   
		            //check for global_makespan
		            if ( (new_mak_target_m < new_mak_source_m) && ( new_mak_source_m < best_delta_task) ) {
		            	best_task_swap    = t;
						best_machine_swap = targetM;
						best_delta_task   = new_mak_source_m;
#if DEBUG
printf("\t Move: improvement on source (%d, %d) -> %f\n", targetM, t, best_delta_task);
#endif
					} else {
						if ( (new_mak_source_m < new_mak_target_m) && ( new_mak_target_m < best_delta_task ) ) {
							best_task_swap    = t;
							best_machine_swap = targetM;
							best_delta_task = new_mak_target_m;
#if DEBUG
printf("\t Move: improvement on target (%d, %d) -> %f\n", targetM, t, best_delta_task);
#endif
		              	}
					}
					taskIndex++;
				}
				if (best_delta_task < delta_best) {
					delta_best   = best_delta_task;
		            best_task    = sourceT;
		            best_source_m = sourceM;
		            best_swap    = best_task_swap;
		            best_machine = best_machine_swap;
		            operator_type = 1;
#if DEBUG
printf("\tMove -> best_machine = %d; best_swap = %d\n", best_machine, best_swap);
#endif
				}
			}
		}
        
		old_makespan = makespan;

        if ((best_task != -1) && (operator_type == 0)) {        
//printf("Swapeando tareas: %d de mach %d (%d tasks,l_mak=%f) a mach %d (%d tasks,l_mak=%f)\n",best_swap,best_machine,ntasks[best_machine],l_mak[best_machine],best_source_m,ntasks[best_source_m],l_mak[best_source_m]);
        	l_mak[best_source_m] = l_mak[best_source_m]-ETC[sol[best_source_m][best_task]][best_source_m]+ETC[sol[best_machine][best_swap]][best_source_m];
        	l_mak[best_machine] = l_mak[best_machine]-ETC[sol[best_machine][best_swap]][best_machine]+ETC[sol[best_source_m][best_task]][best_machine];
        	tmp = sol[best_source_m][best_task];
        	sol[best_source_m][best_task] = sol[best_machine][best_swap];
        	sol[best_machine][best_swap] = tmp;
//printf("Swapeadas tareas: %d de mach %d (%d tasks,l_mak=%f) a mach %d (%d tasks,l_mak=%f)\n",best_swap,best_machine,ntasks[best_machine],l_mak[best_machine],best_source_m,ntasks[best_source_m],l_mak[best_source_m]);
          	makespan = 0.0;
          	for (j=0;j<NM;j++) {
				if (l_mak[j]>makespan) {
					makespan = l_mak[j];
					heavy_mach = j;
				}
			}
//printf("New makespan: %f\n",makespan);
		} else {
        	if ((best_task != -1) && (operator_type == 1)) {
//printf("Moviendo tarea %d de mach %d (%d tasks,l_mak=%f) a mach %d (%d tasks,l_mak=%f)\n",best_swap,best_machine,ntasks[best_machine],l_mak[best_machine],best_source_m,ntasks[best_source_m],l_mak[best_source_m]);
		      	l_mak[best_source_m]= l_mak[best_source_m]+ETC[sol[best_machine][best_swap]][best_source_m];
		      	l_mak[best_machine] = l_mak[best_machine]-ETC[sol[best_machine][best_swap]][best_machine];
		      	int freeSlot = ntasks[best_source_m];
		      	ntasks[best_source_m]++;
		      	sol[best_source_m][freeSlot] = sol[best_machine][best_swap];
		      	//compactar best_machine tasks
		      	i = best_swap;
		      	while ((i < ntasks[best_machine]) && (i < NT)) {
		      		sol[best_machine][i] = sol[best_machine][i+1];
		      		i++;
		      	}
		      	ntasks[best_machine]--;

if (ntasks[best_machine] == 0){
l_mak[best_machine] = 0.0;
}

//printf("Movida tarea %d de mach %d (%d tasks,l_mak=%f) a mach %d (%d tasks,l_mak=%f)\n",best_swap,best_machine,ntasks[best_machine],l_mak[best_machine],best_source_m,ntasks[best_source_m],l_mak[best_source_m]);
		      	
		      	//sorteo de una nueva máquina sourceM
		      	do {
		        	sourceM = randomMachine[rand_r(&seed_t) % NM];
		        } while (ntasks[sourceM] == 0);
		      	
		        makespan = 0.0;
		        for (j=0;j<NM;j++) {
					if (l_mak[j]>makespan) {
						makespan = l_mak[j];
						heavy_mach = j;
					}			
		        }
//printf("New makespan: %f\n",makespan);
        	}
        }

#if DEBUG
int t_task = 0;
for(i=0;i<NM;i++){
	t_task+=ntasks[i];
}
printf("NUM_TASKS: %d\n",t_task);
#endif

        if (makespan < old_makespan){
			best_found = steps;
			accept++;
//fprintf(stdout,"%d: Makespan: %f (heavy: %d)\n",steps,global_makespan,heavy_mach);
//system("sleep 1");
		}

		steps++;

	} while(steps < MAX_STEPS);

#if DEBUG
printf("thread %d, end.\n",(int) id_thread);
#endif

	// Update global solution, if applies.
	if (makespan < global_makespan){
#if DEBUG
printf("thread %d, update global\n",(int) id_thread);
#endif

		sem_wait(&S3);

#if DEBUG_T
int tot_t = 0;
#endif
		//copy the global makespan and numberOfAssigned tasks into the local one
		for (j=0;j<NM;j++) {
			global_l_mak[j] = l_mak[j];
			global_ntasks[j] = ntasks[j];
#if DEBUG_T
tot_t+=ntasks[j];
printf("Actualizada sol global: global_ntasks[%d] = %d, global_l_mak = %f\n",j,global_ntasks[j],global_l_mak[j]);
#endif
		}
#if DEBUG_T
printf("ntasks: %d\n",tot_t);
#endif
		// update the global solution
		for (i = 0; i < NM; i++) {
			for (j = 0; j < global_ntasks[i]; j++) {
				global_sol[i][j] = sol[i][j];
			}
		}
		global_heavy_mach = heavy_mach;
		global_makespan = makespan;

#if DEBUG_T
printf("Actualizada sol global: makespan = %f\n",global_makespan);
#endif
		sem_post(&S3);
	}

#if DEBUG
fprintf(stdout,"(%d)->Makespan: %f, best_found: %d, accept: %d\n",(int) id_thread,makespan,best_found,accept);
printf("thread %d, sem_post S2\n",(int) id_thread);
#endif

	sem_post(&S2);

#if DEBUG
printf("thread %d, sem_wait S1\n",(int) id_thread);
#endif
	sem_wait(&S1);

#if DEBUG
printf("thread %d, sigue\n",(int) id_thread);
#endif

} while (!end);

pthread_exit(NULL);

}
