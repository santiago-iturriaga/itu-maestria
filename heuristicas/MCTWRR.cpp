// Min-min scheduler.
// Parameters : <instance_ETC_file> <num_tasks> <num_machines>
//	

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#define NO_ASIG -1
#define ASIG 0
#define SIZE_NOM_ARCH 180
#define DEBUG 0

int main(int argc, char *argv[]) {
	if (argc < 2) {
		printf("Sintaxis: %s <archivo_instancias>\n", argv[0]);
		exit(1);
	}

	int NT; 			// Cantidad de tareas.
	int NM; 			// Cantidad de máquinas.
	int *Priority;		// Array de prioridades de tareas.
	float **ETC;		// Matriz de ETC (Tarea x Máquina).
	float *mach;		// Makespan de cada máquina.
	int *asig;			// Array que marca el estado de cada tareas (ASIG/NO_ASIG).
	int **asignMatrix;  // Matriz de asignación (Máquina x Tarea).
	int *lastAsig;		// Array con la cantidad de tareas asignadas a cada máquina.

	{
		char *arch_inst;
		arch_inst = (char *) malloc(sizeof(char) * 120);
		strcpy(arch_inst, argv[1]);

		// [INICIO] Abro el archivo de instancia. ==================================================
		FILE *fi;
		if ((fi = fopen(arch_inst, "r")) == NULL) {
			fprintf(stderr, "No se puede leer archivo de instancia %s\n",
					arch_inst);
			exit(-1);
		}

		// Cargo el tamaño de la instancia de la primer línea del archivo de texto. ================
		fscanf(fi, "%d %d", &NT, &NM);
		fprintf(stdout, "NT: %d, NM: %d\n", NT, NM);

		// Cargo la la lista de prioridades de las tareas. =========================================
		int i, j;
		Priority = (int *) malloc(sizeof(int) * NT);

		for (i = 0; i < NT; i++) {
			fscanf(fi, "%d", &Priority[i]);
		}

		// Cargo la matriz ETC. ====================================================================
		ETC = (float **) malloc(sizeof(float *) * NT);

		if (ETC == NULL) {
			fprintf(stderr,
					"Error in malloc for ETC matrix, dimensions %dx%d\n", NT,
					NM);
			exit(2);
		}

		for (i = 0; i < NT; i++) {
			ETC[i] = (float *) malloc(sizeof(float) * NM);
			if (ETC[i] == NULL) {
				fprintf(stderr, "Error in malloc, row %d in ETC\n", i);
				exit(2);
			}
		}

		for (i = 0; i < NT; i++) {
			for (j = 0; j < NM; j++) {
				fscanf(fi, "%f", &ETC[i][j]);
			}
		}

		// [FIN] Cierro el archivo de instancia. ===================================================
		fclose(fi);

		// Inicializo la matriz de makespan. =======================================================
		mach = (float *) malloc(sizeof(float) * NM);
		if (mach == NULL) {
			fprintf(stderr, "Error in malloc (machine array), dimension %d\n",
					NT);
			exit(2);
		}

		for (j = 0; j < NM; j++) {
			mach[j] = 0.0;
		}

		// Inicializo la matriz de asignación. =====================================================
		asig = (int*) malloc(sizeof(float) * NT);
		if (asig == NULL) {
			fprintf(stderr,
					"Error in malloc (assigned tasks array), dimension %d\n",
					NT);
			exit(2);
		}

		for (i = 0; i < NT; i++) {
			asig[i] = NO_ASIG;
		}

		asignMatrix = (int**) malloc(sizeof(int*) * NM);
		for (i = 0; i < NM; i++) {
			asignMatrix[i] = (int*) malloc(sizeof(int) * NT);

			for (j = 0; j < NT; j++) {
				asignMatrix[i][j] = NO_ASIG;
			}
		}

		lastAsig = (int*) malloc(sizeof(int) * NM);
		for (i = 0; i < NM; i++) {
			lastAsig[i] = NO_ASIG;
		}
	}

	float min_wrr_task;
	int best_mach_task;
	int nro_asig = 0; // Cantidad de tareas sin asignar.

	// Loop in tasks.
	for (int i = 0; i < NT; i++) {
		min_wrr_task = FLT_MAX;
		best_mach_task = -1;
		float wrr = 0.0;

		// Loop in machines
		for (int j = 0; j < NM; j++) {
			if (ETC[i][j] > 0.0) {
				if (mach[j] > 0.0) {
					wrr = Priority[i] * ((mach[j] + ETC[i][j])
							/ ETC[i][j]);
				} else {
					wrr = Priority[i] / ETC[i][j];
				}
			} else {
				wrr = 0.0;
			}

			if (wrr < min_wrr_task) {
				min_wrr_task = wrr;
				best_mach_task = j;
			}
		}

		mach[best_mach_task] += ETC[i][best_mach_task];
		asig[i] = best_mach_task;
		lastAsig[best_mach_task]++;
		asignMatrix[best_mach_task][lastAsig[best_mach_task]] = i;
		nro_asig++;
	}

	// Makespan final de la solución.
	float makespan = 0.0;
	for (int j = 0; j < NM; j++) {
		if (mach[j] > makespan) {
			makespan = mach[j];
		}
	}

	fprintf(stdout, "Makespan: %f\n", makespan);

	// WRR final de la solución.
	float total_wrr = 0.0;

	for (int maquinaId = 0; maquinaId < NM; maquinaId++) {
		float flowtime = 0.0;

		for (int i = 0; i < lastAsig[maquinaId]; i++) {
			int tareaId;
			tareaId = asignMatrix[maquinaId][i];

			if (ETC[tareaId][maquinaId] > 0.0) {
				total_wrr += Priority[tareaId] * (flowtime / ETC[tareaId][maquinaId]);
				flowtime += ETC[tareaId][maquinaId];
			}
		}
	}

	fprintf(stdout, "WRR: %f\n", total_wrr);

	return EXIT_SUCCESS;
}
