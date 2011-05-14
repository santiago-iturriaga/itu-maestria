// Sufferage robustness scheduler.
// Parameters : <instance_ETC_file> <num_tasks> <num_machines> <tau>
//	

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <vector>

#define NO_ASIG -1
#define SIZE_NOM_ARCH 180

#define DEBUG 0

using namespace std;

int main(int argc, char *argv[]) {

	if (argc < 2) {
		printf("Sintaxis: %s <archivo_instancias> \n", argv[0]);
		exit(1);
	}

	char *arch_inst;	// Archivo de texto con la instancia del problema a resolver.
	int NT; // Cantidad de tareas.
	int NM; // Cantidad de máquinas.
	int *Priority; // Array de prioridades de tareas.
	float **ETC; // Matriz de ETC (Tarea x Máquina).
	float *mach; // Makespan de cada máquina.
	int *asig; // Array que marca el estado de cada tareas (ASIG/NO_ASIG).
	int **asignMatrix; // Matriz de asignación (Máquina x Tarea).
	int *lastAsig; // Array con la cantidad de tareas asignadas a cada máquina.

	{
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
		asig = (int*) malloc(sizeof(int) * NT);
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

	float worst_sufferage;
	float min_ct_task;
	float second_min_ct_task;
	float sufferage_task;
	int best_machine;
	int best_mach_task;
	int best_task;
	int nro_asig = 0;

	while (nro_asig < NT) {
		// Select non-assigned tasks with minimum completion time.
		best_task = -1;
		best_machine = -1;
		worst_sufferage = 0.0;

		// Loop in tasks.
		for (int i = 0; i < NT; i++)
		{
			min_ct_task = FLT_MAX;
			second_min_ct_task = FLT_MAX;
			best_mach_task = -1;
			sufferage_task = 0.0;

			if (asig[i] == NO_ASIG)
			{
				float ct = 0.0;

				// Loop in machines
				for (int j = 0; j < NM; j++)
				{
					ct = mach[j] + ETC[i][j];

					if (ct < min_ct_task) {
						second_min_ct_task = min_ct_task;
						min_ct_task = ct;
						best_mach_task = j;
					} else {
						if (ct < second_min_ct_task) {
							second_min_ct_task = ct;
						}
					}
				}

				sufferage_task = second_min_ct_task - min_ct_task;
			}

			if (sufferage_task > worst_sufferage) {
				worst_sufferage = sufferage_task;
				best_task = i;
				best_machine = best_mach_task;
			}
		}

		mach[best_machine] += ETC[best_task][best_machine];
		asig[best_task] = best_machine;
		nro_asig++;
		lastAsig[best_machine]++;
		asignMatrix[best_machine][lastAsig[best_machine]] = best_task;
	}

	// Makespan final de la solución.
	float makespan = 0.0;
	for (int j = 0; j < NM; j++) {
		if (mach[j] > makespan) {
			makespan = mach[j];
		}
	}

	// WRR final de la solución.
	float total_wrr = 0.0;

	vector<int> sortedAssignMatrix;
	sortedAssignMatrix.reserve(NT);

	for (int maquinaId = 0; maquinaId < NM; maquinaId++) {
		float flowtime = 0.0;
		sortedAssignMatrix.clear();

		for (int i = 0; i < lastAsig[maquinaId]; i++) {
			int tareaId;
			tareaId = asignMatrix[maquinaId][i];

			bool no_asignado = true;

			for (vector<int>::iterator j = sortedAssignMatrix.begin(); no_asignado && (j < sortedAssignMatrix.end()); j++) {
				if (ETC[*j][maquinaId] > ETC[tareaId][maquinaId]) {
					sortedAssignMatrix.insert(j, tareaId);
					no_asignado = false;
				}
			}

			if (no_asignado) {
				sortedAssignMatrix.push_back(tareaId);
			}
		}

		for (vector<int>::iterator i = sortedAssignMatrix.begin(); i < sortedAssignMatrix.end(); i++) {
			int tareaId;
			tareaId = *i;

			if (ETC[tareaId][maquinaId] > 0.0) {
				flowtime += ETC[tareaId][maquinaId];
				total_wrr += Priority[tareaId] * (flowtime / ETC[tareaId][maquinaId]);
			}
		}
	}

	if (DEBUG == 0) {
		fprintf(stdout, "%s|%d|%d|%f|%f\n", arch_inst, NT, NM, makespan, total_wrr);
	} else {
		fprintf(stdout, "%s\n", arch_inst);
		fprintf(stdout, "NT: %d, NM: %d\n", NT, NM);
		fprintf(stdout, "Makespan: %f\n", makespan);
		fprintf(stdout, "WRR: %f\n", total_wrr);
	}

	return EXIT_SUCCESS;
}
