#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159
#define EULER   2.71828
#define MAX_DIST 999999999

double EXP(const double x) {
	return pow(EULER, x);
}

// Uso fp <archivo_entrada> <archivo frente>

float distancia(float x1, float x2, float y1, float y2) {
	return (sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2)));
}

int main(int argc, char **argv) {

	FILE *fpin, *fpcalc;
	float f1, f2, f3, fp1, fp2, fp3;
	float dist, mindist, dprom;
	float sumdist = 0;
	float tope_izq_f1, tope_der_f1;
	int cantpts = 0;

	if (argc < 2) {
		printf("Uso : ./fp <archivo_entrada> <archivo frente> \n");
		exit(1);
	}

	printf("Archivo : %s\n", argv[1]);
	if (fpin = fopen(argv[1], "r")) {
		while (fscanf(fpin, "%f %f", &f1, &f2) != EOF) {
			// Para cada punto, calcular distancia a FP
			// Dado (f1,f2) se lee el frente
			if (fpcalc = fopen(argv[2], "r")) {
				mindist = MAX_DIST;
				while (fscanf(fpcalc, "%f %f", &fp1, &fp2) != EOF) {
					dist = distancia(f1, fp1, f2, fp2);
					if (dist < mindist) {
						mindist = dist;
					}
				}
				printf("Minima distancia de (%f %f) a FP: %f\n", f1, f2,
						mindist);
				sumdist += mindist;
				cantpts++;
				//sleep(1);
			} else {
				printf("Error al abrir archivo %s\n", argv[2]);
				exit(1);
			}
			fclose(fpcalc);
		}
		fclose(fpin);
		dprom = sumdist / cantpts;
		printf("%d puntos. Distancia promedio al f. Pareto : %f\n", cantpts,
				dprom);
	} else {
		printf("Error al abrir archivo %s\n", argv[1]);
		exit(1);
	}
}
