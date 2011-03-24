#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Uso FP <archivo_entrada> <archivo frente>

#define MAX_POINTS 200000

int main(int argc, char **argv) {

	FILE *fpin, *fp;
	float f1, f2;

	float p[MAX_POINTS];

	int cant_p = 0;
	int i, k;

	if (argc < 1) {
		printf("Uso : %s <archivo_entrada>\n", argv[0]);
		exit(1);
	}

	for (i = 0; i < MAX_POINTS; i++) {
		p[i] = 0.0;
	}

	printf("Archivo : %s\n", argv[1]);
	if ((fpin = fopen(argv[1], "r"))) {
		while (fscanf(fpin, "%f %f", &f1, &f2) != EOF) {
			// Leer punto
			p[3 * cant_p] = f1;
			p[3 * cant_p + 1] = f2;
			printf("Leido %f %f \n", f1, f2);
			cant_p++;
		}

		printf("cant_p: %d\n", cant_p);

		if ((fp = fopen("FP.out", "w"))) {
			for (i = 0; i < cant_p; i++) {
				// Para cada punto, determinar si es NO dominado
				int nd = 1;
				printf("Chequear %f %f\n", p[3 * i], p[3 * i + 1]);
				for (k = 0; (k < cant_p && nd); k++) {
					if (i != k) {
						if (((p[3 * k] < p[3 * i]) && (p[3 * k + 1] <= p[3 * i
								+ 1])) || ((p[3 * k] <= p[3 * i]) && (p[3 * k
								+ 1] < p[3 * i + 1]))) {
							printf("Dominado por %f %f\n", p[3 * k], p[3 * k
									+ 1]);
							nd = 0;
						}
					}
				}
				if (nd == 1) {
					printf("NO dominado\n");
					fprintf(fp, "%f %f\n", p[3 * i], p[3 * i + 1]);
				}
			}
			fclose(fp);
		}

	} else {
		printf("Error al abrir archivo %s\n", argv[1]);
		exit(1);
	}
}
