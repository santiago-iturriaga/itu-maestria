#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Uso FP <archivo_entrada> <archivo frente>

#define MAX_POINTS 200000

int main(int argc, char **argv) {

	FILE *fpin, *fp, *fpaux;
	float f1, f2, f3;

	int tam = 16;

	float p[MAX_POINTS];
	int cont[tam];

	int cant_p = 0;
	int i, k;

	if (argc < 1) {
		printf("Uso : %s <archivo_entrada>\n", argv[0]);
		exit(1);
	}

	for (i = 0; i < MAX_POINTS; i++) {
		p[i] = 0.0;
	}

	for (i = 0; i < tam; i++) {
		cont[i] = 0;
	}

	printf("Archivo : %s\n", argv[1]);
	if ((fpin = fopen(argv[1], "r"))) {
		while (fscanf(fpin, "%f %f %f", &f1, &f2, &f3) != EOF) {
			// Leer punto
			p[3 * cant_p] = f1;
			p[3 * cant_p + 1] = f2;
			p[3 * cant_p + 2] = f3;
			printf("Leido %f %f %f\n", f1, f2, f3);
			cant_p++;
		}

		printf("cant_p: %d\n", cant_p);

		fpaux = fopen("FP.out.disc", "w");

		if ((fp = fopen("FP.out", "w"))) {
			for (i = 0; i < cant_p; i++) {
				// Para cada punto, determinar si es NO dominado
				int nd = 1;
				printf("Chequear %f %f %f\n", p[3 * i], p[3 * i + 1], p[3 * i
						+ 2]);

				for (k = 0; (k < cant_p && nd); k++) {
					if (i != k) {
						if (((p[3 * k] < p[3 * i]) && (p[3 * k + 1] <= p[3 * i
								+ 1])) || ((p[3 * k] <= p[3 * i]) && (p[3 * k
								+ 1] < p[3 * i + 1]))) {

							printf("Dominado por %f %f %f\n", p[3 * k], p[3 * k
									+ 1], p[3 * k + 2]);
							nd = 0;
						}
					}
					if (i < k) {
						if ((p[3 * k] == p[3 * i]) && (p[3 * k + 1] == p[3 * i
								+ 1])) {
							printf("Ya existe\n");
							nd = 0;
						}
					}
				}
				if (nd == 1) {
					printf("NO dominado\n");
					fprintf(fp, "%f %f\n", p[3 * i], p[3 * i + 1]);
					f3 = p[3 * i + 2];
					fprintf(fpaux, "%f %f %f\n", p[3 * i], p[3 * i + 1], f3);

					cont[(int) f3] = cont[(int) f3] + 1;

					/*
					 if (f3 == 0.0){
					 cont[0] = cont[0]+1;
					 } else {
					 if (f3 == 1.0){
					 cont[1] = cont[1]+1;
					 } else {
					 if (f3 == 2.0){
					 cont[2] = cont[2]+1;
					 } else {
					 if (f3 == 3.0){
					 cont[3] = cont[3]+1;
					 }
					 }
					 }
					 }
					 */
				}
			}
			fclose(fp);
			for (i = 0; i < tam; i++) {
				printf("%d - %d\n", i, cont[i]);
			}
		}

	} else {
		printf("Error al abrir archivo %s\n", argv[1]);
		exit(1);
	}
}
