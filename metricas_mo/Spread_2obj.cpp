#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_DIST 999999999
#define MAX_PUNTOS 1000

float distancia(float x1, float x2, float y1, float y2) {
	float deltax = x1 - x2;
	float deltay = y1 - y2;
	float dx2 = pow(deltax, 2);
	float dy2 = pow(deltay, 2);

	float sum = dy2 + dx2;
	float res = sqrt(sum);
	return (res);
}

int main(int argc, char **argv) {
	if (argc < 3) {
		printf("Uso : ./spread <archivo_ejecucion> <archivo_global> [debug]\n");
		exit(1);
	}

	bool debug = argc >= 4;
	int size = 0;

	FILE *fpin, *fglob;

	float func[MAX_PUNTOS][2];
	float dist[MAX_PUNTOS];

	float sum = 0.0;
	float f1, f2;

	float e1f1, e1f2;
	float e2f1, e2f2;

	float de1, de2;

	if (debug)
		printf("Archivo instancia: %s\n", argv[1]);
	if (debug)
		printf("Archivo global: %s\n", argv[2]);

	// Cargar extremo 1
	e1f1 = INFINITY; //atof(argv[5]);
	e1f2 = INFINITY; //atof(argv[6]);

	// Cargar extremo 2
	e2f1 = INFINITY; //atof(argv[7]);
	e2f2 = INFINITY; //atof(argv[8]);

	if ((fglob = fopen(argv[2], "r")) && (fpin = fopen(argv[1], "r"))) {
		while (fscanf(fglob, "%f %f", &f1, &f2) != EOF) {
			if (f1 < e1f1) {
				e1f1 = f1;
				e1f2 = f2;
			}

			if (f2 < e2f2) {
				e2f1 = f1;
				e2f2 = f2;
			}
		}

		if (debug)
			printf("Extremo 1: (%f %f)\n", e1f1, e1f2);
		if (debug)
			printf("Extremo 2: (%f %f)\n", e2f1, e2f2);

		while (fscanf(fpin, "%f %f", &f1, &f2) != EOF) {
			// Leer e inicalizar func
			func[size][0] = f1;
			func[size][1] = f2;

			if (f1 < e1f1) {
				e1f1 = f1;
				e1f2 = f2;
			}

			if (f2 < e2f2) {
				e2f1 = f1;
				e2f2 = f2;
			}

			//printf("Leo (%f %f)\n",func[i][0],func[i][1]);
			// sleep(1);
			size++;
		}

		if (debug)
			printf("size: %d\n", size);

		float mindist, distcalc;
		int k, h, l, min;

		// Calculo de1
		min = 0;
		l = 0;
		mindist = MAX_DIST;
		for (h = 0; h < size; h++) {
			distcalc = distancia(e1f1, func[h][0], e1f2, func[h][1]);
			if (distcalc < mindist) {
				mindist = distcalc;
				min = h;
			}
		}
		de1 = mindist;

		if (debug)
			printf("e1, cercano:%d\n", min);
		if (debug)
			printf("Dist (%f %f) <-> (%f %f) = %f\n", e1f1, e1f2, func[min][0],
					func[min][1], de1);

		// Calculo de2
		min = 0;
		mindist = MAX_DIST;
		for (h = 0; h < size; h++) {
			distcalc = distancia(e2f1, func[h][0], e2f2, func[h][1]);
			if (distcalc < mindist) {
				mindist = distcalc;
				min = h;
			}
		}
		de2 = mindist;

		if (debug)
			printf("e2, cercano:%d\n", min);
		if (debug)
			printf("Dist (%f %f) <-> (%f %f) = %f\n", e2f1, e2f2, func[min][0],
					func[min][1], de2);

		int count = 0;
		for (k = 0; k < size - 1; k++) {
			mindist = MAX_DIST;
			min = -1;
			for (h = 0; h < size; h++) {
				if ((h != k)) {
					distcalc = distancia(func[k][0], func[h][0], func[k][1],
							func[h][1]);
					if (distcalc < mindist) {
						mindist = distcalc;
						min = h;
					}
				}
			}
			if (debug)
				printf("k:%d, cercano:%d\n", k, min);
			if (debug)
				printf("Dist (%f %f) <-> (%f %f) = %f\n", func[k][0],
						func[k][1], func[min][0], func[min][1], mindist);

			dist[k] = mindist;
			sum += dist[k];
			count++;
		}

		float dprom = sum / count;
		float desv, delta;
		float sumdesv = 0.0;
		if (debug)
			printf("%d distancias calculadas\n", count);
		if (debug)
			printf("Dist prom = %f \n", dprom);

		for (int i = 0; i < size - 1; i++) {
			if (i != l) {
				desv = fabs(dist[i] - dprom);
				sumdesv += desv;
			}
		}

		if (debug)
			printf("Sum desv: %f\n", sumdesv);
		float sumext = de1 + de2;
		if (debug)
			printf("Sum ext: %f\n", sumext);

		delta = (sumext + sumdesv) / (sumext + dprom * (count));

		if (debug)
			printf("Delta %f\n", delta);
		else
			printf("%f\n", delta);
	} else {
		printf("Error al abrir archivo %s\n", argv[1]);
	}
}

