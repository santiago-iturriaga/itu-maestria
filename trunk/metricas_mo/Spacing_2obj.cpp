#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_DIST 999999999
#define MAX_PUNTOS 1000

float distancia3(float x1, float x2, float y1, float y2, float z1, float z2) {
	return (fabs(x1 - x2) + fabs(y1 - y2) + fabs(z1 - z2));
}

float distancia2(float x1, float x2, float y1, float y2) {
	return (fabs(x1 - x2) + fabs(y1 - y2));
}

int main(int argc, char **argv) {

	if (argc < 2) {
		printf("Uso : ./spacing <archivo_entrada>\n");
		exit(1);
	}

	FILE *fpin;

	float func[MAX_PUNTOS][2];
	float dist[MAX_PUNTOS];

	int i = 0;
	float sum = 0.0;
	float f1, f2;

	printf("Archivo : %s\n", argv[1]);

	if ((fpin = fopen(argv[1], "r"))) {
		while (fscanf(fpin, "%f %f", &f1, &f2) != EOF) {
			// Leer e inicalizar func
			func[i][0] = f1;
			func[i][1] = f2;
			// printf("Leo (%f %f)\n",func[i][0],func[i][1]);
			i++;
		}

		printf("size: %d\n", i);

		float mindist, distcalc;
		int k, h, min;

		int count = 0;
		k = 0;
		for (k = 0; k < i - 1; k++) {
			mindist = MAX_DIST;
			min = -1;
			for (h = 0; h < i - 1; h++) {
				if (h != k) {
					distcalc = distancia2(func[k][0], func[h][0], func[k][1],
							func[h][1]);
					if (distcalc < mindist) {
						mindist = distcalc;
						min = h;
					}
				}
			}
			printf("%d : Dist (%f %f) <-> (%f %f) = %f (cercano %d)\n", k,
					func[k][0], func[k][1], func[min][0], func[min][1],
					mindist, min);
			dist[k] = mindist;
			sum += dist[k];
			count++;
		}

		float dprom = sum / count;
		float desv, spacing;
		float sumdesv = 0.0;
		printf("%d puntos\n", count);
		printf("Dist prom = %f \n", dprom);

		for (int j = 0; j < i - 1; j++) {
			desv = pow(dist[j] - dprom, 2);
			sumdesv += desv;
		}
		printf("Sum desv: %f\n", sumdesv);

		spacing = sqrt(sum / (i - 1));
		printf("Spacing: %f\n", spacing);

	} else {
		printf("Error al abrir archivo %s\n", argv[1]);
	}
}

