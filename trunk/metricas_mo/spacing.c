#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_DIST 999999999

float distancia3 (float x1, float x2, float y1, float y2, float z1, float z2) {
        return(fabs(x1-x2)+fabs(y1-y2)+fabs(z1-z2));
}

float distancia2 (float x1, float x2, float y1, float y2) {
        return(fabs(x1-x2)+fabs(y1-y2));
}

main(int argc, char **argv) {

if ( argc < 3 ) {
        printf("Uso : ./spacing <archivo_entrada> <numero de funciones> <numero de puntos> <dist. desconexo>\n"); 
        exit(1);
}
int nfunc=atoi(argv[2]);
int size=atoi(argv[3]);
float desc=atof(argv[4]);

FILE *fpin;
float func[size][nfunc];

float dist[size],max;

int i=0;
float sum=0.0;
float f1,f2,f3;

printf("Archivo : %s\n",argv[1]);
printf("nfunc: %d\n",nfunc);
printf("size: %d\n",size);
printf("desc: %f\n",desc);

if (fpin=fopen(argv[1],"r")){
	switch (nfunc) {
		case 2: while (fscanf(fpin,"%f %f",&f1,&f2) != EOF) {
			// Leer e inicalizar func
				func[i][0] = f1;
				func[i][1] = f2;
				func[i][2] = f3;
				// printf("Leo (%f %f)\n",func[i][0],func[i][1]);
				i++;
			}
			break;;
		case 3: while (fscanf(fpin,"%f %f %f",&f1,&f2,&f3) != EOF) {
			// Leer e inicalizar func
				func[i][0] = f1;
				func[i][1] = f2;
				func[i][2] = f3;
				//printf("Leo (%f %f %f)\n",func[i][0],func[i][1],func[i][2]);
				i++;
			}
			break;;
	}

  	float mindist,distcalc;
  	int k,h,min;
	
	int count=0;
	k=0;
  	for(k=0;k<size-1;k++) {
		mindist=MAX_DIST;
		min=-1;
		for(h=0;h < size-1;h++) {
			if ( h != k ){
				switch (nfunc) {
					case 2: distcalc = distancia2(func[k][0],func[h][0],func[k][1],func[h][1]);
							break;;
					case 3: distcalc = distancia3(func[k][0],func[h][0],func[k][1],func[h][1],func[k][2],func[h][2]); 
							break;;
				}
				if ( distcalc < mindist ) {
					mindist=distcalc;
					min=h;
				}			
			}
		}
		switch (nfunc) {
			case 2: printf("%d : Dist (%f %f) <-> (%f %f) = %f (cercano %d)\n",k,func[k][0],func[k][1],func[min][0],func[min][1],mindist,min);
					break;;
			case 3: printf("Dist (%f %f %f) <-> (%f %f %f) = %f\n",func[k][0],func[k][1],func[k][2],func[min][0],func[min][1],func[min][2],mindist);
					break;;
		}
		dist[k]=mindist;
		sum+=dist[k];
		count++;
	}

	float dprom=sum/count;
	float desv, spacing;
	float sumdesv=0.0;
printf("%d puntos\n",count);
printf("Dist prom = %f \n",dprom);

	for(i = 0;i < size-1 ;i++) {
		desv=pow(dist[i]-dprom,2);
		sumdesv+=desv;
	}
printf("Sum desv%f\n",sumdesv);

	spacing=sqrt(sum/(size-1));
	printf("Spacing: %f\n",spacing);

} else {
	printf("Error al abrir archivo %s\n",argv[1]);
}	
}

