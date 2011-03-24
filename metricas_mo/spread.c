#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_DIST 999999999

float distancia (float x1, float x2, float y1, float y2) {
	float deltax=x1-x2;
	float deltay=y1-y2;
	float dx2=pow(deltax,2);
	float dy2=pow(deltay,2);

	float sum=dy2+dx2;
	float res=sqrt(sum);
        return(res);
}

main(int argc, char **argv) {

if ( argc < 8 ) {
        printf("Uso : ./spread <archivo_entrada> <numero de funciones> <cantidad puntos> <distancia desconexo> <e1f1> <e1f2> <e2f1> <e2f2> \n");
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

float e1f1,e1f2;
float e2f1,e2f2;
float e3f1,e3f2;

float de1,de2,de3;

printf("Archivo : %s\n",argv[1]);
printf("nfunc: %d\n",nfunc);
printf("size: %d\n",size);
printf("desc: %f\n",desc);

// Cargar extremo 1
e1f1= atof(argv[5]);
e1f2= atof(argv[6]);

// Cargar extremo 2
e2f1= atof(argv[7]);
e2f2= atof(argv[8]);

if (fpin=fopen(argv[1],"r")){
	while (fscanf(fpin,"%f %f",&f1,&f2) != EOF) {
		// Leer e inicalizar func
		func[i][0] = f1;
		func[i][1] = f2;
//printf("Leo (%f %f)\n",func[i][0],func[i][1]);
// sleep(1);
		i++;
	}

  	float mindist,distcalc;
  	int k,h,l,min;

	// Calculo de1	
	min=0;
	mindist=MAX_DIST;
	for(h=0;h < size;h++) {
		distcalc = distancia(e1f1,func[h][0],e1f2,func[h][1]); 
		if ( distcalc < mindist ) {
			mindist=distcalc;
			min=h;
		}			
	}
	de1=mindist;
	printf("e1, cercano:%d\n", min);
	printf("Dist (%f %f) <-> (%f %f) = %f\n",e1f1,e1f2,func[min][0],func[min][1],de1);

	// Calculo de2	
	min=0;	
	mindist=MAX_DIST;
	for(h=0;h < size;h++) {
		distcalc = distancia(e2f1,func[h][0],e2f2,func[h][1]); 
		if ( distcalc < mindist ) {
			mindist=distcalc;
			min=h;
		}			
	}
	de2=mindist;
	printf("e2, cercano:%d\n", min);
	printf("Dist (%f %f) <-> (%f %f) = %f\n",e2f1,e2f2,func[min][0],func[min][1],de2);
	
	int count=0;
  	for(k=0;k<size-1;k++) {
		mindist=MAX_DIST;
		min=-1;	
		for(h=0;h < size;h++) {
			if ( ( h != k ) ) {
				distcalc = distancia(func[k][0],func[h][0],func[k][1],func[h][1]);
				if ( distcalc < mindist ) {
					mindist=distcalc;
					min=h;
				}			
			}
		}
		printf("k:%d, cercano:%d\n", k, min);
printf("Dist (%f %f) <-> (%f %f) = %f\n",func[k][0],func[k][1],func[min][0],func[min][1],mindist);
		dist[k]=mindist;
		sum+=dist[k];
		count++;
	}

	float dprom=sum/count;
	float desv, delta;
	float sumdesv=0.0;
printf("%d distancias calculadas\n",count);
printf("Dist prom = %f \n",dprom);

	for(i = 0;i < size-1 ;i++) {
		if ( i != l ) {
			desv=fabs(dist[i]-dprom);
			sumdesv+=desv;
		}
	}
printf("Sum desv: %f\n",sumdesv);
float sumext=de1+de2;
printf("Sum ext: %f\n",sumext);

	delta=(sumext+sumdesv)/(sumext+dprom*(count));
	printf("Delta %f\n",delta);

} else {
	printf("Error al abrir archivo %s\n",argv[1]);
}	
}

