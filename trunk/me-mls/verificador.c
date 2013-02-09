// Verificador de soluciones para instancias de Braun et al.
// Parametros: <archivo_instancia> <archivo_sol>
// El archivo de instancias DEBE llevar el cabezal con datos (NT, NM).
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INFT 9999999999.0
#define NO_ASIG -1

#define DEBUG 0

int main(int argc, char *argv[])
{

    if (argc < 6)
    {
        printf("Sintaxis: %s <scenario> <workload> <archivo_sol> <#tasks> <#machines>\n", argv[0]);
        exit(1);
    }

    char *arch_scenario = argv[1];
    char *arch_workload = argv[2];
    char *arch_sol = argv[3];
    int NT = atoi(argv[4]);
    int NM = atoi(argv[5]);

    FILE *fscenario, *fworkload, *fsolucion;

    if((fscenario = fopen(arch_scenario, "r"))==NULL)
    {
        printf("No se puede leer archivo de scenario %s\n", arch_scenario);
        exit(1);
    }

    if((fworkload = fopen(arch_workload, "r"))==NULL)
    {
        printf("No se puede leer archivo de workload %s\n", arch_workload);
        exit(1);
    }

    if((fsolucion = fopen(arch_sol, "r"))==NULL)
    {
        printf("No se puede leer archivo de solucion %s\n", arch_sol);
        exit(1);
    }

    int i,j;
    int solution_count;

    fscanf(fsolucion,"%d",&solution_count);
    //printf("#solutions: %d\n",solution_count);

    float **ETC = (float **) malloc(sizeof(float *)*NT);

    if (ETC == NULL)
    {
        printf("Error al reservar memoria para ETC, dimensiones %dx%d\n",NT,NM);
        exit(2);
    }

    for (i=0;i<NT;i++)
    {
        ETC[i] = (float *) malloc(sizeof(float)*NM);
        if (ETC[i] == NULL)
        {
            printf("Error al reservar memoria para fila %d de ETC\n",i);
            exit(2);
        }
    }

    int *cores = (int*)malloc(sizeof(int) * NM);
    float *ssj = (float*)malloc(sizeof(float) * NM);
    float *energy_idle = (float*)malloc(sizeof(float) * NM);
    float *energy_max = (float*)malloc(sizeof(float) * NM);

    for (j=0;j<NM;j++)
    {
        fscanf(fscenario,"%d %f %f %f\n",&cores[j],&ssj[j],&energy_idle[j],&energy_max[j]);
    }

    float aux_etc;

    for (i=0;i<NT;i++)
    {
        for (j=0;j<NM;j++)
        {
            fscanf(fworkload,"%f",&aux_etc);
            ETC[i][j] = aux_etc / (ssj[j] / (cores[j] * 1000));
            //ETC[i][j] = aux_etc / ssj[j];
        }
    }

    fclose(fworkload);
    fclose(fscenario);

    int *asig= (int*) malloc(sizeof(float)*NT);
    float *mach = (float *) malloc(sizeof(float)*NM);

    for (int s_idx = 0; s_idx < solution_count; s_idx++) {
        // Array de maquinas, almacena el MAT
        for (j=0;j<NM;j++)
        {
            mach[j]=0.0;
        }

        // Array de asignaciones
        for (i=0;i<NT;i++)
        {
            asig[i]=NO_ASIG;
        }

        for (i=0;i<NT;i++)
        {
            fscanf(fsolucion,"%d ",&asig[i]);
        }

        for (i=0;i<NT;i++)
        {
            if (asig[i]==NO_ASIG)
            {
                printf("ERROR!!! tarea %d no asignada!!!\n", i);
                exit(1);
            }
        }

        int maq;

        for(i=0;i<NT;i++)
        {
            maq = asig[i];
            mach[maq] = mach[maq]+ETC[i][maq];
        }

        float makespan = 0.0;
        for (j=0;j<NM;j++)
        {
            if (mach[j]>makespan)
            {
                makespan = mach[j];
            }
        }

        float energy = 0.0;
        for (j=0;j<NM;j++)
        {
            energy += (mach[j] * energy_max[j]) + ((makespan - mach[j]) * energy_idle[j]);
        }

        printf("%f %f\n",makespan, energy);
    }

    fclose(fsolucion);
}
