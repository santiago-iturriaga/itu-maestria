$TITLE multicore_scheduling_min_wqt
* Scheduling of tasks in multicore machine minimizing
* the weighted queued time (WQT)

SET t 'tasks' /1*8/;
SET m 'machines' /1*2/;
SET c 'cores' /1*4/;

alias(t,tt);
alias(m,mm);
alias(c,cc);

* dimension 8x2
* scenario_c4_high.1
* arrival.0
* cores_c2.0
* priorities.0
* workload_high.0

PARAMETER m_cant_cores(m) 'cantidad de cores de la maquina m'
    /    1    2
        2    4 /;
PARAMETER m_cores(m, c) 'cores de la maquina m'
    /    1    .1    1
        1    .2    1
        1    .3    0
        1    .4    0
        2    .1    1
        2    .2    1
        2    .3    1
        2    .4    1 /;
PARAMETER t_arrival(t) 'arribo (en segundos) de la tarea t'
    /    1    102
        2    104
        3    115
        4    121
        5    195
        6    212
        7    219
        8    232 /;
PARAMETER t_cores(t) 'cores requeridos por la tarea t'
    /    1    1
        2    1
        3    1
        4    1
        5    1
        6    2
        7    1
        8    1 /;
PARAMETER t_priorities(t) 'prioridad de la tarea t (1 = tarea menos prioritaria, 5 tarea mas prioritaria)'
    /    1    3
        2    3
        3    3
        4    3
        5    3
        6    2
        7    4
        8    3 /;
PARAMETER etc(t,m) 'costo computacional por core de la ejecucion de la tarea t en la maquina m'
    /    1    .1    4729.8822
        1    .2    4544.9179
        2    .1    3367.2307
        2    .2    3235.5535
        3    .1    14080.8667
        3    .2    13530.2278
        4    .1    1452.9241
        4    .2    1396.1068
        5    .1    24536.6753
        5    .2    23577.1571
        6    .1    18191.0048
        6    .2    17479.6370
        7    .1    14216.0811
        7    .2    13660.1545
        8    .1    1799.5884
        8    .2    1729.2147 /;
PARAMETER m_eidle(m) 'consumo por unidad de tiempo de la maquina m en estado ocioso'
    /    1    86.0
        2    63.5 /;
PARAMETER eec(t,m) 'costo energetico de la ejecucion de la tarea t en la maquina m'
    /    1    .1    101692.4668
        1    .2    74423.0307
        2    .1    72395.4595
        2    .2    52982.1891
        3    .1    302738.6350
        3    .2    221557.4806
        4    .1    31237.8686
        4    .2    22861.2495
        5    .1    527538.5190
        5    .2    386075.9470
        6    .1    782213.2075
        6    .2    572458.1124
        7    .1    305645.7427
        7    .2    223685.0302
        8    .1    38691.1515
        8    .2    28315.8906 /;


VARIABLE f objetivo;
POSITIVE VARIABLE starting_time(t)       'comienzo de ejecucion de la tarea t';
POSITIVE VARIABLE completion_time(t)     'tiempo de finalizacion de la tarea t';
BINARY VARIABLE assignment_m(t,m)        'asignacion de la tarea t en la maquina m';
BINARY VARIABLE assignment_mc(t,m,c)     'asignacion de la tarea t en la maquina m en el core c';
BINARY VARIABLE precedence(t,tt,m,c)     't es previa de tt en la maquina m, core c';
BINARY VARIABLE start(t,m,c)             'la tarea es la primera del core c en la maquina m';

scalar TMAX 'horizonte maximo de tiempo';
TMAX = sum(t,smax(m,etc(t,m)));

EQUATION exec_all_m(t)                   'cada tarea se debe ejecutar en una maquina';
EQUATION exec_all_c(t,m)                 'se deben ejecutar todos los cores requeridos por la tarea';
EQUATION machine_nc(t,m)                 'cores disponibles en la maquina';
EQUATION machine_nc_2(t,m,c)             'cores disponibles en la maquina';
EQUATION eq_completion_time(t)           'asigno el completion time de cada tarea';
EQUATION no_overlap(t,tt,m,c)            'verifico que las tareas no se pisen';
EQUATION objective                       'evaluo el objetivo';
EQUATION enforce_precedence(t,m,c)       'toda tarea tiene que tener una unica tarea previa';
EQUATION enforce_precedence_2(tt,m,c)    'el previous es único dos tareas en un core no pueden tener el mismo previous';
EQUATION enforce_start(m,c)              'todo core tiene que tener una tarea start';
EQUATION enforce_arrival(t)              'toda tarea debe empezar luego de su arrival time';

exec_all_m(t)..          sum(m, assignment_m(t,m)) =e= 1;
exec_all_c(t,m)..        sum(c, assignment_mc(t,m,c)) =e= assignment_m(t,m) * t_cores(t);
machine_nc(t,m)..        sum(c, assignment_mc(t,m,c)) =l= assignment_m(t,m) * m_cant_cores(m);
machine_nc_2(t,m,c)..    assignment_mc(t,m,c) =l= m_cores(m,c);

eq_completion_time(t)..  completion_time(t) =e= starting_time(t) + sum(m, assignment_m(t,m) * etc(t,m));
enforce_arrival(t)..     starting_time(t) =g= t_arrival(t);
no_overlap(t,tt,m,c)..   starting_time(tt) =g= completion_time(t)
                                 - TMAX*(start(tt,m,c))
                                 - TMAX*(1-precedence(t,tt,m,c))
                                 - TMAX*(1-assignment_mc(t,m,c))
                                 - TMAX*(1-assignment_mc(t,m,c));

enforce_precedence(t,m,c)..      1 =e= sum(tt, precedence(t,tt,m,c))
                                         + (1 - assignment_mc(t,m,c));

enforce_precedence_2(tt,m,c)..   1 =e= sum(t, precedence(t,tt,m,c))
                                         + (1 - assignment_mc(tt,m,c));

enforce_start(m,c)..             1 =e= sum(t, start(t,m,c));

objective..                      f =e= sum(t, t_priorities(t) * (completion_time(t)-t_arrival(t)));

MODEL min_f/ALL/;
OPTION MINLP = bonmin;
*OPTION MIP = bonmin;
OPTION MIP = cplex;
SOLVE min_f using mip minimizing f;

DISPLAY TMAX;
DISPLAY f.l;
DISPLAY starting_time.l;
DISPLAY completion_time.l;
DISPLAY assignment_m.l;
DISPLAY assignment_mc.l;
DISPLAY precedence.l;
DISPLAY start.l;

file salida_starting_time /wqt_starting_time.txt/;
put salida_starting_time;
loop(t, put starting_time.l(t):<:4 /);
putclose salida_starting_time;

file salida_assignment_m /wqt_assignment_m.txt/;
put salida_assignment_m;
loop((t,m)$(assignment_m.l(t,m) = 1), put ord(m):<:0 /);
putclose salida_assignment_m;