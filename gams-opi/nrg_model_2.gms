* Model for scheduling of tasks in multicore
* machine minimizing the weighted queued time (WQT)

alias(t,tt);
alias(m,mm);

PARAMETER m_cores(m) 'cantidad de cores de la maquina m';
PARAMETER t_arrival(t) 'arribo (en segundos) de la tarea t';
PARAMETER t_cores(t) 'cores requeridos por la tarea t';
PARAMETER t_priorities(t) 'prioridad de la tarea t (1 = tarea menos prioritaria, 5 tarea mas prioritaria)';
PARAMETER m_eidle(m) 'consumo por unidad de tiempo de la maquina m en estado ocioso';
PARAMETER etc(t,m) 'costo computacional por core de la ejecucion de la tarea t en la maquina m';
PARAMETER eec(t,m) 'costo energetico de la ejecucion de la tarea t en la maquina m';

VARIABLE f objetivo;
POSITIVE VARIABLE starting_time(t)       'comienzo de ejecucion de la tarea t';
POSITIVE VARIABLE completion_time(t)     'tiempo de finalizacion de la tarea t';
BINARY VARIABLE assignment(t,m)        'asignacion de la tarea t en la maquina m';
BINARY VARIABLE precedence(t,tt,m)     't es previa de tt en la maquina m, core c';
INTEGER VARIABLE prec_res(t,tt,m)

scalar TOTAL_CORES;
TOTAL_CORES = sum(m, m_cores(m));

scalar BIG_M 'horizonte maximo de tiempo';
BIG_M = sum(t,smax(m,etc(t,m)));

EQUATION no_loop(t,tt,m)
EQUATION exec_all_1(t)                   'cada tarea se debe ejecutar en una maquina';
EQUATION exec_all_2(t,m)              'se deben ejecutar todos los cores requeridos por la tarea';
EQUATION m_resources_1(m,tt)                   'cores disponibles en la maquina';
EQUATION m_resources_2(m,tt)                 'cores disponibles en la maquina';
EQUATION m_resources_3(m,tt)                 'cores disponibles en la maquina';
EQUATION m_resources_4(m,tt)                 'cores disponibles en la maquina';
EQUATION eq_completion_time(t)           'asigno el completion time de cada tarea';
EQUATION no_overlap(t,tt,m)              'verifico que las tareas no se pisen';
EQUATION t_precedence_1(t,tt,m)          'toda tarea tiene que tener una unica tarea previa';
EQUATION t_precedence_2(t,tt,m)          'el previous es �nico dos tareas en un core no pueden tener el mismo previous';
EQUATION t_precedence_3(t,tt,m)          'el previous es �nico dos tareas en un core no pueden tener el mismo previous';
EQUATION t_precedence_4(t,tt,m)          'el previous es �nico dos tareas en un core no pueden tener el mismo previous';
EQUATION t_precedence_5(tt,m)          'el previous es �nico dos tareas en un core no pueden tener el mismo previous';
EQUATION t_precedence_6(t,m)          'el previous es �nico dos tareas en un core no pueden tener el mismo previous';
EQUATION enforce_arrival(t)              'toda tarea debe empezar luego de su arrival time';
EQUATION objective                       'evaluo el objetivo';

no_loop(t,tt,m)..          precedence(t,tt,m) + precedence(tt,t,m) =l= 1;

exec_all_1(t)$(ord(t)>1 and ord(t)<card(t))..          sum(m, assignment(t,m)) =e= 1;
exec_all_2(t,m)$(ord(t)=1 and ord(t)=card(t))..        assignment(t,m) =e= 1;

m_resources_1(m,tt)$(ord(tt)=1)..       sum(t, prec_res(tt,t,m)) =l= m_cores(m);
m_resources_2(m,tt)$(ord(tt)=1)..       sum(t, prec_res(t,tt,m)) =e= 0;
m_resources_3(m,tt)$(ord(tt)=card(t))..       sum(t, prec_res(t,tt,m)) =e= m_cores(m);
m_resources_4(m,tt)$(ord(tt)=card(t))..       sum(t, prec_res(tt,t,m)) =e= 0;

eq_completion_time(t)..  completion_time(t) =e= starting_time(t) + sum(m, assignment(t,m) * etc(t,m));

no_overlap(t,tt,m)..     starting_time(tt) - starting_time(t) =g= -BIG_M + (etc(t,m) + BIG_M) * precedence(t,tt,m);
*no_overlap(t,tt,m)..   starting_time(tt) =g= completion_time(t) - bigM*(1-precedence(t,tt,m));

t_precedence_1(t,tt,m)$(ord(t)>1 and ord(t)<card(t) and ord(tt)>1 and ord(tt)<card(t))..
prec_res(t,tt,m) =l= t_cores(t) * precedence(t,tt,m);

t_precedence_2(t,tt,m)$(ord(t)>1 and ord(t)<card(t) and ord(tt)>1 and ord(tt)<card(t))..
prec_res(t,tt,m) =l= t_cores(tt) * precedence(t,tt,m);

t_precedence_3(t,tt,m)..
precedence(t,tt,m) =l= assignment(t,m);

t_precedence_4(t,tt,m)..
precedence(t,tt,m) =l= assignment(tt,m);

t_precedence_5(tt,m)$(ord(tt)>1 and ord(tt)<card(t))..
sum(t, prec_res(t,tt,m)) =e= t_cores(tt) * assignment(tt,m);

t_precedence_6(t,m)$(ord(t)>1 and ord(t)<card(t))..
sum(tt, prec_res(t,tt,m)) =e= t_cores(t) * assignment(t,m);

enforce_arrival(t)..     starting_time(t) =g= t_arrival(t);

POSITIVE VARIABLE makespan       'makespan de la planificacion';
POSITIVE VARIABLE max_energy       'makespan de la planificacion';
POSITIVE VARIABLE idle_energy       'makespan de la planificacion';

EQUATION def_makespan(t)         'asigno un valor al makespan';
def_makespan(t)..                makespan =g= completion_time(t);

objective..                      f =e= makespan * sum(m, m_cores(m) * m_eidle(m)) + sum((t,m), assignment(t,m) * eec(t,m));
*sum((t,m), assignment(t,m) * etc(t,m) * eec(t,m)) + sum(m, m_cores(m) * makespan * m_eidle(m));

MODEL min_wqt /all/;

OPTION Reslim = 21600;
OPTION MIP = cplex;

min_wqt.optfile=1;

SOLVE min_wqt using mip minimizing f;

DISPLAY f.l;
DISPLAY starting_time.l;
DISPLAY completion_time.l;
DISPLAY assignment.l;
DISPLAY precedence.l;
DISPLAY prec_res.l;
DISPLAY makespan.l;

