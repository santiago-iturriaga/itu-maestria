* Model for scheduling of tasks in multicore
* machine minimizing the weighted queued time (WQT)

SET t 'tasks';
SET m 'machines';
SET c 'cores';

alias(t,tt);
alias(m,mm);
alias(c,cc);

PARAMETER m_cant_cores(m) 'cantidad de cores de la maquina m';
PARAMETER m_cores(m, c) 'cores de la maquina m';
PARAMETER t_arrival(t) 'arribo (en segundos) de la tarea t';
PARAMETER t_cores(t) 'cores requeridos por la tarea t';
PARAMETER t_priorities(t) 'prioridad de la tarea t (1 = tarea menos prioritaria, 5 tarea mas prioritaria)';
PARAMETER etc(t,m) 'costo computacional por core de la ejecucion de la tarea t en la maquina m';
PARAMETER m_eidle(m) 'consumo por unidad de tiempo de la maquina m en estado ocioso';
PARAMETER eec(t,m) 'costo energetico de la ejecucion de la tarea t en la maquina m';

VARIABLE f objetivo;
POSITIVE VARIABLE starting_time(t)       'comienzo de ejecucion de la tarea t';
POSITIVE VARIABLE completion_time(t)     'tiempo de finalizacion de la tarea t';
BINARY VARIABLE assignment_m(t,m)        'asignacion de la tarea t en la maquina m';
BINARY VARIABLE assignment_mc(t,m,c)     'asignacion de la tarea t en la maquina m en el core c';
BINARY VARIABLE precedence(t,tt,m,c)     't es previa de tt en la maquina m, core c';
BINARY VARIABLE start(t,m,c)             'la tarea es la primera del core c en la maquina m';

scalar TMAX 'horizonte maximo de tiempo';

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
                                 - TMAX*(1-precedence(t,tt,m,c));

enforce_precedence(t,m,c)..      1 =e= sum(tt, precedence(t,tt,m,c))
                                         + (1 - assignment_mc(t,m,c));

enforce_precedence_2(tt,m,c)..   1 =e= sum(t, precedence(t,tt,m,c))
                                         + (1 - assignment_mc(tt,m,c));

enforce_start(m,c)..             1 =e= sum(t, start(t,m,c));

objective..                      f =e= sum(t, t_priorities(t) * (completion_time(t)-t_arrival(t)));

MODEL min_wqt /all/;
