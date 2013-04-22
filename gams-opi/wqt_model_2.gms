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

scalar bigM 'horizonte maximo de tiempo';
bigM = sum(t,smax(m,etc(t,m)));

EQUATION no_loop(t,tt,m)
EQUATION exec_all_1(t)                   'cada tarea se debe ejecutar en una maquina';
EQUATION exec_all_2(t,m)              'se deben ejecutar todos los cores requeridos por la tarea';
EQUATION m_resources_1(m)                   'cores disponibles en la maquina';
EQUATION m_resources_2(m)                 'cores disponibles en la maquina';
EQUATION m_resources_3(m)                 'cores disponibles en la maquina';
EQUATION m_resources_4(m)                 'cores disponibles en la maquina';
EQUATION eq_completion_time(t)           'asigno el completion time de cada tarea';
EQUATION no_overlap(t,tt,m)              'verifico que las tareas no se pisen';
EQUATION t_precedence_1(t,tt,m)          'toda tarea tiene que tener una unica tarea previa';
EQUATION t_precedence_2(t,tt,m)          'el previous es único dos tareas en un core no pueden tener el mismo previous';
EQUATION t_precedence_3(t,tt,m)          'el previous es único dos tareas en un core no pueden tener el mismo previous';
EQUATION t_precedence_4(t,tt,m)          'el previous es único dos tareas en un core no pueden tener el mismo previous';
EQUATION t_precedence_5(tt,m)          'el previous es único dos tareas en un core no pueden tener el mismo previous';
EQUATION t_precedence_6(t,m)          'el previous es único dos tareas en un core no pueden tener el mismo previous';
EQUATION enforce_arrival(t)              'toda tarea debe empezar luego de su arrival time';
EQUATION objective                       'evaluo el objetivo';

no_loop(t,tt,m)..          precedence(t,tt,m) + precedence(tt,t,m) =l= 1;

exec_all_1(t)$(ord(t)>1 and ord(t)<10)..          sum(m, assignment(t,m)) =e= 1;
exec_all_2(t,m)$(ord(t)=1 and ord(t)=10)..        assignment(t,m) =e= 1;

m_resources_1(m)..       sum(t, prec_res('0',t,m)) =l= m_cores(m);
m_resources_2(m)..       sum(t, prec_res(t,'0',m)) =e= 0;
m_resources_3(m)..       sum(t, prec_res(t,'9',m)) =e= m_cores(m);
m_resources_4(m)..       sum(t, prec_res('9',t,m)) =e= 0;

eq_completion_time(t)..  completion_time(t) =e= starting_time(t) + sum(m, assignment(t,m) * etc(t,m));

no_overlap(t,tt,m)..     starting_time(tt) - starting_time(t) =g= -bigM + (etc(t,m) + bigM) * precedence(t,tt,m);
*no_overlap(t,tt,m)..   starting_time(tt) =g= completion_time(t) - bigM*(1-precedence(t,tt,m));

t_precedence_1(t,tt,m)$(ord(t)>1 and ord(t)<10 and ord(tt)>1 and ord(tt)<10)..
prec_res(t,tt,m) =l= t_cores(t) * precedence(t,tt,m);

t_precedence_2(t,tt,m)$(ord(t)>1 and ord(t)<10 and ord(tt)>1 and ord(tt)<10)..
prec_res(t,tt,m) =l= t_cores(tt) * precedence(t,tt,m);

t_precedence_3(t,tt,m)..
precedence(t,tt,m) =l= assignment(t,m);

t_precedence_4(t,tt,m)..
precedence(t,tt,m) =l= assignment(tt,m);

t_precedence_5(tt,m)$(ord(tt)>1 and ord(tt)<10)..
sum(t, prec_res(t,tt,m)) =e= t_cores(tt) * assignment(tt,m);

t_precedence_6(t,m)$(ord(t)>1 and ord(t)<10)..
sum(tt, prec_res(t,tt,m)) =e= t_cores(t) * assignment(t,m);

enforce_arrival(t)..     starting_time(t) =g= t_arrival(t);

objective..              f =e= sum(t, t_priorities(t) * (completion_time(t)-t_arrival(t)));

MODEL min_wqt /all/;

OPTION Reslim = 1000000;
OPTION MIP = cplex;
SOLVE min_wqt using mip minimizing f;

DISPLAY f.l;
DISPLAY starting_time.l;
DISPLAY completion_time.l;
DISPLAY assignment.l;
DISPLAY precedence.l;
DISPLAY prec_res.l;

file salida_starting_time /m2_starting_time.txt/;
put salida_starting_time;
loop(t$(ord(t)>1 and ord(t)<10), put starting_time.l(t):<:4 /);
putclose salida_starting_time;

file salida_assignment /m2_assignment.txt/;
put salida_assignment;
loop((t,m)$(assignment.l(t,m) = 1 and ord(t)>1 and ord(t)<10), put ord(m):<:0 /);
putclose salida_assignment;