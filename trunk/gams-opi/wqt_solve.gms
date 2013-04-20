scalar TMAX;
TMAX = sum(t,smax(m,etc(t,m)));

OPTION Reslim=2000;
OPTION MIP = cplex;
SOLVE min_wqt using mip minimizing f;

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
