$TITLE multicore_scheduling_min_wqt_8x2

$Include wqt_params_2_8x2
$Include wqt_model_2

file salida_starting_time /m2_starting_time_8x2.txt/;
put salida_starting_time;
loop(t$(ord(t)>1 and ord(t)<card(t)), put starting_time.l(t):<:4 /);
putclose salida_starting_time;

file salida_assignment /m2_assignment_8x2.txt/;
put salida_assignment;
loop((t,m)$(assignment.l(t,m) = 1 and ord(t)>1 and ord(t)<card(t)), put ord(m):<:0 /);
putclose salida_assignment;