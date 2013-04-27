$TITLE multicore_scheduling_min_wqt_s1

Option optcr = 0.0;

$Include params_2_32x4_3
$Include wqt_model_2

file salida_starting_time /m2_wqt_stime_32x4_3.txt/;
put salida_starting_time;
loop(t$(ord(t)>1 and ord(t)<card(t)), put starting_time.l(t):<:4 /);
putclose salida_starting_time;

file salida_assignment /m2_wqt_assign_32x4_3.txt/;
put salida_assignment;
loop((t,m)$(assignment.l(t,m) = 1 and ord(t)>1 and ord(t)<card(t)), put ord(m):<:0 /);
putclose salida_assignment;
