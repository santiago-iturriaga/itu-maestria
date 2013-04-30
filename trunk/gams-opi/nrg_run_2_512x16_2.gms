$TITLE multicore_scheduling_min_wqt_s1

Option optcr = 0.0;

$Include params_2_512x16_2
$Include nrg_model_2_512x16

file salida_starting_time /m2_nrg_stime_512x16_2.txt/;
put salida_starting_time;
loop(t$(ord(t)>1 and ord(t)<card(t)), put starting_time.l(t):<:4 /);
putclose salida_starting_time;

file salida_assignment /m2_nrg_assign_512x16_2.txt/;
put salida_assignment;
loop((t,m)$(assignment.l(t,m) = 1 and ord(t)>1 and ord(t)<card(t)), put ord(m):<:0 /);
putclose salida_assignment;
