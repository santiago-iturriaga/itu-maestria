(/home/siturria/bin/valgrind/bin/valgrind --tool=callgrind bin/pals_cpu_prof_aga_mt instancias/1024x32.ME/scenario.0 instancias/1024x32.ME/workload.A.u_c_hihi 1024 32 1 1 0 1000 100000 34) &> mt-1.txt
(/home/siturria/bin/valgrind/bin/valgrind --tool=callgrind bin/pals_cpu_prof_aga_mt instancias/1024x32.ME/scenario.0 instancias/1024x32.ME/workload.A.u_c_hihi 1024 32 1 2 0 1000 100000 34) &> mt-2.txt
(/home/siturria/bin/valgrind/bin/valgrind --tool=callgrind bin/pals_cpu_prof_aga_mt instancias/1024x32.ME/scenario.0 instancias/1024x32.ME/workload.A.u_c_hihi 1024 32 1 4 0 1000 100000 34) &> mt-4.txt

#-----------------------

(/home/siturria/bin/valgrind/bin/valgrind --tool=callgrind bin/pals_cpu_prof_aga_randr instancias/1024x32.ME/scenario.0 instancias/1024x32.ME/workload.A.u_c_hihi 1024 32 1 1 0 1000 100000 34) &> randr-1.txt
(/home/siturria/bin/valgrind/bin/valgrind --tool=callgrind bin/pals_cpu_prof_aga_randr instancias/1024x32.ME/scenario.0 instancias/1024x32.ME/workload.A.u_c_hihi 1024 32 1 2 0 1000 100000 34) &> randr-2.txt
(/home/siturria/bin/valgrind/bin/valgrind --tool=callgrind bin/pals_cpu_prof_aga_randr instancias/1024x32.ME/scenario.0 instancias/1024x32.ME/workload.A.u_c_hihi 1024 32 1 4 0 1000 100000 34) &> randr-4.txt

#-----------------------

(/home/siturria/bin/valgrind/bin/valgrind --tool=callgrind bin/pals_cpu_prof_aga_drand48r instancias/1024x32.ME/scenario.0 instancias/1024x32.ME/workload.A.u_c_hihi 1024 32 1 1 0 1000 100000 34) &> drand48r-1.txt
(/home/siturria/bin/valgrind/bin/valgrind --tool=callgrind bin/pals_cpu_prof_aga_drand48r instancias/1024x32.ME/scenario.0 instancias/1024x32.ME/workload.A.u_c_hihi 1024 32 1 2 0 1000 100000 34) &> drand48r-2.txt
(/home/siturria/bin/valgrind/bin/valgrind --tool=callgrind bin/pals_cpu_prof_aga_drand48r instancias/1024x32.ME/scenario.0 instancias/1024x32.ME/workload.A.u_c_hihi 1024 32 1 4 0 1000 100000 34) &> drand48r-4.txt
