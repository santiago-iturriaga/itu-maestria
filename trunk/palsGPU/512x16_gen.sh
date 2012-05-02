#Sintaxis: ./generator <num_tasks> <num_machines> <task_heterogeneity> <machine_heterogeneity> <consistency> [model] [type] [seed]
#Task heterogeneity levels: (0-Low, 1-High), machine heterogeneity levels: (0-Low, 1-High).
#Consistency type: (0-Consistent, 1-Semiconsistent, 2-Inconsistent).
#Optional: heterogeneity model: (0-Ali et al., 1-Braun et al.).
#	Ranks (tasks, machines) 0(Ali):(10-100000,10-100), 1(Braun):(100-3000,10-1000).
#	(ranks by Braun et al. (100-3000,10-1000) assumed by default).
#Optional: type of task execution times: (0-real, 1-integer).
#Optional: seed for the pseudorandom number generator.

mkdir -p 512x16/instances.src
cd 512x16/instances.src

#A.u_c_hihi
../../generator 512 16 1 1 0 0 0 1240440634
#A.u_c_hilo
../../generator 512 16 1 0 0 0 0 1240440631
#A.u_c_lohi
../../generator 512 16 0 1 0 0 0 1240440639
#A.u_c_lolo
../../generator 512 16 0 0 0 0 0 1240440550

#A.u_s_hihi
../../generator 512 16 1 1 1 0 0 1240440656
#A.u_s_hilo
../../generator 512 16 1 0 1 0 0 1240440659
#A.u_s_lohi
../../generator 512 16 0 1 1 0 0 1240440665
#A.u_s_lolo
../../generator 512 16 0 0 1 0 0 1240440662

#A.u_u_hihi
../../generator 512 16 1 1 2 0 0 1240440654
#A.u_u_hilo
../../generator 512 16 1 0 2 0 0 1240440649
#A.u_u_lohi
../../generator 512 16 0 1 2 0 0 1240440643
#A.u_u_lolo
../../generator 512 16 0 0 2 0 0 1240440646
