#Sintaxis: ./generator <num_tasks> <num_machines> <task_heterogeneity> <machine_heterogeneity> <consistency> [model] [type] [seed]
#Task heterogeneity levels: (0-Low, 1-High), machine heterogeneity levels: (0-Low, 1-High).
#Consistency type: (0-Consistent, 1-Semiconsistent, 2-Inconsistent).
#Optional: heterogeneity model: (0-Ali et al., 1-Braun et al.).
#	Ranks (tasks, machines) 0(Ali):(10-100000,10-100), 1(Braun):(100-3000,10-1000).
#	(ranks by Braun et al. (100-3000,10-1000) assumed by default).
#Optional: type of task execution times: (0-real, 1-integer).
#Optional: seed for the pseudorandom number generator.

#A.u_c_hihi
../generator 2048 64 1 1 0 0 0 1214389178
#A.u_c_hilo
../generator 2048 64 1 0 0 0 0 1214389175 
#A.u_c_lohi
../generator 2048 64 0 1 0 0 0 1214389170 
#A.u_c_lolo
../generator 2048 64 0 0 0 0 0 1214389159 

#A.u_s_hihi
../generator 2048 64 1 1 1 0 0 1214389217 
#A.u_s_hilo
../generator 2048 64 1 0 1 0 0 1214389221 
#A.u_s_lohi
../generator 2048 64 0 1 1 0 0 1214389227 
#A.u_s_lolo
../generator 2048 64 0 0 1 0 0 1214389224 

#A.u_u_hihi
../generator 2048 64 1 1 2 0 0 1214389262 
#A.u_u_hilo
../generator 2048 64 1 0 2 0 0 1214389259 
#A.u_u_lohi
../generator 2048 64 0 1 2 0 0 1214389253 
#A.u_u_lolo
../generator 2048 64 0 0 2 0 0 1214389256 

#B.u_c_hihi
../generator 2048 64 1 1 0 1 0 1214389181 
#B.u_c_hilo
../generator 2048 64 1 0 0 1 0 1214389187 
#B.u_c_lohi
../generator 2048 64 0 1 0 1 0 1214389194 
#B.u_c_lolo
../generator 2048 64 0 0 0 1 0 1214389191 

#B.u_s_hihi
../generator 2048 64 1 1 1 1 0 1214389214 
#B.u_s_hilo
../generator 2048 64 1 0 1 1 0 1214389211 
#B.u_s_lohi
../generator 2048 64 0 1 1 1 0 1214389207 
#B.u_s_lolo
../generator 2048 64 0 0 1 1 0 1214389204 

#B.u_u_hihi
../generator 2048 64 1 1 2 1 0 1214389267 
#B.u_u_hilo
../generator 2048 64 1 0 2 1 0 1214389271 
#B.u_u_lohi
../generator 2048 64 0 1 2 1 0 1214389278 
#B.u_u_lolo
../generator 2048 64 0 0 2 1 0 1214389281 

