#Sintaxis: ./generator <num_tasks> <num_machines> <task_heterogeneity> <machine_heterogeneity> <consistency> [model] [type] [seed]
#Task heterogeneity levels: (0-Low, 1-High), machine heterogeneity levels: (0-Low, 1-High).
#Consistency type: (0-Consistent, 1-Semiconsistent, 2-Inconsistent).
#Optional: heterogeneity model: (0-Ali et al., 1-Braun et al.).
#	Ranks (tasks, machines) 0(Ali):(10-100000,10-100), 1(Braun):(100-3000,10-1000).
#	(ranks by Braun et al. (100-3000,10-1000) assumed by default).
#Optional: type of task execution times: (0-real, 1-integer).
#Optional: seed for the pseudorandom number generator.

mkdir 4096x128.M.c
cd 4096x128.M.c

#A.u_c_hihi
../generator 4096 128 1 1 0 0 0 1214389178
mv A.u_c_hihi A.u_c_hihi_0
../generator 4096 128 1 1 0 0 0 1214389179
mv A.u_c_hihi A.u_c_hihi_1
../generator 4096 128 1 1 0 0 0 1214389180
mv A.u_c_hihi A.u_c_hihi_2

#A.u_c_hilo
../generator 4096 128 1 0 0 0 0 1214389181 
mv A.u_c_hilo A.u_c_hilo_0
../generator 4096 128 1 0 0 0 0 1214389182
mv A.u_c_hilo A.u_c_hilo_1
../generator 4096 128 1 0 0 0 0 1214389183
mv A.u_c_hilo A.u_c_hilo_2

#A.u_c_lohi
../generator 4096 128 0 1 0 0 0 1214389184 
mv A.u_c_lohi A.u_c_lohi_0
../generator 4096 128 0 1 0 0 0 1214389185
mv A.u_c_lohi A.u_c_lohi_1
../generator 4096 128 0 1 0 0 0 1214389186
mv A.u_c_lohi A.u_c_lohi_2

#A.u_c_lolo
../generator 4096 128 0 0 0 0 0 1214389187 
mv A.u_c_lolo A.u_c_lolo_0
../generator 4096 128 0 0 0 0 0 1214389188
mv A.u_c_lolo A.u_c_lolo_1
../generator 4096 128 0 0 0 0 0 1214389189
mv A.u_c_lolo A.u_c_lolo_2

#B.u_c_hihi
../generator 4096 128 1 1 0 1 0 1214389190 
mv B.u_c_hihi B.u_c_hihi_0
../generator 4096 128 1 1 0 1 0 1214389191
mv B.u_c_hihi B.u_c_hihi_1
../generator 4096 128 1 1 0 1 0 1214389192
mv B.u_c_hihi B.u_c_hihi_2

#B.u_c_hilo
../generator 4096 128 1 0 0 1 0 1214389193 
mv B.u_c_hilo B.u_c_hilo_0
../generator 4096 128 1 0 0 1 0 1214389194
mv B.u_c_hilo B.u_c_hilo_1
../generator 4096 128 1 0 0 1 0 1214389195
mv B.u_c_hilo B.u_c_hilo_2

#B.u_c_lohi
../generator 4096 128 0 1 0 1 0 1214389196
mv B.u_c_lohi B.u_c_lohi_0
../generator 4096 128 0 1 0 1 0 1214389197 
mv B.u_c_lohi B.u_c_lohi_1
../generator 4096 128 0 1 0 1 0 1214389198 
mv B.u_c_lohi B.u_c_lohi_2

#B.u_c_lolo
../generator 4096 128 0 0 0 1 0 1214389199 
mv B.u_c_lolo B.u_c_lolo_0
../generator 4096 128 0 0 0 1 0 1214389200
mv B.u_c_lolo B.u_c_lolo_1
../generator 4096 128 0 0 0 1 0 1214389201
mv B.u_c_lolo B.u_c_lolo_2

rm *.log
