# Convierte el frente de Pareto resultado de cada ejecución
# a un formato más limpio (.out -> .pf)

dims=( 100 200 300 )
cover=( 40 60 80 )
nexec=20

for (( d=0; d<3; d++ )) do
    for (( c=0; c<3; c++ )) do
        for (( e=0; e<nexec; e++ )) do
            cmd="python convert2.py mls-eval-d${dims[d]}-c${cover[c]}/mls-eval-d${dims[d]}-c${cover[c]}.${e}.out 0"
            echo $cmd
            
            ${cmd} > mls-eval-d${dims[d]}-c${cover[c]}/mls-eval-d${dims[d]}-c${cover[c]}.${e}.pf
        done
    done
done

