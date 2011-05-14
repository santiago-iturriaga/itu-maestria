ejecutable="./SufferageMakespan"

camino="/home/santiago/eclipse/c-c++-workspace/AE/ProblemInstances/HCSP/Braun_et_al.CPrio/"
echo "INSTANCIA | NT | NM | MAKESPAN | WRR"
for archivo in $(ls ${camino})
do
	${ejecutable} ${camino}/${archivo}
done