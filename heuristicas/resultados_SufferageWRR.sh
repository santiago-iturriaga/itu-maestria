ejecutable="./SufferageWRR"

camino="/home/santiago/eclipse/c-c++-workspace/AE/ProblemInstances/HCSP/Braun_et_al.CPrio/"
for archivo in $(ls ${camino})
do
	${ejecutable} ${camino}/${archivo}
done

echo ""
echo ""
echo ""

camino="/home/santiago/eclipse/c-c++-workspace/AE/ProblemInstances/HCSP/1024x32.CPrio/"
for archivo in $(ls ${camino})
do
	${ejecutable} ${camino}/${archivo}
done

echo ""
echo ""
echo ""

camino="/home/santiago/eclipse/c-c++-workspace/AE/ProblemInstances/HCSP/2048x64.CPrio/"
for archivo in $(ls ${camino})
do
	${ejecutable} ${camino}/${archivo}
done

echo ""
echo ""
echo ""

camino="/home/santiago/eclipse/c-c++-workspace/AE/ProblemInstances/HCSP/4096x128.CPrio/"
for archivo in $(ls ${camino})
do
	${ejecutable} ${camino}/${archivo}
done