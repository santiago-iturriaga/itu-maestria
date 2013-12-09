function do_compute {
    ./${METRIC} best-pf-d100-c40.pf ${MOEA}/${MOEA_ALG}/100dev/FUN 30 mls-eval-d100-c40/mls-eval-d100-c40 20 10 > ${OUTPUT}-d100-c40.log
    ./${METRIC} best-pf-d100-c60.pf ${MOEA}/${MOEA_ALG}/100dev/FUN 30 mls-eval-d100-c60/mls-eval-d100-c60 20 15 > ${OUTPUT}-d100-c60.log
    ./${METRIC} best-pf-d100-c80.pf ${MOEA}/${MOEA_ALG}/100dev/FUN 30 mls-eval-d100-c80/mls-eval-d100-c80 20 20 > ${OUTPUT}-d100-c80.log

    ./${METRIC} best-pf-d200-c40.pf ${MOEA}/${MOEA_ALG}/200dev/FUN 30 mls-eval-d200-c40/mls-eval-d200-c40 20 20 > ${OUTPUT}-d200-c40.log
    ./${METRIC} best-pf-d200-c60.pf ${MOEA}/${MOEA_ALG}/200dev/FUN 30 mls-eval-d200-c60/mls-eval-d200-c60 20 30 > ${OUTPUT}-d200-c60.log
    ./${METRIC} best-pf-d200-c80.pf ${MOEA}/${MOEA_ALG}/200dev/FUN 30 mls-eval-d200-c80/mls-eval-d200-c80 20 40 > ${OUTPUT}-d200-c80.log

    ./${METRIC} best-pf-d300-c40.pf ${MOEA}/${MOEA_ALG}/300dev/FUN 30 mls-eval-d300-c40/mls-eval-d300-c40 20 30 > ${OUTPUT}-d300-c40.log
    ./${METRIC} best-pf-d300-c60.pf ${MOEA}/${MOEA_ALG}/300dev/FUN 30 mls-eval-d300-c60/mls-eval-d300-c60 20 45 > ${OUTPUT}-d300-c60.log
    ./${METRIC} best-pf-d300-c80.pf ${MOEA}/${MOEA_ALG}/300dev/FUN 30 mls-eval-d300-c80/mls-eval-d300-c80 20 60 > ${OUTPUT}-d300-c80.log
}

MOEA="/home/santiago/Dropbox/Facultad/Publicaciones/EvoComnetAEDB/IJAHUC/NIDISC/comparison/data"

METRIC="epsilon-compare.py"
OUTPUT="epsilon-kw-cellde"
MOEA_ALG="CellDE"

do_compute

OUTPUT="epsilon-kw-nsgaii"
MOEA_ALG="NSGAII"

do_compute

METRIC="hv-compare.py"
OUTPUT="hv-kw-cellde.log"
MOEA_ALG="CellDE"

do_compute

OUTPUT="hv-kw-nsgaii.log"
MOEA_ALG="NSGAII"

do_compute

METRIC="hv-compare-2.py"
MOEA_ALG1="CellDE"
MOEA_ALG2="NSGAII"

./${METRIC} best-pf-d100-c40.pf ${MOEA}/${MOEA_ALG1}/100dev/FUN 30 ${MOEA}/${MOEA_ALG2}/100dev/FUN 30 10 >> ${OUTPUT}-d100-c40.log
./${METRIC} best-pf-d100-c60.pf ${MOEA}/${MOEA_ALG1}/100dev/FUN 30 ${MOEA}/${MOEA_ALG2}/100dev/FUN 30 15 >> ${OUTPUT}-d100-c60.log
./${METRIC} best-pf-d100-c80.pf ${MOEA}/${MOEA_ALG1}/100dev/FUN 30 ${MOEA}/${MOEA_ALG2}/100dev/FUN 30 20 >> ${OUTPUT}-d100-c80.log

./${METRIC} best-pf-d200-c40.pf ${MOEA}/${MOEA_ALG1}/200dev/FUN 30 ${MOEA}/${MOEA_ALG2}/200dev/FUN 30 20 >> ${OUTPUT}-d200-c40.log
./${METRIC} best-pf-d200-c60.pf ${MOEA}/${MOEA_ALG1}/200dev/FUN 30 ${MOEA}/${MOEA_ALG2}/200dev/FUN 30 30 >> ${OUTPUT}-d200-c60.log
./${METRIC} best-pf-d200-c80.pf ${MOEA}/${MOEA_ALG1}/200dev/FUN 30 ${MOEA}/${MOEA_ALG2}/200dev/FUN 30 40 >> ${OUTPUT}-d200-c80.log

./${METRIC} best-pf-d300-c40.pf ${MOEA}/${MOEA_ALG1}/300dev/FUN 30 ${MOEA}/${MOEA_ALG2}/300dev/FUN 30 30 >> ${OUTPUT}-d300-c40.log
./${METRIC} best-pf-d300-c60.pf ${MOEA}/${MOEA_ALG1}/300dev/FUN 30 ${MOEA}/${MOEA_ALG2}/300dev/FUN 30 45 >> ${OUTPUT}-d300-c60.log
./${METRIC} best-pf-d300-c80.pf ${MOEA}/${MOEA_ALG1}/300dev/FUN 30 ${MOEA}/${MOEA_ALG2}/300dev/FUN 30 60 >> ${OUTPUT}-d300-c80.log

METRIC="igd-compare.py"
OUTPUT="igd-kw-cellde.log"
MOEA_ALG="CellDE"

do_compute

OUTPUT="igd-kw-nsgaii.log"
MOEA_ALG="NSGAII"

do_compute
