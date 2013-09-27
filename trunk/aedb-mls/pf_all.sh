./fp-eval.py mls-eval-d100-c40/mls-eval-d100-c40 20 10 > pf-d100-c40.log
./fp-eval.py mls-eval-d100-c60/mls-eval-d100-c60 20 15 > pf-d100-c60.log
./fp-eval.py mls-eval-d100-c80/mls-eval-d100-c80 20 20 > pf-d100-c80.log

./fp-eval.py mls-eval-d200-c40/mls-eval-d200-c40 20 20 > pf-d200-c40.log
./fp-eval.py mls-eval-d200-c60/mls-eval-d200-c60 20 30 > pf-d200-c60.log
./fp-eval.py mls-eval-d200-c80/mls-eval-d200-c80 20 40 > pf-d200-c80.log

./fp-eval.py mls-eval-d300-c40/mls-eval-d300-c40 10 30 > pf-d300-c40.log
./fp-eval.py mls-eval-d300-c60/mls-eval-d300-c60 10 45 > pf-d300-c60.log
./fp-eval.py mls-eval-d300-c80/mls-eval-d300-c80 10 60 > pf-d300-c80.log

./fp-best.py best_pf/All3Algs100dev-c40.pf pf-d100-c40.log > best-pf-d100-c40.pf
./fp-best.py best_pf/All3Algs100dev-c60.pf pf-d100-c60.log > best-pf-d100-c60.pf
./fp-best.py best_pf/All3Algs100dev-c80.pf pf-d100-c80.log > best-pf-d100-c80.pf

./fp-best.py best_pf/All3Algs200dev-c40.pf pf-d200-c40.log > best-pf-d200-c40.pf
./fp-best.py best_pf/All3Algs200dev-c60.pf pf-d200-c60.log > best-pf-d200-c60.pf
./fp-best.py best_pf/All3Algs200dev-c80.pf pf-d200-c80.log > best-pf-d200-c80.pf

./fp-best.py best_pf/All3Algs300dev-c40.pf pf-d300-c40.log > best-pf-d300-c40.pf
./fp-best.py best_pf/All3Algs300dev-c60.pf pf-d300-c60.log > best-pf-d300-c60.pf
./fp-best.py best_pf/All3Algs300dev-c80.pf pf-d300-c80.log > best-pf-d300-c80.pf
