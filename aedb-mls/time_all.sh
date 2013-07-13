echo "" > time.log

echo "d100-c40" >> time.log
./time_eval.py mls-eval-d100-c40/mls-eval-d100-c10 20 >> time.log
echo "d100-c60" >> time.log
./time_eval.py mls-eval-d100-c60/mls-eval-d100-c15 20 >> time.log
echo "d100-c80" >> time.log
./time_eval.py mls-eval-d100-c80/mls-eval-d100-c20 20 >> time.log

echo "d200-c40" >> time.log
./time_eval.py mls-eval-d200-c40/mls-eval-d200-c20 20 >> time.log
echo "d200-c60" >> time.log
./time_eval.py mls-eval-d200-c60/mls-eval-d200-c60 5 >> time.log
echo "d200-c80" >> time.log
./time_eval.py mls-eval-d200-c80/mls-eval-d200-c80 5 >> time.log
