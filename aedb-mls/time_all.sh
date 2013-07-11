echo "" > time.log

echo "d100-c10" >> time.log
./time_eval.py mls-eval-d100-c10/mls-eval-d100-c10 20 >> time.log
echo "d100-c15" >> time.log
./time_eval.py mls-eval-d100-c15/mls-eval-d100-c15 20 >> time.log
echo "d100-c20" >> time.log
./time_eval.py mls-eval-d100-c20/mls-eval-d100-c20 20 >> time.log

echo "d200-c10" >> time.log
./time_eval.py mls-eval-d200-c10/mls-eval-d200-c10 20 >> time.log
echo "d200-c15" >> time.log
./time_eval.py mls-eval-d200-c15/mls-eval-d200-c15 20 >> time.log
echo "d200-c20" >> time.log
./time_eval.py mls-eval-d200-c20/mls-eval-d200-c20 20 >> time.log
