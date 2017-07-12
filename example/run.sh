#!/bin/bash

#wget -qO- http://hltshare.fbk.eu/IWSLT2012/training-monolingual-europarl.tgz | tar xvz
rm -rf ./out2
echo "Step 1/3"
mkdir ./out2
grep -v " '[^ ]" ../data/training-monolingual-europarl/europarl-v7.en | \
grep -v \'\ s\   | \
grep -v \'\ ll\  | \
grep -v \'\ ve\  | \
grep -v \'\ m\   > step1.txt
echo "Step 2/3"
python dont_run_me_run_the_other_script_instead2.py step1.txt step2.txt
echo "Step 3/3"
head -n -400000 step2.txt > ./out2/ep.train.txt
tail -n 400000 step2.txt > step3.txt
head -n -200000 step3.txt > ./out2/ep.dev.txt
tail -n 200000 step3.txt > ./out2/ep.test.txt
echo "Cleaning up..."
rm -f step1.txt step2.txt step3.txt
echo "Preprocessing done. Now you can give the produced ./out dir as <data_dir> argument to data.py script for conversion and continue as described in the main README.md"
