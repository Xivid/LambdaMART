#!/bin/bash

create_conf()
{
    echo "train_data:data/MSLR-WEB30K/$1.train" > tmp.$1.conf
    echo "train_query:data/MSLR-WEB30K/$1.train.query" >> tmp.$1.conf
    echo "valid_data:data/MSLR-WEB30K/17.vali" >> tmp.$1.conf
    echo "valid_query:data/MSLR-WEB30K/17.vali.query" >> tmp.$1.conf
    echo "num_iterations:100" >> tmp.$1.conf
    echo "learning_rate:0.1" >> tmp.$1.conf
    echo "verbosity:1" >> tmp.$1.conf
    echo "max_depth:9" >> tmp.$1.conf
    echo "max_splits:256" >> tmp.$1.conf
    echo "eval_at:1,3,5,10" >> tmp.$1.conf
    echo "eval_interval:10" >> tmp.$1.conf
}


mkdir -p logs

for i in $(seq 10 1 21)
do
    create_conf $i
    ./lambdamart tmp.$i.conf &> logs/mslr.$i.log
    rm tmp.$i.conf
done
