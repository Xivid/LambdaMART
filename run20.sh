
for i in $(seq 10 1 16)
    do
        echo "" > out_$i
        for num in $(seq 1 1 10)
        do
            ./cmake-build-debug/lambdamart tests/perf/mslr.$i.conf >> out_$i
        done
    done
