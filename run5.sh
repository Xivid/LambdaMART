for i in $(seq 17 1 20)
do
    echo "" > out_$i
    for num in $(seq 1 1 3)
    do
        ./cmake-build-debug/lambdamart tests/perf/mslr.$i.conf >> out_$i
    done
done

echo "" > out_21
./cmake-build-debug/lambdamart tests/perf/mslr.21.conf >> out_21