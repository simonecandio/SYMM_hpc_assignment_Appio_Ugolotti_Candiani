#!/bin/bash

make EXT_CFLAGS="-D POLYBENCH_TIME -D STANDARD_DATASET" clean all run
#-D POLYBENCH_DUMP_ARRAYS
for i in {1..5}
do
	./symm_acc
done
