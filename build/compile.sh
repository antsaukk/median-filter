#!/bin/bash

if [ -d "bin/" ]
then
	rm -rf bin/
fi
mkdir bin
cd src
${CXX:-g++} -O3 -Werror -Wall -Wno-error=unknown-pragmas -Wno-error=unused-but-set-variable -Wno-error=unused-local-typedefs -Wno-error=unused-function -Wno-error=unused-label -Wno-error=unused-value -Wno-error=unused-variable -Wno-error=unused-parameter -Wno-error=unused-but-set-parameter --pedantic -std=c++17 -march=native -fopenmp -o ../bin/mf mf.cpp
# ${CXX:-g++} -O3 -Werror -Wall --pedantic -fsanitize=address -fsanitize=undefined -std=c++17 -march=native -fopenmp -o ../bin/mf mf.cpp