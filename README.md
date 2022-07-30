# **Fast median-filter** 
### For description of the problem please see [Wikipedia](https://en.wikipedia.org/wiki/Median_filter).
#### *Note: this is working solution, which was extensively tested and tests were probided by [PPC](https://ppc.cs.aalto.fi/), where this solution was developed to take part in the programming contest. Some test cases and benchmarks are also added to this version.

### This algorithm achieves significant speed up in comparison to naive solution due to implemetation of entirely new algorithm, which:
- efficiently uses memory hierarchies, by splitting image into the blocks
- exploits multhtreading to process each block in parallel with help of OpenMP
- utilizes "bitvector" compressed storage scheme, which allows to keep most of the data in closest caches
- does not need to compute sliding window for every pixel due to pixel-to-ordinal mapping and effective resetting of only 2k+1 bits inside the sliding window

### Execution: 
```
./build/compile.sh
./build/execute.sh
```

### Potential speed-up improvements: 
- Use of vector registers
- Implementation of vertical sliding-window scanning scheme, instead of horizontal, may improve memory access pattern
- Median does not have to be recalculated everytime sliding window moves to the next pixel

### Todo: 
- Add more tests + tests with libasan
- Split execution of tests and benchmarks
- Implement better refactoring and renaming
- Split to separate source and header files
