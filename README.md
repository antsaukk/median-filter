# **Fast median-filter** 
### For description of the problem please see [Wikipedia](https://en.wikipedia.org/wiki/Median_filter).
#### *Note: this is working solution, which was extensively tested and tests were probided by [PPC](https://ppc.cs.aalto.fi/), where this solution was developed to take part in the programming contest. Author-developed testing and benchmarking infrastructure will be added later, as the project is in WIP stage. Author can provide additional information on running times.*

### This algorithm achieves significant speed up in comparison to naive solution due to implemetation of entirely new algorithm, which:
- efficiently uses memory hierarchies by splitting image into the blocks
- exploits multhtreading to process each block in parallel with help of OpenMP
- utilizes "bitvector" compressed storage scheme, which allows to keep most of the data in closest caches
- does not need to compute sliding window for every pixel due to pixel-to-ordinal mapping and effective resetting of only 2k+1 bits inside the sliding window

### Potential speed-up improvements: 
- Exloit use of vector registers
- Implementation of vertical sliding-window scanning scheme instead of horizontal may improve memory access pattern
- Median does not have to be recalculated everytime sliding window moves to the next pixel

### Todo: 
- Add behcmark results
- Add test results
