# **Fast median-filter** 
### For description of the problem please see [Wikipedia](https://en.wikipedia.org/wiki/Median_filter)
#### *Note: this is working solution, which was extensively tested and tests were probided by (https://ppc.cs.aalto.fi/), where this solution was developed to take part in the programming contest. Author-developed testsing and benchmarking infrastructure will be added later, as the project is in WIP stage. Author can provide additional information on running times.*

### This algorithm achieves significant speed up in comparison to naive solution due to implemetation of entirely new algorithm, which:
- does not need to compute sliding window for every pixel
- efficiently uses memory hierarchies by splitting image into the blocks
- exploits multhtreading to process each block in parallel with help of OpenMP
- utilizes "bitvector" compressed storage scheme, which allows to keep most of the data in closest caches
