#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <x86intrin.h>
#include <bitset>
#include <stdint.h>

constexpr uint64_t DIV = 64;
constexpr uint64_t ZERO = 0L;
constexpr uint64_t ONE = 1L;


//returns number of 1's in the 64bit uns.integer
inline int countbits(uint64_t bitvector) {
    return __builtin_popcountll(bitvector); 
}

//returns Nset bit under pos index
inline uint64_t getNthBit(int N, uint64_t mask){
    return __builtin_ctzll(_pdep_u64(ONE << N, mask));
}

/* 
(ny, nx) size of the image 
(hy, hx) size of sliding window
(in) pointer to the data
(out) pointer to median-fitlered data
*/
void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {

    // 1 PARTITIIONING
    constexpr int factor = 2;

    int kernelX = hx * factor;
    int kernelY = hy * factor;

    // 1.1 compute block size based on window size
    int blockside = factor * std::max(kernelX, kernelY);
    int blocksize = blockside * blockside;

    // 1.2 initialize block attributes 
    int strideY = blockside - kernelY - 1;
    int bsny = blockside < ny ? (ny - kernelY + strideY - 1) / strideY : 1;

    int strideX = blockside - kernelX - 1; 
    int bsnx = blockside < nx ? (nx - kernelX + strideX - 1) / strideX : 1;

    // 2 COMPUTE median of blocks
    #pragma omp parallel for schedule(dynamic, 1)
    for(int iy = 0; iy < bsny; iy++) { //iterate over blocks
    	for (int ix = 0; ix < bsnx; ix++) { 

            std::pair<float, int> *block_pixels; 
            block_pixels = new std::pair<float, int>[blocksize];
            std::vector<int> ordinals(blocksize);

            // coordinates inside the block on the pixel map
            int y0block = iy * strideY; //y-starting coordinate of the block on image
            int y1block = std::min(ny, y0block + strideY + kernelY); // y-ending coordinate of the block on image
            int y_range = y1block - y0block;


            int x0block = ix * strideX; //x-starting coordinate of the block on image
            int x1block = std::min(nx, x0block + strideX + kernelX); // x-ending coordinate of the block on image
            int x_range = x1block - x0block;

            // collect pixels into containers
            int localIndex = 0; // container index
    		for (int y = 0; y < y_range; y++) {
                for (int x = 0; x < x_range; x++) {
                    int globalIndex = x + x0block + (y + y0block) * nx; // index of the pixel on image / global index
                    block_pixels[localIndex] = std::make_pair(in[globalIndex], localIndex);
                    localIndex++;
                }
            }

            //sort pixels
            std::sort(block_pixels, block_pixels + localIndex);

            // compute ordinals
            for (int i = 0; i < localIndex; i++) {
                ordinals[block_pixels[i].second] = i;
            }

            //size of bitvector
            int sz = x_range*y_range;
            int bvsize = (sz + DIV - 1) / DIV;

            uint64_t *bitvector; 
            bitvector = new uint64_t[bvsize];

            //mind the overlap
            int sty = (iy == 0) ? 0 : hy; 
            int stx = (ix == 0) ? 0 : hx; 
            int eny = (iy == bsny - 1) ? y_range : y_range - hy;
            int enx = (ix == bsnx - 1) ? x_range : x_range - hx;

            for (int y = sty; y < eny; y++) {
                //init bitvector with zeros
                for(int r = 0; r < bvsize; r++){
                    bitvector[r] = ZERO;
                }

                //set values near initial position of running window
                int y_str = std::max(y - hy, 0); 
                int x_str = std::max(stx - hx, 0); 
                int y_end = std::min(y + hy + 1, y_range);
                int x_end = std::min(stx + hx + 1, x_range);

                //set bit to 1 if it is inside running window
                for (int i = y_str; i < y_end; i++) {
                    for(int j = x_str; j < x_end; j++){
                        int ind = j + i * x_range;
                        int position = ordinals[ind];
                        int bitix = position / DIV;
                        int shift = DIV - position % DIV - 1; 
                        bitvector[bitix] |= (ONE << shift);
                    }
                }
                for (int x = stx; x < enx; x++) {
                    //sliding window bounds
                    int sy = std::max(y - hy, 0);
                    int sx = std::max(x - hx, 0);
                    int ey = std::min(y_range, y + hy + 1);
                    int ex = std::min(x_range, x + hx + 1);

                    //size of running window
                    int window_size = (ey-sy)*(ex-sx);

                    int sxlr = std::max(x - hx - 1, 0);
                    int exlr = std::max(x - hx, 0);
                    int ii = std::min(x + hx, ex - 1);

                    //unset left most vertical bits inside running window
                    for (int i = sxlr; i < exlr; i++) {
                        for (int j = sy; j < ey; j++) {
                            int ind = sxlr + j * x_range;
                            int position = ordinals[ind];
                            int bitix = position / DIV;
                            int shift = DIV - position % DIV - 1;
                            bitvector[bitix] &= ~(ONE << shift);
                        }
                    }

                    //set right most vertical bits inside running window
                    for(int j = sy; j < ey; j++) {
                        int ind = ii + j * x_range;
                        int position = ordinals[ind];
                        int bitix = position / DIV;
                        int shift = DIV - position % DIV - 1;
                        bitvector[bitix] |= (ONE << shift);
                    }

                    //compute median of the sliding window from the bit vector and set to result
                    int globalIndex = x + x0block + (y + y0block) * nx;
                    if(window_size % 2 == 1) {
                        int position = window_size / 2 + 1;
                        int remainder = position; 
                        int d = 0;
                        while (remainder > 0) {
                            int bb = countbits(bitvector[d]);
                            remainder -= bb; 
                            d++;
                        }

                        int N = std::abs(remainder);
                        int ord = (d - 1) * DIV + (DIV - getNthBit(N, bitvector[d-1])); 
                        float median = block_pixels[ord - 1].first;
                        out[globalIndex] = median;
                        

                    } else {
                        int position1 = (window_size / 2);
                        int position2 = (window_size / 2) + 1;
                        int remainder = position1;
                        int d = 0; 
                        while(remainder > 0) {
                            int bb = countbits(bitvector[d]);
                            remainder -= bb; 
                            d++; 
                        }
                        int N1 = std::abs(remainder); 
                        int ord1 = (d-1) * DIV + (DIV - getNthBit(N1, bitvector[d-1]));
                        remainder = position2;
                        d = 0;
                        while(remainder > 0) {
                            int bb = countbits(bitvector[d]);
                            remainder -= bb; 
                            d++; 
                        }
                        int N2 = std::abs(remainder); 
                        int ord2 = (d-1) * DIV + (DIV - getNthBit(N2, bitvector[d-1]));
                        double median1 = block_pixels[ord1 - 1].first;
                        double median2 = block_pixels[ord2 - 1].first;
                        out[globalIndex] = (median1 + median2)/2;
                    }
                }
            }

            delete[] block_pixels;
            delete[] bitvector;
    	}
    }
}