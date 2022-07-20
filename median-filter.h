#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <x86intrin.h>
#include <stdint.h>
#include <cassert>

constexpr uint64_t DIV      = 64;
constexpr uint64_t ZERO     = 0L;
constexpr uint64_t ONE      = 1L;
constexpr int scalingFactor = 2;


//returns number of 1's in the 64bit uns.integer
inline int countbits(uint64_t bitvector) {
    return __builtin_popcountll(bitvector); 
}

//returns Nset bit under pos index
inline uint64_t getNthBit(int N, uint64_t mask){
    return __builtin_ctzll(_pdep_u64(ONE << N, mask));
}

template <typename T>
struct ImageSubBlock {
    explicit ImageSubBlock(const T NX, const T NY, const T hx, const T hy) :

    imageSizeX(NX),
    imageSizeY(NY),

    SWlength(hx),
    SWheight(hy),

    kernelSizeX(hx * scalingFactor),
    kernelSizeY(hy * scalingFactor),

    blockside(CalculateSizeOfBlockSide()),
    blocksize(blockside * blockside),

    strideSizeX(CalculateSizeOfStride(kernelSizeX)),
    strideSizeY(CalculateSizeOfStride(kernelSizeY)),

    bsnx(CalculateBlockSizeN(NX, kernelSizeX, strideSizeX)),
    bsny(CalculateBlockSizeN(NY, kernelSizeY, strideSizeY))

    {}

    inline const T GetImageX()      const { return imageSizeX; }
    inline const T GetImageY()      const { return imageSizeY; }

    inline const T GetSWlength()    const { return SWlength; }
    inline const T GetSWheight()    const { return SWheight; }

    inline const T GetKernelSizeX() const { return kernelSizeX; }
    inline const T GetKernelSizeY() const { return kernelSizeY; }

    inline const T GetBlockSide()   const { return blockside; }
    inline const T GetBlockSize()   const { return blocksize; }

    inline const T GetStrideSizeX() const { return strideSizeX; }
    inline const T GetStrideSizeY() const { return strideSizeY; }

    inline const T GetBSNX()        const { return bsnx; }
    inline const T GetBSNY()        const { return bsny; }

private:

    inline T CalculateSizeOfBlockSide() {
        return scalingFactor * std::max(kernelSizeX, kernelSizeY);
    }

    inline T CalculateSizeOfStride(T kernelSize) {
        return blockside - kernelSize - 1;
    }

    inline T CalculateBlockSizeN(T N, T kernelSize, T strideSize) {
        return blockside < N ? (N - kernelSize + strideSize - 1) / strideSize : 1;
    }

    const T imageSizeX;
    const T imageSizeY;

    const T SWlength;
    const T SWheight;

    const T kernelSizeX;
    const T kernelSizeY;

    const T blockside;
    const T blocksize;

    const T strideSizeX;
    const T strideSizeY;

    const T bsnx;
    const T bsny; 


};

template <typename T>
struct BlockCoordinates {
    explicit BlockCoordinates(const ImageSubBlock<T>& Block, const int iy, const int ix) : 
    y0(ComputeOriginCoordinates(Block.GetStrideSizeY(), iy)),
    x0(ComputeOriginCoordinates(Block.GetStrideSizeX(), ix)),

    y1(ComputeUpperCoordinates(Block.GetImageY(), y0, Block.GetStrideSizeY(), Block.GetKernelSizeY())),
    x1(ComputeUpperCoordinates(Block.GetImageX(), x0, Block.GetStrideSizeX(), Block.GetKernelSizeX()))

    {}

    // const
    inline int GetY0() const { return y0; }
    inline int GetY1() const { return y1; }
    inline int GetX0() const { return x0; }
    inline int GetX1() const { return x1; }

    inline int GetRangeX() const { return x1 - x0; }
    inline int GetRangeY() const { return y1 - y0; }

private:
    inline int ComputeOriginCoordinates(const int stride, const int index) {
        return stride * index;
    }

    inline int ComputeUpperCoordinates(const int n, const int start, const int stride, const int kernel) {
        return std::min(n, start + kernel + stride);
    }

    const int y0;
    const int x0;
    const int y1;
    const int x1;
};

/* 
(ny, nx) size of the image 
(hy, hx) size of sliding window
(in) pointer to the data
(out) pointer to median-fitlered data
*/
void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {

    // 1 Partition the image into blocks
    ImageSubBlock<int> Block(nx, ny, hx, hy);

    // 2 Compute median of each block in parallel
    #pragma omp parallel for schedule(dynamic, 1)
    for(int iy = 0; iy < Block.GetBSNY(); iy++) { //iterate over blocks
    	for (int ix = 0; ix < Block.GetBSNX(); ix++) { 

            BlockCoordinates Coords(Block, iy, ix);
            //__________________________________________________________________________________________________

            auto bs = Block.GetBlockSize();
            std::pair<float, int> *block_pixels; 
            block_pixels = new std::pair<float, int>[bs]; //blocksize
            std::vector<int> ordinals(bs); //blocksize

            // collect pixels into containers
            int localIndex = 0; // container index
    		for (int y = 0; y < Coords.GetRangeY(); y++) {
                for (int x = 0; x < Coords.GetRangeX(); x++) {
                    int globalIndex = x + Coords.GetX0() + (y + Coords.GetY0()) * nx; // index of the pixel on image / global index
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

            //________________________________________________________________________________________________

            //size of bitvector
            int sz = Coords.GetRangeX()*Coords.GetRangeY();
            int bvsize = (sz + DIV - 1) / DIV;

            uint64_t *bitvector; 
            bitvector = new uint64_t[bvsize];

            //mind the overlap
            int sty = (iy == 0) ? 0 : hy; 
            int stx = (ix == 0) ? 0 : hx; 
            int eny = (iy == Block.GetBSNY() - 1) ? Coords.GetRangeY() : Coords.GetRangeY() - hy;
            //int eny = (iy == bsny - 1) ? y_range : y_range - hy;
            int enx = (ix == Block.GetBSNX() - 1) ? Coords.GetRangeX() : Coords.GetRangeX() - hx;
            //int enx = (ix == bsnx - 1) ? x_range : x_range - hx;

            for (int y = sty; y < eny; y++) {
                //init bitvector with zeros
                for(int r = 0; r < bvsize; r++){
                    bitvector[r] = ZERO;
                }

                //set values near initial position of running window
                int y_str = std::max(y - hy, 0); 
                int x_str = std::max(stx - hx, 0); 
                int y_end = std::min(y + hy + 1, Coords.GetRangeY());
                int x_end = std::min(stx + hx + 1, Coords.GetRangeX());

                //set bit to 1 if it is inside running window
                for (int i = y_str; i < y_end; i++) {
                    for(int j = x_str; j < x_end; j++){
                        int ind = j + i * Coords.GetRangeX();
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
                    int ey = std::min(Coords.GetRangeY(), y + hy + 1);
                    int ex = std::min(Coords.GetRangeX(), x + hx + 1);

                    //size of running window
                    int window_size = (ey-sy)*(ex-sx);

                    int sxlr = std::max(x - hx - 1, 0);
                    int exlr = std::max(x - hx, 0);
                    int ii = std::min(x + hx, ex - 1);

                    //unset left most vertical bits inside running window
                    for (int i = sxlr; i < exlr; i++) {
                        for (int j = sy; j < ey; j++) {
                            int ind = sxlr + j * Coords.GetRangeX();
                            int position = ordinals[ind];
                            int bitix = position / DIV;
                            int shift = DIV - position % DIV - 1;
                            bitvector[bitix] &= ~(ONE << shift);
                        }
                    }

                    //set right most vertical bits inside running window
                    for(int j = sy; j < ey; j++) {
                        int ind = ii + j * Coords.GetRangeX();
                        int position = ordinals[ind];
                        int bitix = position / DIV;
                        int shift = DIV - position % DIV - 1;
                        bitvector[bitix] |= (ONE << shift);
                    }

                    //compute median of the sliding window from the bit vector and set to result
                    //int globalIndex = x + x0block + (y + y0block) * nx;
                    int globalIndex = x + Coords.GetX0() + (y + Coords.GetY0()) * nx;
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