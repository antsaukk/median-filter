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

struct ImageSubBlock {
    explicit ImageSubBlock(const int NX, const int NY, const int hx, const int hy) :

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

    inline int GetImageX()      const { return imageSizeX; }
    inline int GetImageY()      const { return imageSizeY; }

    inline int GetSWlength()    const { return SWlength; }
    inline int GetSWheight()    const { return SWheight; }

    inline int GetKernelSizeX() const { return kernelSizeX; }
    inline int GetKernelSizeY() const { return kernelSizeY; }

    inline int GetBlockSide()   const { return blockside; }
    inline int GetBlockSize()   const { return blocksize; }

    inline int GetStrideSizeX() const { return strideSizeX; }
    inline int GetStrideSizeY() const { return strideSizeY; }

    inline int GetBSNX()        const { return bsnx; }
    inline int GetBSNY()        const { return bsny; }

private:

    inline int CalculateSizeOfBlockSide() {
        return scalingFactor * std::max(kernelSizeX, kernelSizeY);
    }

    inline int CalculateSizeOfStride(int kernelSize) {
        return blockside - kernelSize - 1;
    }

    inline int CalculateBlockSizeN(int N, int kernelSize, int strideSize) {
        return blockside < N ? (N - kernelSize + strideSize - 1) / strideSize : 1;
    }

    const int imageSizeX;
    const int imageSizeY;

    const int SWlength;
    const int SWheight;

    const int kernelSizeX;
    const int kernelSizeY;

    const int blockside;
    const int blocksize;

    const int strideSizeX;
    const int strideSizeY;

    const int bsnx;
    const int bsny; 


};

// this depends on ImageSubBlock and it is bad
// remove this dependency and init only with correct arguments
struct BlockCoordinates {
    explicit BlockCoordinates(const ImageSubBlock& Block, const int iy, const int ix) : 
    y0(ComputeOriginCoordinates(Block.GetStrideSizeY(), iy)),
    x0(ComputeOriginCoordinates(Block.GetStrideSizeX(), ix)),

    y1(ComputeUpperCoordinates(Block.GetImageY(), y0, Block.GetStrideSizeY(), Block.GetKernelSizeY())),
    x1(ComputeUpperCoordinates(Block.GetImageX(), x0, Block.GetStrideSizeX(), Block.GetKernelSizeX())),

    NX(Block.GetImageX()),
    HX(Block.GetSWlength()), //?
    HY(Block.GetSWheight()), //?

    sty(ComputeStartCoordY(iy, Block.GetSWheight())),
    stx(ComputeStartCoordX(ix, Block.GetSWlength())),
    eny(ComputeEndCoordY(iy, Block.GetBSNY(), Block.GetSWheight())),
    enx(ComputeEndCoordX(ix, Block.GetBSNX(), Block.GetSWlength()))
    {}

    // const
    inline int GetY0() const { return y0; }
    inline int GetY1() const { return y1; }
    inline int GetX0() const { return x0; }
    inline int GetX1() const { return x1; }

    inline int GetRangeX() const { return x1 - x0; }
    inline int GetRangeY() const { return y1 - y0; }

    inline int GetNX() const { return NX; }
    inline int GetHY() const { return HY; }
    inline int GetHX() const { return HX; }

    inline int ComputeGlobalIndex(const int x, const int y) const {
        return x + GetX0() + (y + GetY0()) * GetNX();
    }

    inline int GetSTY() const { return sty; }
    inline int GetSTX() const { return stx; }
    inline int GetENY() const { return eny; }
    inline int GetENX() const { return enx; }

private:
    inline int ComputeOriginCoordinates(const int stride, const int index) {
        return stride * index;
    }

    inline int ComputeUpperCoordinates(const int n, const int start, const int stride, const int kernel) {
        return std::min(n, start + kernel + stride);
    }

    inline int ComputeStartCoordY(const int iy, const int hy) {
        return (iy == 0) ? 0 : hy;
    }

    inline int ComputeStartCoordX(const int ix, const int hx) {
        return (ix == 0) ? 0 : hx;
    }

    inline int ComputeEndCoordY(const int iy, const int bsny, const int hy) {
        return (iy == bsny - 1) ? GetRangeY() : GetRangeY() - hy;
    }

    inline int ComputeEndCoordX(const int ix, const int bsnx, const int hx) {
        return (ix == bsnx - 1) ? GetRangeX() : GetRangeX() - hx;
    }

    const int y0;
    const int x0;
    const int y1;
    const int x1;

    const int NX;

    const int HX;
    const int HY;

    const int sty;
    const int stx;
    const int eny;
    const int enx;
};

class TransformedData {
public:
    explicit TransformedData(const int blocksize) :
    block_pixels(blocksize),
    ordinals(blocksize),
    localIndex(0)
    {}

    inline float GetPixel(const int index) const { return block_pixels[index].first; }
    inline int GetOrdinal(const int index) const { return ordinals[index]; }

    inline void SetOrdinals(const int value, const int index) {
        ordinals[index] = value;
    }

    // also remove this dependency on coords
    inline void MapPixelsToOrdinals(const BlockCoordinates& coords, const float *in) {
        for (int y = 0; y < coords.GetRangeY(); y++) {
            for (int x = 0; x < coords.GetRangeX(); x++) {
                const int globalIndex = coords.ComputeGlobalIndex(x, y);
                block_pixels[localIndex] = std::make_pair(in[globalIndex], localIndex);
                IncreaseLocalIndex();
            }
        }
    }

    inline void InitializeOrdinals() {
        std::sort(block_pixels.begin(), block_pixels.begin() + localIndex);

        // compute ordinals
        for (int i = 0; i < localIndex; i++) {
            ordinals[block_pixels[i].second] = i;
        }
    }

private:
    inline void IncreaseLocalIndex() { localIndex++; }
    
    std::vector<std::pair<float, int>> block_pixels;
    std::vector<int> ordinals;

    int localIndex;
};

class BitVector {
public:
    explicit BitVector(const int blockSize) :
    bitvector((blockSize + DIV - 1) / DIV)
    {}

    inline void ReInitWithZeros() {
        for(size_t i = 0; i < bitvector.size(); i++)
            bitvector[i] = ZERO;
    }

    inline void SetOne(const int position) {
        const int bitix = position / DIV;
        const int shift = ComputeShift(position);
        bitvector[bitix] |= (ONE << shift);
    }

    inline void SetZero(const int position) {
        const int bitix = position / DIV;
        const int shift = ComputeShift(position);
        bitvector[bitix] &= ~(ONE << shift);
    }

    inline uint64_t GetBits(const int position) const { return bitvector[position]; }

private:
    inline int ComputeShift(const int position) const { //class member or separate function? 
        return DIV - position % DIV - 1;
    }

    std::vector<uint64_t> bitvector;
};

class MedianFilter {
public:
    explicit MedianFilter(const float *in, float *out, const ImageSubBlock& Block, const int iy, const int ix) :
    inputData(in),
    outputData(out),
    Coords(Block, iy, ix),
    Data(Block.GetBlockSize()),
    Bitvec(Coords.GetRangeX() * Coords.GetRangeY())
    {}

    inline void Run() {
        Data.MapPixelsToOrdinals(Coords, inputData);
        Data.InitializeOrdinals();

        for (int y = Coords.GetSTY(); y < Coords.GetENY(); y++) { //sty, eny --- this iterates over the coordinates of the a single block over y dimentsion
                //init bitvector with zeros
                Bitvec.ReInitWithZeros();

                //set values near initial position of running window --- coordinates of the running window inside the block
                int y_str = std::max(y - Coords.GetHY(), 0); 
                int x_str = std::max(Coords.GetSTX() - Coords.GetHX(), 0);
                int y_end = std::min(y + Coords.GetHY() + 1, Coords.GetRangeY());
                int x_end = std::min(Coords.GetSTX() + Coords.GetHX() + 1, Coords.GetRangeX());

                //set bit to 1 if it is inside running window
                for (int i = y_str; i < y_end; i++) {
                    for(int j = x_str; j < x_end; j++){
                        int ind = j + i * Coords.GetRangeX();
                        int position = Data.GetOrdinal(ind);
                        Bitvec.SetOne(position);
                    }
                }

                for (int x = Coords.GetSTX(); x < Coords.GetENX(); x++) { //stx, enx --- this iterates over the coordinates of the a single block over x dimentsion
                    //sliding window bounds
                    int sy = std::max(y - Coords.GetHY(), 0);
                    int sx = std::max(x - Coords.GetHX(), 0);
                    int ey = std::min(Coords.GetRangeY(), y + Coords.GetHY() + 1);
                    int ex = std::min(Coords.GetRangeX(), x + Coords.GetHX() + 1);

                    //size of running window
                    int window_size = (ey-sy)*(ex-sx);

                    int sxlr = std::max(x - Coords.GetHX() - 1, 0);
                    int exlr = std::max(x - Coords.GetHX(), 0);
                    int ii   = std::min(x + Coords.GetHX(), ex - 1);

                    //unset left most vertical bits inside running window
                    for (int i = sxlr; i < exlr; i++) {
                        for (int j = sy; j < ey; j++) {
                            int ind = sxlr + j * Coords.GetRangeX();
                            int position = Data.GetOrdinal(ind);
                            Bitvec.SetZero(position);
                        }
                    }

                    //set right most vertical bits inside running window
                    for(int j = sy; j < ey; j++) {
                        int ind = ii + j * Coords.GetRangeX();
                        int position = Data.GetOrdinal(ind);
                        Bitvec.SetOne(position);
                    }

                    //compute median of the sliding window from the bit vector and set to result
                    const int globalIndex = Coords.ComputeGlobalIndex(x, y); // index of the pixel on image / global index
                    ComputeMedian(globalIndex, window_size);
                }
            }
    }

private:
    inline void ComputeMedian(const int globalIndex, const int window_size) {
        if(window_size % 2 == 1)
            outputData[globalIndex] = ComputeMedianForOddSizedWindow(window_size);
        else 
            outputData[globalIndex] = ComputeMedianForEvenSizedWindow(window_size);
    }

    // merge these two
    inline float ComputeMedianForOddSizedWindow(const int window_size) {
        int position  = window_size / 2 + 1;
        auto indexes  = ComputeRemainder(0, position);

        const int N   = std::abs(indexes.second);
        const int ord = ComputePixelOrder(indexes.first, N);

        return Data.GetPixel(ord);
    }

    //these two
    inline float ComputeMedianForEvenSizedWindow(const int window_size) {
        int position1  = (window_size / 2);
        int position2  = (window_size / 2) + 1;

        auto indexes1  = ComputeRemainder(0, position1);
        auto indexes2  = ComputeRemainder(0, position2);

        const int N1   = std::abs(indexes1.second);
        const int N2   = std::abs(indexes2.second);

        const int ord1 = ComputePixelOrder(indexes1.first, N1);
        const int ord2 = ComputePixelOrder(indexes2.first, N2);

        return (Data.GetPixel(ord1) + Data.GetPixel(ord2)) / 2;
    }

    inline int ComputePixelOrder(const int index, const int order) const {
        return (index - 1) * DIV + (DIV - getNthBit(order, Bitvec.GetBits(index - 1))) - 1;
    }

    inline std::pair<int, int> ComputeRemainder(int d, int remainder) { //bitvec function? d not as a parameter
        while (remainder > 0) {
            int bb = countbits(Bitvec.GetBits(d));
            remainder -= bb;
            d++;
        }
        return {d, remainder};
    }

    const float *inputData;
    float *outputData;

    BlockCoordinates Coords;
    TransformedData Data;
    BitVector Bitvec;
};

/* 
(ny, nx) size of the image 
(hy, hx) size of sliding window
(in)     pointer to the data
(out)    pointer to median-fitlered data
*/
void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {

    // 1 Partition the image into blocks
    ImageSubBlock Block(nx, ny, hx, hy);

    // 2 Iterate over all blocks and 
    // compute median of each block in parallel
    #pragma omp parallel for schedule(dynamic, 1)
    for(int iy = 0; iy < Block.GetBSNY(); iy++) {
        for (int ix = 0; ix < Block.GetBSNX(); ix++) {
            MedianFilter Mf(in, out, Block, iy, ix);
            Mf.Run();
        }
    }
}
