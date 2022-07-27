#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <x86intrin.h>
#include <stdint.h>
#include <cassert>

/* 
TODO:
1) Names
2) ComputePixelOrder belonging?
*/

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

    inline int CalculateSizeOfBlockSide() {
        return scalingFactor * std::max(kernelSizeX, kernelSizeY);
    }

    inline int CalculateSizeOfStride(int kernelSize) {
        return blockside - kernelSize - 1;
    }

    inline int CalculateBlockSizeN(int N, int kernelSize, int strideSize) {
        return blockside < N ? (N - kernelSize + strideSize - 1) / strideSize : 1;
    }

private:

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

struct BlockCoordinates {
    explicit BlockCoordinates(const ImageSubBlock& Block, const int iy, const int ix) : 
    y0(ComputeOriginCoordinates(Block.GetStrideSizeY(), iy)),
    x0(ComputeOriginCoordinates(Block.GetStrideSizeX(), ix)),

    y1(ComputeUpperCoordinates(Block.GetImageY(), y0, Block.GetStrideSizeY(), Block.GetKernelSizeY())),
    x1(ComputeUpperCoordinates(Block.GetImageX(), x0, Block.GetStrideSizeX(), Block.GetKernelSizeX())),

    NX(Block.GetImageX()),
    HX(Block.GetSWlength()),
    HY(Block.GetSWheight()),

    sty(ComputeStartCoordY(iy, Block.GetSWheight())),
    stx(ComputeStartCoordX(ix, Block.GetSWlength())),
    eny(ComputeEndCoordY(iy, Block.GetBSNY(), Block.GetSWheight())),
    enx(ComputeEndCoordX(ix, Block.GetBSNX(), Block.GetSWlength()))
    {}

    inline int GetY0() const { return y0; }
    inline int GetY1() const { return y1; }
    inline int GetX0() const { return x0; }
    inline int GetX1() const { return x1; }
    inline int GetNX() const { return NX; }
    inline int GetHY() const { return HY; }
    inline int GetHX() const { return HX; }

    inline int GetRangeX() const { return x1 - x0; }
    inline int GetRangeY() const { return y1 - y0; }

    inline int ComputeGlobalIndex(const int x, const int y) const {
        return x + GetX0() + (y + GetY0()) * GetNX();
    }

    inline int GetSTY() const { return sty; }
    inline int GetSTX() const { return stx; }
    inline int GetENY() const { return eny; }
    inline int GetENX() const { return enx; }

    // bad names => has to be redone
    inline int GetStartY(const int y) const {
        return std::max(y - GetHY(), 0);    
    }

    inline int GetStartX() const {
        return std::max(GetSTX() - GetHX(), 0);
    }

    inline int GetEndY(const int y) const {
        return std::min(y + GetHY() + 1, GetRangeY());
    }

    inline int GetEndX() const {
        return std::min(GetSTX() + GetHX() + 1, GetRangeX());
    }

    // --------------------
    inline int GetSY(const int y) const {
        return std::max(y - GetHY(), 0);
    }
    
    inline int GetEY(const int y) const {
        return std::min(GetRangeY(), y + GetHY() + 1);
    }

    inline int GetSizeOfSlidingWindow(const int x, const int y) const {
        return (GetEY(y) - GetSY(y)) * (GetEX(x) - GetSX(x));
    }

    inline int GetStartCoordinateOfTheLeftmostVerticalBlock(const int x) const {
        return std::max(x - GetHX() - 1, 0);
                
    }

    inline int GetEndCoordinateOfTheLeftmostVerticalBlock(const int x) const {
        return std::max(x - GetHX(), 0);
    }

    inline int GetOffsetOfTheRightmostVerticalBlock(const int x) const {
        return std::min(x + GetHX(), GetEX(x) - 1);
    }

private:

    inline int GetSX(const int x) const {
        return std::max(x - GetHX(), 0);           
    }

    inline int GetEX(const int x) const {
        return std::min(GetRangeX(), x + GetHX() + 1);
    }

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

template <typename F>
class TransformedData {
public:
    explicit TransformedData(const int blocksize) :
    block_pixels(blocksize),
    ordinals(blocksize),
    localIndex(0)
    {}

    inline F GetPixel(const int index) const { return block_pixels[index].first; }
    inline int GetOrdinal(const int index) const { return ordinals[index]; }

    inline void SetOrdinals(const int value, const int index) {
        ordinals[index] = value;
    }

    inline void MapPixelsToOrdinals(const BlockCoordinates& coords, const F *in) {
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

        for (int i = 0; i < localIndex; i++) {
            ordinals[block_pixels[i].second] = i;
        }
    }

private:
    inline void IncreaseLocalIndex() { localIndex++; }
    
    std::vector<std::pair<F, int>> block_pixels;
    std::vector<int> ordinals;

    int localIndex;
};

class BitVector {
public:
    explicit BitVector(const int blockSize) :
    bitvector(ComputeSizeOfBitvector(blockSize))
    {}

    inline void ReInitWithZeros() {
        for(size_t i = 0; i < bitvector.size(); i++)
            bitvector[i] = ZERO;
    }

    inline void SetBit(const int position) {
        const int bitix  = position / DIV;
        const int shift  = ComputeShift(position);
        bitvector[bitix] |= (ONE << shift);
    }

    inline void UnsetBit(const int position) {
        const int bitix  = position / DIV;
        const int shift  = ComputeShift(position);
        bitvector[bitix] &= ~(ONE << shift);
    }

    inline int ComputeSizeOfBitvector(const int blockSize) const {
        return (blockSize + DIV - 1) / DIV;
    }

    inline uint64_t GetBits(const int position) const { 
        return bitvector[position];
    }

private:
    inline int ComputeShift(const int position) const {
        return DIV - position % DIV - 1;
    }

    std::vector<uint64_t> bitvector;
};

template <typename F>
class MedianFilter {
public:
    explicit MedianFilter(const F *in, F *out, const ImageSubBlock& Block, const int iy, const int ix) :
    inputData(in),
    outputData(out),
    Coords(Block, iy, ix),
    Data(Block.GetBlockSize()),
    Bitvec(Coords.GetRangeX() * Coords.GetRangeY())
    {}

    inline void ExecuteFiltering() {
        Data.MapPixelsToOrdinals(Coords, inputData);
        Data.InitializeOrdinals();
        ComputeMediansForCoordinateBlock();
    }

private:
    inline void ComputeMedianForSlidingWindow(const int globalIndex, const int windowSize) {
        if(windowSize % 2 == 1) {
            const int position      = ComputeMedianPosition(windowSize) + 1;
            outputData[globalIndex] = ComputeMedianForOddSizedWindow(position);
        } else {
            const int position1     = ComputeMedianPosition(windowSize);
            const int position2     = ComputeMedianPosition(windowSize) + 1;
            outputData[globalIndex] = ComputeMedianForEvenSizedWindow(position1, position2);
        }
    }

    inline int ComputeMedianPosition(const int windowSize) const {
        return windowSize * 0.5;
    }

    inline F ComputeMedianForOddSizedWindow(const int position) const {
        const auto indexes  = ComputeRemainder(0, position);
        const int N         = std::abs(indexes.second);
        const int ord       = ComputePixelOrder(indexes.first, N);

        return Data.GetPixel(ord);
    }

    inline F ComputeMedianForEvenSizedWindow(const int position1, const int position2) const { 
        return (ComputeMedianForOddSizedWindow(position1) + ComputeMedianForOddSizedWindow(position2)) * 0.5;
    }

    inline int ComputePixelOrder(const int index, const int order) const {
        return (index - 1) * DIV + (DIV - getNthBit(order, Bitvec.GetBits(index - 1))) - 1;
    }

    inline std::pair<int, int> ComputeRemainder(int d, int remainder) const {
        while (remainder > 0) {
            int bb = countbits(Bitvec.GetBits(d));
            remainder -= bb;
            d++;
        }
        return {d, remainder};
    }

    inline int GetLinearizedIndex(const int y, const int x) const {
        return x + y * Coords.GetRangeX();
    }

    inline void SetBitsInsideWindow(const int y) {
        for (int i = Coords.GetStartY(y); i < Coords.GetEndY(y); i++) {
            for(int j = Coords.GetStartX(); j < Coords.GetEndX(); j++){
                const int index    = GetLinearizedIndex(i, j);
                const int position = Data.GetOrdinal(index);
                Bitvec.SetBit(position);
            }
        }
    }

    inline void UnsetLeftmostVerticalBits(const int x, const int y) {
        const int leftStart = Coords.GetStartCoordinateOfTheLeftmostVerticalBlock(x);
        const int leftEnd   = Coords.GetEndCoordinateOfTheLeftmostVerticalBlock(x);

        for (int i = leftStart; i < leftEnd; i++) {
            for (int j = Coords.GetSY(y); j < Coords.GetEY(y); j++) {
                const int index    = GetLinearizedIndex(j, leftStart);
                const int position = Data.GetOrdinal(index);
                Bitvec.UnsetBit(position);
            }
        }
    }

    inline void SetRightmostVerticalBits(const int x, const int y) {
        const int offset = Coords.GetOffsetOfTheRightmostVerticalBlock(x);

        for(int j = Coords.GetSY(y); j < Coords.GetEY(y); j++) {
            const int index    = GetLinearizedIndex(j, offset);
            const int position = Data.GetOrdinal(index);
            Bitvec.SetBit(position);
        }
    }

    inline void ComputeMediansForCoordinateBlock() {
        for (int y = Coords.GetSTY(); y < Coords.GetENY(); y++) { // iterates over the coordinates of the a single block over y dimentsion
            Bitvec.ReInitWithZeros();
            SetBitsInsideWindow(y);

            for (int x = Coords.GetSTX(); x < Coords.GetENX(); x++) { // this iterates over the coordinates of the a single block over x dimentsio
                UnsetLeftmostVerticalBits(x, y);
                SetRightmostVerticalBits(x, y);

                const int globalIndex = Coords.ComputeGlobalIndex(x, y);
                const int windowSize  = Coords.GetSizeOfSlidingWindow(x, y);
                ComputeMedianForSlidingWindow(globalIndex, windowSize);
            }
        }
    }

    const F *inputData;
    F *outputData;

    BlockCoordinates Coords;
    TransformedData<F> Data;
    BitVector Bitvec;
};

/* 
(ny, nx) size of the image 
(hy, hx) size of sliding window
(in)     pointer to the data
(out)    pointer to median-fitlered data
*/
template <typename F>
void mf(int ny, int nx, int hy, int hx, const F *in, F *out) {
    // 1 Partition the image into blocks
    ImageSubBlock Block(nx, ny, hx, hy);

    // 2 Iterate over all blocks and 
    // compute median of each block in parallel
    #pragma omp parallel for schedule(dynamic, 1)
    for(int iy = 0; iy < Block.GetBSNY(); iy++) {
        for (int ix = 0; ix < Block.GetBSNX(); ix++) {
            MedianFilter<F> Mf(in, out, Block, iy, ix);
            Mf.ExecuteFiltering();
        }
    }
}

template void mf<float>(int ny, int nx, int hy, int hx, const float *in, float *out);
template void mf<double>(int ny, int nx, int hy, int hx, const double *in, double *out);
