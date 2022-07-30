#pragma once

#include "util.h"
#include "median-filter.h"
#include "profiling.h"

#include <vector>

void Benchmark(const int ny,
	const int nx,
	const int hy,
	const int hx,
	const int lower,
	const int upper,
	const std::string& message) {
	std::vector<float> input(nx*ny);
	std::vector<float> output(nx*ny);

	RandomGenerator rng;
	rng.GenerateUniformFloat(input, lower, upper);

	LOG_DURATION(message)
	{
		mf<float>(ny, nx, hy, hx, input.data(), output.data());
	}
}