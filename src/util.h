#pragma once

#include <random>

class RandomGenerator {
public:
	RandomGenerator()
	{
		std::random_device rd;
		gen = std::mt19937(rd());
	}

	void GenerateUniformFloat(vector<float>& container, const int lower, const int upper) {
		std::uniform_real_distribution<> dist(lower, upper);

		for (size_t i = 0; i < container.size(); i++)
			container[i] = dist(gen);
	}

private:
	std::mt19937 gen;
};
