#include "test-runner.h"
#include "median-filter.h"
#include "integration-tests.h" 

// for tests 
// g++ -O3 -Werror -Wall --pedantic -std=c++17 -march=native -fopenmp -o mf-main mf-main.cpp

void IntegrationTests() {
	TestRunner tr;
	tr.RunTest(TestResult1,  "Test-small-1");
	tr.RunTest(TestResult2,  "Test-small-2");
	tr.RunTest(TestResult3,  "Test-small-3");
	tr.RunTest(TestResult4,  "Test-small-4");
	tr.RunTest(TestResult5,  "Test-small-5");
	tr.RunTest(TestResult6,  "Test-small-6");
	tr.RunTest(TestResult7,  "Test-small-7");
	tr.RunTest(TestResult8,  "Test-small-8");
	tr.RunTest(TestResult9,  "Test-small-9");
	tr.RunTest(TestResult10, "Test-small-10");
	tr.RunTest(TestResult11, "Test-small-11");
	tr.RunTest(TestResult12, "Test-small-12");
	tr.RunTest(TestResult13, "Test-small-13");
	tr.RunTest(TestResult14, "Test-small-14");
	tr.RunTest(TestResult15, "Test-small-15");
}

void Benchmarks() {

}

int main() {
	IntegrationTests();
	Benchmarks();
	return 0;
}