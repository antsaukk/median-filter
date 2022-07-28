#pragma once

#include <sstream>
#include <stdexcept>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <cmath>

using namespace std;

template <class T>
ostream& operator << (ostream& os, const vector<T>& s) {
  os << "{";
  bool first = true;
  for (const auto& x : s) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << x;
  }
  return os << "}";
}

template <class T>
ostream& operator << (ostream& os, const set<T>& s) {
  os << "{";
  bool first = true;
  for (const auto& x : s) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << x;
  }
  return os << "}";
}

template <class K, class V>
ostream& operator << (ostream& os, const map<K, V>& m) {
  os << "{";
  bool first = true;
  for (const auto& kv : m) {
    if (!first) {
      os << ", ";
    }
    first = false;
    os << kv.first << ": " << kv.second;
  }
  return os << "}";
}

template<class T, class U>
void AssertEqual(const T& t, const U& u, const string& hint = {}) {
  if (!(t == u)) {
    ostringstream os;
    os << "Assertion failed: " << t << " != " << u;
    if (!hint.empty()) {
       os << " hint: " << hint;
    }
    throw runtime_error(os.str());
  }
}

inline void Assert(bool b, const string& hint) {
  AssertEqual(b, true, hint);
}

// small quick add-hoc for testing floating points.
// improve using ideas from here:
//
// 1) https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
// 2) https://stackoverflow.com/questions/17333/what-is-the-most-effective-way-for-float-and-double-comparison?page=1&tab=scoredesc#tab-top
// another thing that can be improved is to identify the digits where numbers are different
template <> 
void AssertEqual<vector<float>, vector<float>>(const vector<float>& t,
                                              const vector<float>& u,
                                              const string& hint) {
  bool equal      = true; 
  float tolerance = 1e-6f;
  for (size_t i = 0; i < t.size(); i++) {
    equal &= std::abs(t[i] - u[i]) < tolerance;
  }

  Assert(equal, hint);
}

template <> 
void AssertEqual<vector<double>, vector<double>>(const vector<double>& t,
                                                const vector<double>& u,
                                                const string& hint) {
  bool equal       = true; 
  double tolerance = 1e-9;
  for (size_t i = 0; i < t.size(); i++) {
    equal &= std::abs(t[i] - u[i]) < tolerance;
  }

  Assert(equal, hint);
}

class TestRunner {
public:
  template <class TestFunc>
  void RunTest(TestFunc func, const string& test_name) {
    try {
      func();
      cerr << test_name << " passed" << endl;
    } catch (exception& e) {
      ++fail_count;
      cerr << test_name << " fail: " << e.what() << endl;
    } catch (...) {
      ++fail_count;
      cerr << "Unknown exception caught" << endl;
    }
  }

  ~TestRunner() {
    if (fail_count > 0) {
      cerr << fail_count << " unit tests failed. Terminate" << endl;
      exit(1);
    }
  }

private:
  int fail_count = 0;
};

#define ASSERT_EQUAL(x, y) {                          \
  ostringstream __assert_equal_private_os;            \
  __assert_equal_private_os                           \
    << #x << " != " << #y << ", "                     \
    << __FILE__ << ":" << __LINE__;                   \
  AssertEqual(x, y, __assert_equal_private_os.str()); \
}

#define ASSERT(x) {                     \
  ostringstream os;                     \
  os << #x << " is false, "             \
    << __FILE__ << ":" << __LINE__;     \
  Assert(x, os.str());                  \
}

#define RUN_TEST(tr, func) \
  tr.RunTest(func, #func)
