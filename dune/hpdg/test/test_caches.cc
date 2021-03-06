#include <config.h>
#include <cmath>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/fvector.hh>

#include <dune/hpdg/common/mappedcache.hh>
#include <dune/hpdg/common/indexedcache.hh>
using namespace Dune;

using Range = int;
using Domain = int;

// global variables, ouch.
int function_called = 0;
int class_called = 0;

Range foo(Domain x) {
  function_called++;
  return 2*x;
}

struct F {
  int operator()(int i) {
    class_called++; // increase call-counter
    return 2*i;
  }

};

template<class Cache>
TestSuite test_cache() {
  TestSuite suite;

  auto cache_from_function = Cache(foo);
  suite.check(cache_from_function.value(2) == 4);
  suite.check(function_called==1);
  // get value another time, function_called should not change because the value is already there
  suite.check(cache_from_function.value(2) == 4 && function_called == 1);
  // we can alternatively use the [] operator
  suite.check(cache_from_function[2] == 4 && function_called == 1);

  // we can still use the cache in a const context, even though it's technically modified:
  const auto& const_cache = cache_from_function;
  // call with a new value
  suite.check(const_cache[1] == 2);
  suite.check(function_called == 2); // for the new value, "foo" was called again.

  // check contains:
  suite.check(cache_from_function.contains(1));
  // clear cache:
  cache_from_function.clear();
  // value should no longer be there:
  suite.check(not cache_from_function.contains(1));

  // create cache from a lambda:
  Domain factor = 2;
  auto cache_from_lambda = Cache([&](int i) {
      return factor*i;
      });

  suite.check(cache_from_lambda.value(1) == factor);

  // create from a class with () operator:
  // Note that this will be copied/moved from, so if your f is a big object,
  // consider using a lambda instead.
  F f;
  auto cache_from_class = Cache(std::move(f));
  suite.check(cache_from_class.value(1) == 2);
  suite.check(cache_from_class.value(1) == 2);
  suite.check(class_called == 1, "Check call counter in f");

  // test operator=
  auto assigned = Cache();
  assigned = cache_from_class;
  suite.check(assigned.value(1) == 2);
  suite.check(class_called == 1, "Check call counter in f");

  // Check with user supplied lambda and default constructed cache:
  Cache def_constr;
  suite.check(def_constr.value(1, [](int i){return 3*i;}) == 3);

  // check moved
  auto moved = std::move(def_constr);
  suite.check(moved.value(1) == 3); // should be still in cache, hence we don't need a generator. Other indices woudl throw here

  // check copied
  auto copied = moved;
  suite.check(copied.value(1) == 3); // should be still in cache, hence we don't need a generator. Other indices woudl throw here

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  TestSuite suite;
  std::cout << "Testing MappedCache" << std::endl;
  suite.subTest(test_cache<HPDG::MappedCache<Domain, Range>>());

  // reset counters:
  function_called = 0;
  class_called = 0;

  std::cout << "Testing IndexedCache" << std::endl;
  suite.subTest(test_cache<HPDG::IndexedCache<Domain>>());
  return suite.exit();
}
