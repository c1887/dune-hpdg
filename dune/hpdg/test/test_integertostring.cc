#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/hpdg/common/integertostring.hh>

#include <cstring>

using namespace Dune;

TestSuite test_integerToString() {
  TestSuite suite("test_integerToString");

  /*
   * We just test the things
   * decribed in the docs of the function:
   *
   * Examples:
   * n_digit_string(42, 4) -> "0042"
   * n_digit_string(42, 5) -> "00042"
   * n_digit_string(1234, 4) -> "1234"
   * n_digit_string(12345, 4) -> "12345"
   */

  suite.check(std::strcmp(HPDG::n_digit_string(42, 4).c_str(), "0042")==0);
  suite.check(std::strcmp(HPDG::n_digit_string(42, 5).c_str(), "00042")==0);
  suite.check(std::strcmp(HPDG::n_digit_string(1234, 4).c_str(), "1234")==0);
  suite.check(std::strcmp(HPDG::n_digit_string(12345, 4).c_str(), "12345")==0); // note that the string is longer than required

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  TestSuite suite;
  suite.subTest(test_integerToString());
  return suite.exit();
}
