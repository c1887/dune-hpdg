#include <config.h>
#include <iostream>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/hpdg/common/commandlineargs.hh>

using namespace Dune;

TestSuite test_commandline_pt() {
  TestSuite suite("Test command line parser");


  // We make up some artifical command line data.
  // The following is actually not legal in C++ but should be valid C.
  // Compiler will warn but grumpily accept it. Don't try this at home, kids.
  // This is really just for testing.
  char* argv[] = {"./foo", "--bool0", "--double=4.2", "--int", "3", "--bool"};
  constexpr int argc = 6;

  auto pt = HPDG::CommandLine::parameterTreeFromCommandLine(argc, argv);

  suite.check(pt.hasKey("bool0")) << "Did not find key \"bool0\"";
  suite.check(pt.hasKey("double")) << "Did not find key \"double\"";
  suite.check(pt.hasKey("int")) << "Did not find key \"int\"";
  suite.check(pt.hasKey("bool")) << "Did not find key \"bool\"";

  suite.check(pt.get("bool0", false)== true);
  suite.check(pt.get("double", 1.0) == 4.2) << "Expected 4.2, got " << pt.get("double", 1.0);
  suite.check(pt.get("int", 1)== 3);
  suite.check(pt.get("bool", false)== true);

  //check if we can change values in a existing tree
  char* new_argv[] = {"./foo", "--int", "4"};
  HPDG::CommandLine::insertKeysFromCommandLine(pt, 3, new_argv);
  suite.check(pt.get("int", 1)== 4); // was previously 3

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  // If there is a "--help" string, we print some help string and exit.
  HPDG::CommandLine::help(argc, argv, "Help called");

  auto pt = HPDG::CommandLine::parameterTreeFromCommandLine(argc, argv);
  for (auto key: pt.getValueKeys()) {
    std::cout << "pt["<<key<<"]=" << pt[key] << std::endl;
  }


  TestSuite suite;
  suite.subTest(test_commandline_pt());
  return suite.exit();
}
