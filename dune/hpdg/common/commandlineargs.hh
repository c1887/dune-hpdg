#pragma once
#include <cstring>
#include <string>

#include <dune/common/parametertree.hh>
#include <dune/common/rangeutilities.hh>

namespace Dune {
namespace HPDG {
namespace CommandLine {

  /** If the option "--help" was specified", print help text and exit */
  void help(int argc, char** argv, std::string helpText =
      "Syntax: ./foo --option=value") {

    auto isOption = [](const char* value) {
      return std::strncmp(value, "--",2) == 0;
    };

    // scan for help option:
    for (auto i : range(1,argc)) {
      if(isOption(argv[i]) && std::strncmp(argv[i], "--help", 6)==0) {
        std::cout << helpText <<std::endl;
        std::exit(0);
      }
    }
  }

  /** Convert command line args into a parameter tree:
    Syntax is either
    ./foo --option=value
    or
    ./foo --option value

    Pure "--option" (without any value) is equivalent to
    --option=true
  */
  auto insertKeysFromCommandLine(ParameterTree& pt, int argc, char** argv) {
    // quick check if string starts with "--".
    auto isOption = [](const char* value) {
      return std::strncmp(value, "--",2) == 0;
    };

    for (auto i : range(1,argc)) {
      if (isOption(argv[i])) { // string starts with --
        auto key = std::string(argv[i]);
        key.erase(0,2); // remove "--" part
        auto equals = key.find_first_of("=");


        // if "=" was found, use the rest of the string as value
        if (equals != std::string::npos) {
          pt[key.substr(0, equals)] = key.substr(equals+1, key.size());
        }
        else

          // if option was supplied without any value, assume true
          if(i==argc-1 or isOption(argv[i+1])) {
            pt[key] = "true";
          }
          else
            pt[key] = argv[i+1];
      }
    }
  }

  /** Creates a parameter tree from command line args */
  auto parameterTreeFromCommandLine(int argc, char** argv) {
    auto pt = Dune::ParameterTree();

    insertKeysFromCommandLine(pt, argc, argv);
    return pt;
  }
}
}
}
