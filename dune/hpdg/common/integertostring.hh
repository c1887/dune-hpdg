// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2
#pragma once
#include <string>
#include <sstream>
#include <iomanip>


namespace Dune {
namespace HPDG {

/** Converts an integer to a string containing that integer plus some
 * leading zeros if the integer has less than specified digits.
 *
 * Examples:
 * n_digit_string(42, 4) -> "0042"
 * n_digit_string(42, 5) -> "00042"
 * n_digit_string(1234, 4) -> "1234"
 * n_digit_string(12345, 4) -> "12345" // more digits used than required!
 */
std::string n_digit_string(const unsigned int input, unsigned int length) {
  std::ostringstream out;
  out << std::setfill('0') << std::setw(length) << input;
  return out.str();
}

}}
