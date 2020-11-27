/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2016, Open Source Robotics Foundation (OSRF)
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the OSRF nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Author: Steve Peters */

#ifndef URDF_INTERFACE_UTILS_H
#define URDF_INTERFACE_UTILS_H

#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace urdf {

// Replacement for boost::split( ... , ... , boost::is_any_of(" "))
inline
void split_string(std::vector<std::string> &result,
                  const std::string &input,
                  const std::string &isAnyOf)
{
  std::string::size_type start = 0;
  std::string::size_type end = input.find_first_of(isAnyOf, start);
  while (end != std::string::npos)
  {
    result.push_back(input.substr(start, end-start));
    start = end + 1;
    end = input.find_first_of(isAnyOf, start);
  }
  if (start < input.length())
  {
    result.push_back(input.substr(start));
  }
}

// This is a locale-safe version of string-to-double, which is suprisingly
// difficult to do correctly.  This function ensures that the C locale is used
// for parsing, as that matches up with what the XSD for double specifies.
// On success, the double is returned; on failure, a std::runtime_error is
// thrown.
static inline double strToDouble(const char *in)
{
  std::stringstream ss;
  ss.imbue(std::locale::classic());

  ss << in;

  double out;
  ss >> out;

  if (ss.fail() || !ss.eof()) {
    throw std::runtime_error("Failed converting string to double");
  }

  return out;
}

}

#endif
