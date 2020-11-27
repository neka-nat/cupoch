/*********************************************************************
* Software License Agreement (BSD License)
* 
*  Copyright (c) 2008, Willow Garage, Inc.
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
*   * Neither the name of the Willow Garage nor the names of its
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

/* Author: Wim Meeussen */

#ifndef URDF_PARSER_URDF_PARSER_H
#define URDF_PARSER_URDF_PARSER_H

#include <stdexcept>
#include <string>
#include <vector>

#include <tinyxml.h>
#include <urdf_model/model.h>
#include <urdf_model/color.h>
#include <urdf_world/types.h>
#include <urdf_model/utils.h>

#include "exportdecl.h"

namespace urdf_export_helpers {

URDFDOM_DLLAPI std::string values2str(unsigned int count, const double *values, double (*conv)(double) = NULL);
URDFDOM_DLLAPI std::string values2str(urdf::Vector3 vec);
URDFDOM_DLLAPI std::string values2str(urdf::Rotation rot);
URDFDOM_DLLAPI std::string values2str(urdf::Color c);
URDFDOM_DLLAPI std::string values2str(double d);

// This lives here (rather than in model.cpp) so we can run tests on it.
class URDFVersion final
{
public:
  explicit URDFVersion(const char *attr)
  {
    // If the passed in attribute is NULL, it means it wasn't specified in the
    // XML, so we just assume version 1.0.
    if (attr == nullptr)
    {
      major_ = 1;
      minor_ = 0;
      return;
    }

    // We only accept version strings of the type <major>.<minor>
    std::vector<std::string> split;
    urdf::split_string(split, std::string(attr), ".");
    if (split.size() == 2)
    {
      major_ = strToUnsigned(split[0].c_str());
      minor_ = strToUnsigned(split[1].c_str());
    }
    else
    {
      throw std::runtime_error("The version attribute should be in the form 'x.y'");
    }
  }

  bool equal(uint32_t maj, uint32_t min)
  {
    return this->major_ == maj && this->minor_ == min;
  }

  uint32_t getMajor() const
  {
    return major_;
  }

  uint32_t getMinor() const
  {
    return minor_;
  }

private:
  uint32_t strToUnsigned(const char *str)
  {
    if (str[0] == '\0')
    {
      // This would get caught below, but we can make a nicer error message
      throw std::runtime_error("One of the fields of the version attribute is blank");
    }
    char *end = const_cast<char *>(str);
    long value = strtol(str, &end, 10);
    if (end == str)
    {
      // If the pointer didn't move at all, then we couldn't convert any of
      // the string to an integer.
      throw std::runtime_error("Version attribute is not an integer");
    }
    if (*end != '\0')
    {
      // Here, we didn't go all the way to the end of the string, which
      // means there was junk at the end
      throw std::runtime_error("Extra characters after the version number");
    }
    if (value < 0)
    {
      throw std::runtime_error("Version number must be positive");
    }

    return value;
  }

  uint32_t major_;
  uint32_t minor_;
};

}

namespace urdf{

  URDFDOM_DLLAPI ModelInterfaceSharedPtr parseURDF(const std::string &xml_string);
  URDFDOM_DLLAPI ModelInterfaceSharedPtr parseURDFFile(const std::string &path);
  URDFDOM_DLLAPI TiXmlDocument*  exportURDF(ModelInterfaceSharedPtr &model);
  URDFDOM_DLLAPI TiXmlDocument*  exportURDF(const ModelInterface &model);
  URDFDOM_DLLAPI bool parsePose(Pose&, TiXmlElement*);
}

#endif
