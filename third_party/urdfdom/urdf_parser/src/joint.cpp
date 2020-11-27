/*********************************************************************
* Software Ligcense Agreement (BSD License)
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

/* Author: John Hsu */

#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <urdf_model/joint.h>
#include <console_bridge/console.h>
#include <tinyxml.h>
#include <urdf_parser/urdf_parser.h>

namespace urdf{

bool parsePose(Pose &pose, TiXmlElement* xml);

bool parseJointDynamics(JointDynamics &jd, TiXmlElement* config)
{
  jd.clear();

  // Get joint damping
  const char* damping_str = config->Attribute("damping");
  if (damping_str == NULL){
    CONSOLE_BRIDGE_logDebug("urdfdom.joint_dynamics: no damping, defaults to 0");
    jd.damping = 0;
  }
  else
  {
    try {
      jd.damping = strToDouble(damping_str);
    } catch(std::runtime_error &) {
      CONSOLE_BRIDGE_logError("damping value (%s) is not a valid float", damping_str);
      return false;
    }
  }

  // Get joint friction
  const char* friction_str = config->Attribute("friction");
  if (friction_str == NULL){
    CONSOLE_BRIDGE_logDebug("urdfdom.joint_dynamics: no friction, defaults to 0");
    jd.friction = 0;
  }
  else
  {
    try {
      jd.friction = strToDouble(friction_str);
    } catch (std::runtime_error &) {
      CONSOLE_BRIDGE_logError("friction value (%s) is not a valid float", friction_str);
      return false;
    }
  }

  if (damping_str == NULL && friction_str == NULL)
  {
    CONSOLE_BRIDGE_logError("joint dynamics element specified with no damping and no friction");
    return false;
  }
  else{
    CONSOLE_BRIDGE_logDebug("urdfdom.joint_dynamics: damping %f and friction %f", jd.damping, jd.friction);
    return true;
  }
}

bool parseJointLimits(JointLimits &jl, TiXmlElement* config)
{
  jl.clear();

  // Get lower joint limit
  const char* lower_str = config->Attribute("lower");
  if (lower_str == NULL){
    CONSOLE_BRIDGE_logDebug("urdfdom.joint_limit: no lower, defaults to 0");
    jl.lower = 0;
  }
  else
  {
    try {
      jl.lower = strToDouble(lower_str);
    } catch (std::runtime_error &) {
      CONSOLE_BRIDGE_logError("lower value (%s) is not a valid float", lower_str);
      return false;
    }
  }

  // Get upper joint limit
  const char* upper_str = config->Attribute("upper");
  if (upper_str == NULL){
    CONSOLE_BRIDGE_logDebug("urdfdom.joint_limit: no upper, , defaults to 0");
    jl.upper = 0;
  }
  else
  {
    try {
      jl.upper = strToDouble(upper_str);
    } catch(std::runtime_error &) {
      CONSOLE_BRIDGE_logError("upper value (%s) is not a valid float", upper_str);
      return false;
    }
  }

  // Get joint effort limit
  const char* effort_str = config->Attribute("effort");
  if (effort_str == NULL){
    CONSOLE_BRIDGE_logError("joint limit: no effort");
    return false;
  }
  else
  {
    try {
      jl.effort = strToDouble(effort_str);
    } catch(std::runtime_error &) {
      CONSOLE_BRIDGE_logError("effort value (%s) is not a valid float", effort_str);
      return false;
    }
  }

  // Get joint velocity limit
  const char* velocity_str = config->Attribute("velocity");
  if (velocity_str == NULL){
    CONSOLE_BRIDGE_logError("joint limit: no velocity");
    return false;
  }
  else
  {
    try {
      jl.velocity = strToDouble(velocity_str);
    } catch(std::runtime_error &) {
      CONSOLE_BRIDGE_logError("velocity value (%s) is not a valid float", velocity_str);
      return false;
    }
  }

  return true;
}

bool parseJointSafety(JointSafety &js, TiXmlElement* config)
{
  js.clear();

  // Get soft_lower_limit joint limit
  const char* soft_lower_limit_str = config->Attribute("soft_lower_limit");
  if (soft_lower_limit_str == NULL)
  {
    CONSOLE_BRIDGE_logDebug("urdfdom.joint_safety: no soft_lower_limit, using default value");
    js.soft_lower_limit = 0;
  }
  else
  {
    try {
      js.soft_lower_limit = strToDouble(soft_lower_limit_str);
    } catch(std::runtime_error &) {
      CONSOLE_BRIDGE_logError("soft_lower_limit value (%s) is not a valid float", soft_lower_limit_str);
      return false;
    }
  }

  // Get soft_upper_limit joint limit
  const char* soft_upper_limit_str = config->Attribute("soft_upper_limit");
  if (soft_upper_limit_str == NULL)
  {
    CONSOLE_BRIDGE_logDebug("urdfdom.joint_safety: no soft_upper_limit, using default value");
    js.soft_upper_limit = 0;
  }
  else
  {
    try {
      js.soft_upper_limit = strToDouble(soft_upper_limit_str);
    } catch(std::runtime_error &) {
      CONSOLE_BRIDGE_logError("soft_upper_limit value (%s) is not a valid float", soft_upper_limit_str);
      return false;
    }
  }

  // Get k_position_ safety "position" gain - not exactly position gain
  const char* k_position_str = config->Attribute("k_position");
  if (k_position_str == NULL)
  {
    CONSOLE_BRIDGE_logDebug("urdfdom.joint_safety: no k_position, using default value");
    js.k_position = 0;
  }
  else
  {
    try {
      js.k_position = strToDouble(k_position_str);
    } catch(std::runtime_error &) {
      CONSOLE_BRIDGE_logError("k_position value (%s) is not a valid float", k_position_str);
      return false;
    }
  }
  // Get k_velocity_ safety velocity gain
  const char* k_velocity_str = config->Attribute("k_velocity");
  if (k_velocity_str == NULL)
  {
    CONSOLE_BRIDGE_logError("joint safety: no k_velocity");
    return false;
  }
  else
  {
    try {
      js.k_velocity = strToDouble(k_velocity_str);
    } catch(std::runtime_error &) {
      CONSOLE_BRIDGE_logError("k_velocity value (%s) is not a valid float", k_velocity_str);
      return false;
    }
  }

  return true;
}

bool parseJointCalibration(JointCalibration &jc, TiXmlElement* config)
{
  jc.clear();

  // Get rising edge position
  const char* rising_position_str = config->Attribute("rising");
  if (rising_position_str == NULL)
  {
    CONSOLE_BRIDGE_logDebug("urdfdom.joint_calibration: no rising, using default value");
    jc.rising.reset();
  }
  else
  {
    try {
      jc.rising.reset(new double(strToDouble(rising_position_str)));
    } catch(std::runtime_error &) {
      CONSOLE_BRIDGE_logError("rising value (%s) is not a valid float", rising_position_str);
      return false;
    }
  }

  // Get falling edge position
  const char* falling_position_str = config->Attribute("falling");
  if (falling_position_str == NULL)
  {
    CONSOLE_BRIDGE_logDebug("urdfdom.joint_calibration: no falling, using default value");
    jc.falling.reset();
  }
  else
  {
    try {
      jc.falling.reset(new double(strToDouble(falling_position_str)));
    } catch(std::runtime_error &) {
      CONSOLE_BRIDGE_logError("falling value (%s) is not a valid float", falling_position_str);
      return false;
    }
  }

  return true;
}

bool parseJointMimic(JointMimic &jm, TiXmlElement* config)
{
  jm.clear();

  // Get name of joint to mimic
  const char* joint_name_str = config->Attribute("joint");

  if (joint_name_str == NULL)
  {
    CONSOLE_BRIDGE_logError("joint mimic: no mimic joint specified");
    return false;
  }
  else
    jm.joint_name = joint_name_str;
  
  // Get mimic multiplier
  const char* multiplier_str = config->Attribute("multiplier");

  if (multiplier_str == NULL)
  {
    CONSOLE_BRIDGE_logDebug("urdfdom.joint_mimic: no multiplier, using default value of 1");
    jm.multiplier = 1;    
  }
  else
  {
    try {
      jm.multiplier = strToDouble(multiplier_str);
    } catch(std::runtime_error &) {
      CONSOLE_BRIDGE_logError("multiplier value (%s) is not a valid float", multiplier_str);
      return false;
    }
  }

  
  // Get mimic offset
  const char* offset_str = config->Attribute("offset");
  if (offset_str == NULL)
  {
    CONSOLE_BRIDGE_logDebug("urdfdom.joint_mimic: no offset, using default value of 0");
    jm.offset = 0;
  }
  else
  {
    try {
      jm.offset = strToDouble(offset_str);
    } catch(std::runtime_error &) {
      CONSOLE_BRIDGE_logError("offset value (%s) is not a valid float", offset_str);
      return false;
    }
  }

  return true;
}

bool parseJoint(Joint &joint, TiXmlElement* config)
{
  joint.clear();

  // Get Joint Name
  const char *name = config->Attribute("name");
  if (!name)
  {
    CONSOLE_BRIDGE_logError("unnamed joint found");
    return false;
  }
  joint.name = name;

  // Get transform from Parent Link to Joint Frame
  TiXmlElement *origin_xml = config->FirstChildElement("origin");
  if (!origin_xml)
  {
    CONSOLE_BRIDGE_logDebug("urdfdom: Joint [%s] missing origin tag under parent describing transform from Parent Link to Joint Frame, (using Identity transform).", joint.name.c_str());
    joint.parent_to_joint_origin_transform.clear();
  }
  else
  {
    if (!parsePose(joint.parent_to_joint_origin_transform, origin_xml))
    {
      joint.parent_to_joint_origin_transform.clear();
      CONSOLE_BRIDGE_logError("Malformed parent origin element for joint [%s]", joint.name.c_str());
      return false;
    }
  }

  // Get Parent Link
  TiXmlElement *parent_xml = config->FirstChildElement("parent");
  if (parent_xml)
  {
    const char *pname = parent_xml->Attribute("link");
    if (!pname)
    {
      CONSOLE_BRIDGE_logInform("no parent link name specified for Joint link [%s]. this might be the root?", joint.name.c_str());
    }
    else
    {
      joint.parent_link_name = std::string(pname);
    }
  }

  // Get Child Link
  TiXmlElement *child_xml = config->FirstChildElement("child");
  if (child_xml)
  {
    const char *pname = child_xml->Attribute("link");
    if (!pname)
    {
      CONSOLE_BRIDGE_logInform("no child link name specified for Joint link [%s].", joint.name.c_str());
    }
    else
    {
      joint.child_link_name = std::string(pname);
    }
  }

  // Get Joint type
  const char* type_char = config->Attribute("type");
  if (!type_char)
  {
    CONSOLE_BRIDGE_logError("joint [%s] has no type, check to see if it's a reference.", joint.name.c_str());
    return false;
  }
  
  std::string type_str = type_char;
  if (type_str == "planar")
    joint.type = Joint::PLANAR;
  else if (type_str == "floating")
    joint.type = Joint::FLOATING;
  else if (type_str == "revolute")
    joint.type = Joint::REVOLUTE;
  else if (type_str == "continuous")
    joint.type = Joint::CONTINUOUS;
  else if (type_str == "prismatic")
    joint.type = Joint::PRISMATIC;
  else if (type_str == "fixed")
    joint.type = Joint::FIXED;
  else
  {
    CONSOLE_BRIDGE_logError("Joint [%s] has no known type [%s]", joint.name.c_str(), type_str.c_str());
    return false;
  }

  // Get Joint Axis
  if (joint.type != Joint::FLOATING && joint.type != Joint::FIXED)
  {
    // axis
    TiXmlElement *axis_xml = config->FirstChildElement("axis");
    if (!axis_xml){
      CONSOLE_BRIDGE_logDebug("urdfdom: no axis elemement for Joint link [%s], defaulting to (1,0,0) axis", joint.name.c_str());
      joint.axis = Vector3(1.0, 0.0, 0.0);
    }
    else{
      if (axis_xml->Attribute("xyz")){
        try {
          joint.axis.init(axis_xml->Attribute("xyz"));
        }
        catch (ParseError &e) {
          joint.axis.clear();
          CONSOLE_BRIDGE_logError("Malformed axis element for joint [%s]: %s", joint.name.c_str(), e.what());
          return false;
        }
      }
    }
  }

  // Get limit
  TiXmlElement *limit_xml = config->FirstChildElement("limit");
  if (limit_xml)
  {
    joint.limits.reset(new JointLimits());
    if (!parseJointLimits(*joint.limits, limit_xml))
    {
      CONSOLE_BRIDGE_logError("Could not parse limit element for joint [%s]", joint.name.c_str());
      joint.limits.reset();
      return false;
    }
  }
  else if (joint.type == Joint::REVOLUTE)
  {
    CONSOLE_BRIDGE_logError("Joint [%s] is of type REVOLUTE but it does not specify limits", joint.name.c_str());
    return false;
  }
  else if (joint.type == Joint::PRISMATIC)
  {
    CONSOLE_BRIDGE_logError("Joint [%s] is of type PRISMATIC without limits", joint.name.c_str()); 
    return false;
  }

  // Get safety
  TiXmlElement *safety_xml = config->FirstChildElement("safety_controller");
  if (safety_xml)
  {
    joint.safety.reset(new JointSafety());
    if (!parseJointSafety(*joint.safety, safety_xml))
    {
      CONSOLE_BRIDGE_logError("Could not parse safety element for joint [%s]", joint.name.c_str());
      joint.safety.reset();
      return false;
    }
  }

  // Get calibration
  TiXmlElement *calibration_xml = config->FirstChildElement("calibration");
  if (calibration_xml)
  {
    joint.calibration.reset(new JointCalibration());
    if (!parseJointCalibration(*joint.calibration, calibration_xml))
    {
      CONSOLE_BRIDGE_logError("Could not parse calibration element for joint  [%s]", joint.name.c_str());
      joint.calibration.reset();
      return false;
    }
  }

  // Get Joint Mimic
  TiXmlElement *mimic_xml = config->FirstChildElement("mimic");
  if (mimic_xml)
  {
    joint.mimic.reset(new JointMimic());
    if (!parseJointMimic(*joint.mimic, mimic_xml))
    {
      CONSOLE_BRIDGE_logError("Could not parse mimic element for joint  [%s]", joint.name.c_str());
      joint.mimic.reset();
      return false;
    }
  }

  // Get Dynamics
  TiXmlElement *prop_xml = config->FirstChildElement("dynamics");
  if (prop_xml)
  {
    joint.dynamics.reset(new JointDynamics());
    if (!parseJointDynamics(*joint.dynamics, prop_xml))
    {
      CONSOLE_BRIDGE_logError("Could not parse joint_dynamics element for joint [%s]", joint.name.c_str());
      joint.dynamics.reset();
      return false;
    }
  }

  return true;
}


/* exports */
bool exportPose(Pose &pose, TiXmlElement* xml);

bool exportJointDynamics(JointDynamics &jd, TiXmlElement* xml)
{
  TiXmlElement *dynamics_xml = new TiXmlElement("dynamics");
  dynamics_xml->SetAttribute("damping", urdf_export_helpers::values2str(jd.damping) );
  dynamics_xml->SetAttribute("friction", urdf_export_helpers::values2str(jd.friction) );
  xml->LinkEndChild(dynamics_xml);
  return true;
}

bool exportJointLimits(JointLimits &jl, TiXmlElement* xml)
{
  TiXmlElement *limit_xml = new TiXmlElement("limit");
  limit_xml->SetAttribute("effort", urdf_export_helpers::values2str(jl.effort) );
  limit_xml->SetAttribute("velocity", urdf_export_helpers::values2str(jl.velocity) );
  limit_xml->SetAttribute("lower", urdf_export_helpers::values2str(jl.lower) );
  limit_xml->SetAttribute("upper", urdf_export_helpers::values2str(jl.upper) );
  xml->LinkEndChild(limit_xml);
  return true;
}

bool exportJointSafety(JointSafety &js, TiXmlElement* xml)
{
  TiXmlElement *safety_xml = new TiXmlElement("safety_controller");
  safety_xml->SetAttribute("k_position", urdf_export_helpers::values2str(js.k_position) );
  safety_xml->SetAttribute("k_velocity", urdf_export_helpers::values2str(js.k_velocity) );
  safety_xml->SetAttribute("soft_lower_limit", urdf_export_helpers::values2str(js.soft_lower_limit) );
  safety_xml->SetAttribute("soft_upper_limit", urdf_export_helpers::values2str(js.soft_upper_limit) );
  xml->LinkEndChild(safety_xml);
  return true;
}

bool exportJointCalibration(JointCalibration &jc, TiXmlElement* xml)
{
  if (jc.falling || jc.rising)
  {
    TiXmlElement *calibration_xml = new TiXmlElement("calibration");
    if (jc.falling)
      calibration_xml->SetAttribute("falling", urdf_export_helpers::values2str(*jc.falling) );
    if (jc.rising)
      calibration_xml->SetAttribute("rising", urdf_export_helpers::values2str(*jc.rising) );
    //calibration_xml->SetAttribute("reference_position", urdf_export_helpers::values2str(jc.reference_position) );
    xml->LinkEndChild(calibration_xml);
  }
  return true;
}

bool exportJointMimic(JointMimic &jm, TiXmlElement* xml)
{
  if (!jm.joint_name.empty())
  {
    TiXmlElement *mimic_xml = new TiXmlElement("mimic");
    mimic_xml->SetAttribute("offset", urdf_export_helpers::values2str(jm.offset) );
    mimic_xml->SetAttribute("multiplier", urdf_export_helpers::values2str(jm.multiplier) );
    mimic_xml->SetAttribute("joint", jm.joint_name );
    xml->LinkEndChild(mimic_xml);
  }
  return true;
}

bool exportJoint(Joint &joint, TiXmlElement* xml)
{
  TiXmlElement * joint_xml = new TiXmlElement("joint");
  joint_xml->SetAttribute("name", joint.name);
  if (joint.type == urdf::Joint::PLANAR)
    joint_xml->SetAttribute("type", "planar");
  else if (joint.type == urdf::Joint::FLOATING)
    joint_xml->SetAttribute("type", "floating");
  else if (joint.type == urdf::Joint::REVOLUTE)
    joint_xml->SetAttribute("type", "revolute");
  else if (joint.type == urdf::Joint::CONTINUOUS)
    joint_xml->SetAttribute("type", "continuous");
  else if (joint.type == urdf::Joint::PRISMATIC)
    joint_xml->SetAttribute("type", "prismatic");
  else if (joint.type == urdf::Joint::FIXED)
    joint_xml->SetAttribute("type", "fixed");
  else
    CONSOLE_BRIDGE_logError("ERROR:  Joint [%s] type [%d] is not a defined type.\n",joint.name.c_str(), joint.type);

  // origin
  exportPose(joint.parent_to_joint_origin_transform, joint_xml);

  // axis
  TiXmlElement * axis_xml = new TiXmlElement("axis");
  axis_xml->SetAttribute("xyz", urdf_export_helpers::values2str(joint.axis));
  joint_xml->LinkEndChild(axis_xml);

  // parent 
  TiXmlElement * parent_xml = new TiXmlElement("parent");
  parent_xml->SetAttribute("link", joint.parent_link_name);
  joint_xml->LinkEndChild(parent_xml);

  // child
  TiXmlElement * child_xml = new TiXmlElement("child");
  child_xml->SetAttribute("link", joint.child_link_name);
  joint_xml->LinkEndChild(child_xml);

  if (joint.dynamics)
    exportJointDynamics(*(joint.dynamics), joint_xml);
  if (joint.limits)
    exportJointLimits(*(joint.limits), joint_xml);
  if (joint.safety)
    exportJointSafety(*(joint.safety), joint_xml);
  if (joint.calibration)
    exportJointCalibration(*(joint.calibration), joint_xml);
  if (joint.mimic)
    exportJointMimic(*(joint.mimic), joint_xml);

  xml->LinkEndChild(joint_xml);
  return true;
}



}
