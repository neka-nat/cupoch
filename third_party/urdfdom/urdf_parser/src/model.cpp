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

#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include "urdf_parser/urdf_parser.h"
#include <console_bridge/console.h>

namespace urdf{

bool parseMaterial(Material &material, TiXmlElement *config, bool only_name_is_ok);
bool parseLink(Link &link, TiXmlElement *config);
bool parseJoint(Joint &joint, TiXmlElement *config);

ModelInterfaceSharedPtr  parseURDFFile(const std::string &path)
{
    std::ifstream stream( path.c_str() );
    if (!stream)
    {
      CONSOLE_BRIDGE_logError(("File " + path + " does not exist").c_str());
      return ModelInterfaceSharedPtr();
    }

    std::string xml_str((std::istreambuf_iterator<char>(stream)),
	                     std::istreambuf_iterator<char>());
    return urdf::parseURDF( xml_str );
}

bool assignMaterial(const VisualSharedPtr& visual, ModelInterfaceSharedPtr& model, const char* link_name)
{
  if (visual->material_name.empty())
    return true;

  const MaterialSharedPtr& material = model->getMaterial(visual->material_name);
  if (material)
  {
    CONSOLE_BRIDGE_logDebug("urdfdom: setting link '%s' material to '%s'", link_name, visual->material_name.c_str());
    visual->material = material;
  }
  else
  {
    if (visual->material)
    {
      CONSOLE_BRIDGE_logDebug("urdfdom: link '%s' material '%s' defined in Visual.", link_name, visual->material_name.c_str());
      model->materials_.insert(make_pair(visual->material->name, visual->material));
    }
    else
    {
      CONSOLE_BRIDGE_logWarn("link '%s' material '%s' undefined.", link_name,visual->material_name.c_str());
      return false;
    }
  }
  return true;
}

ModelInterfaceSharedPtr  parseURDF(const std::string &xml_string)
{
  ModelInterfaceSharedPtr model(new ModelInterface);
  model->clear();

  TiXmlDocument xml_doc;
  xml_doc.Parse(xml_string.c_str());
  if (xml_doc.Error())
  {
    CONSOLE_BRIDGE_logError(xml_doc.ErrorDesc());
    xml_doc.ClearError();
    model.reset();
    return model;
  }

  TiXmlElement *robot_xml = xml_doc.FirstChildElement("robot");
  if (!robot_xml)
  {
    CONSOLE_BRIDGE_logError("Could not find the 'robot' element in the xml file");
    model.reset();
    return model;
  }

  // Get robot name
  const char *name = robot_xml->Attribute("name");
  if (!name)
  {
    CONSOLE_BRIDGE_logError("No name given for the robot.");
    model.reset();
    return model;
  }
  model->name_ = std::string(name);

  try
  {
    urdf_export_helpers::URDFVersion version(robot_xml->Attribute("version"));
    if (!version.equal(1, 0))
    {
      throw std::runtime_error("Invalid 'version' specified; only version 1.0 is currently supported");
    }
  }
  catch (const std::runtime_error & err)
  {
    CONSOLE_BRIDGE_logError(err.what());
    model.reset();
    return model;
  }

  // Get all Material elements
  for (TiXmlElement* material_xml = robot_xml->FirstChildElement("material"); material_xml; material_xml = material_xml->NextSiblingElement("material"))
  {
    MaterialSharedPtr material;
    material.reset(new Material);

    try {
      parseMaterial(*material, material_xml, false); // material needs to be fully defined here
      if (model->getMaterial(material->name))
      {
        CONSOLE_BRIDGE_logError("material '%s' is not unique.", material->name.c_str());
        material.reset();
        model.reset();
        return model;
      }
      else
      {
        model->materials_.insert(make_pair(material->name,material));
        CONSOLE_BRIDGE_logDebug("urdfdom: successfully added a new material '%s'", material->name.c_str());
      }
    }
    catch (ParseError &/*e*/) {
      CONSOLE_BRIDGE_logError("material xml is not initialized correctly");
      material.reset();
      model.reset();
      return model;
    }
  }

  // Get all Link elements
  for (TiXmlElement* link_xml = robot_xml->FirstChildElement("link"); link_xml; link_xml = link_xml->NextSiblingElement("link"))
  {
    LinkSharedPtr link;
    link.reset(new Link);

    try {
      parseLink(*link, link_xml);
      if (model->getLink(link->name))
      {
        CONSOLE_BRIDGE_logError("link '%s' is not unique.", link->name.c_str());
        model.reset();
        return model;
      }
      else
      {
        // set link visual(s) material
        CONSOLE_BRIDGE_logDebug("urdfdom: setting link '%s' material", link->name.c_str());
        if (link->visual)
        {
          assignMaterial(link->visual, model, link->name.c_str());
        }
        for (const auto& visual : link->visual_array)
        {
          assignMaterial(visual, model, link->name.c_str());
        }

        model->links_.insert(make_pair(link->name,link));
        CONSOLE_BRIDGE_logDebug("urdfdom: successfully added a new link '%s'", link->name.c_str());
      }
    }
    catch (ParseError &/*e*/) {
      CONSOLE_BRIDGE_logError("link xml is not initialized correctly");
      model.reset();
      return model;
    }
  }
  if (model->links_.empty()){
    CONSOLE_BRIDGE_logError("No link elements found in urdf file");
    model.reset();
    return model;
  }

  // Get all Joint elements
  for (TiXmlElement* joint_xml = robot_xml->FirstChildElement("joint"); joint_xml; joint_xml = joint_xml->NextSiblingElement("joint"))
  {
    JointSharedPtr joint;
    joint.reset(new Joint);

    if (parseJoint(*joint, joint_xml))
    {
      if (model->getJoint(joint->name))
      {
        CONSOLE_BRIDGE_logError("joint '%s' is not unique.", joint->name.c_str());
        model.reset();
        return model;
      }
      else
      {
        model->joints_.insert(make_pair(joint->name,joint));
        CONSOLE_BRIDGE_logDebug("urdfdom: successfully added a new joint '%s'", joint->name.c_str());
      }
    }
    else
    {
      CONSOLE_BRIDGE_logError("joint xml is not initialized correctly");
      model.reset();
      return model;
    }
  }


  // every link has children links and joints, but no parents, so we create a
  // local convenience data structure for keeping child->parent relations
  std::map<std::string, std::string> parent_link_tree;
  parent_link_tree.clear();

  // building tree: name mapping
  try 
  {
    model->initTree(parent_link_tree);
  }
  catch(ParseError &e)
  {
    CONSOLE_BRIDGE_logError("Failed to build tree: %s", e.what());
    model.reset();
    return model;
  }

  // find the root link
  try
  {
    model->initRoot(parent_link_tree);
  }
  catch(ParseError &e)
  {
    CONSOLE_BRIDGE_logError("Failed to find root link: %s", e.what());
    model.reset();
    return model;
  }
  
  return model;
}

bool exportMaterial(Material &material, TiXmlElement *config);
bool exportLink(Link &link, TiXmlElement *config);
bool exportJoint(Joint &joint, TiXmlElement *config);
TiXmlDocument*  exportURDF(const ModelInterface &model)
{
  TiXmlDocument *doc = new TiXmlDocument();

  TiXmlElement *robot = new TiXmlElement("robot");
  robot->SetAttribute("name", model.name_);
  doc->LinkEndChild(robot);


  for (std::map<std::string, MaterialSharedPtr>::const_iterator m=model.materials_.begin(); m!=model.materials_.end(); m++)
  {
    CONSOLE_BRIDGE_logDebug("urdfdom: exporting material [%s]\n",m->second->name.c_str());
    exportMaterial(*(m->second), robot);
  }

  for (std::map<std::string, LinkSharedPtr>::const_iterator l=model.links_.begin(); l!=model.links_.end(); l++)  
  {
    CONSOLE_BRIDGE_logDebug("urdfdom: exporting link [%s]\n",l->second->name.c_str());
    exportLink(*(l->second), robot);
  }
  	
  for (std::map<std::string, JointSharedPtr>::const_iterator j=model.joints_.begin(); j!=model.joints_.end(); j++)  
  {
    CONSOLE_BRIDGE_logDebug("urdfdom: exporting joint [%s]\n",j->second->name.c_str());
    exportJoint(*(j->second), robot);
  }

  return doc;
}
    
TiXmlDocument*  exportURDF(ModelInterfaceSharedPtr &model)
{
  return exportURDF(*model);
}


}

