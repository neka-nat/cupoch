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


#include <urdf_parser/urdf_parser.h>
#include <urdf_model/link.h>
#include <fstream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <tinyxml.h>
#include <console_bridge/console.h>

namespace urdf{

bool parsePose(Pose &pose, TiXmlElement* xml);

bool parseMaterial(Material &material, TiXmlElement *config, bool only_name_is_ok)
{
  bool has_rgb = false;
  bool has_filename = false;

  material.clear();

  if (!config->Attribute("name"))
  {
    CONSOLE_BRIDGE_logError("Material must contain a name attribute");
    return false;
  }
  
  material.name = config->Attribute("name");

  // texture
  TiXmlElement *t = config->FirstChildElement("texture");
  if (t)
  {
    if (t->Attribute("filename"))
    {
      material.texture_filename = t->Attribute("filename");
      has_filename = true;
    }
  }

  // color
  TiXmlElement *c = config->FirstChildElement("color");
  if (c)
  {
    if (c->Attribute("rgba")) {

      try {
        material.color.init(c->Attribute("rgba"));
        has_rgb = true;
      }
      catch (ParseError &e) {  
        material.color.clear();
        CONSOLE_BRIDGE_logError(std::string("Material [" + material.name + "] has malformed color rgba values: " + e.what()).c_str());
      }
    }
  }

  if (!has_rgb && !has_filename) {
    if (!only_name_is_ok) // no need for an error if only name is ok
    {
      if (!has_rgb) CONSOLE_BRIDGE_logError(std::string("Material ["+material.name+"] color has no rgba").c_str());
      if (!has_filename) CONSOLE_BRIDGE_logError(std::string("Material ["+material.name+"] not defined in file").c_str());
    }
    return false;
  }
  return true;
}


bool parseSphere(Sphere &s, TiXmlElement *c)
{
  s.clear();

  s.type = Geometry::SPHERE;
  if (!c->Attribute("radius"))
  {
    CONSOLE_BRIDGE_logError("Sphere shape must have a radius attribute");
    return false;
  }

  try {
    s.radius = strToDouble(c->Attribute("radius"));
  } catch(std::runtime_error &) {
    std::stringstream stm;
    stm << "radius [" << c->Attribute("radius") << "] is not a valid float";
    CONSOLE_BRIDGE_logError(stm.str().c_str());
    return false;
  }

  return true;
}

bool parseBox(Box &b, TiXmlElement *c)
{
  b.clear();
  
  b.type = Geometry::BOX;
  if (!c->Attribute("size"))
  {
    CONSOLE_BRIDGE_logError("Box shape has no size attribute");
    return false;
  }
  try
  {
    b.dim.init(c->Attribute("size"));
  }
  catch (ParseError &e)
  {
    b.dim.clear();
    CONSOLE_BRIDGE_logError(e.what());
    return false;
  }
  return true;
}

bool parseCylinder(Cylinder &y, TiXmlElement *c)
{
  y.clear();

  y.type = Geometry::CYLINDER;
  if (!c->Attribute("length") ||
      !c->Attribute("radius"))
  {
    CONSOLE_BRIDGE_logError("Cylinder shape must have both length and radius attributes");
    return false;
  }

  try {
    y.length = strToDouble(c->Attribute("length"));
  } catch(std::runtime_error &) {
    std::stringstream stm;
    stm << "length [" << c->Attribute("length") << "] is not a valid float";
    CONSOLE_BRIDGE_logError(stm.str().c_str());
    return false;
  }

  try {
    y.radius = strToDouble(c->Attribute("radius"));
  } catch(std::runtime_error &) {
    std::stringstream stm;
    stm << "radius [" << c->Attribute("radius") << "] is not a valid float";
    CONSOLE_BRIDGE_logError(stm.str().c_str());
    return false;
  }

  return true;
}


bool parseMesh(Mesh &m, TiXmlElement *c)
{
  m.clear();

  m.type = Geometry::MESH;
  if (!c->Attribute("filename")) {
    CONSOLE_BRIDGE_logError("Mesh must contain a filename attribute");
    return false;
  }

  m.filename = c->Attribute("filename");

  if (c->Attribute("scale")) {
    try {
      m.scale.init(c->Attribute("scale"));
    }
    catch (ParseError &e) {
      m.scale.clear();
      CONSOLE_BRIDGE_logError("Mesh scale was specified, but could not be parsed: %s", e.what());
      return false;
    }
  }
  else
  {
    m.scale.x = m.scale.y = m.scale.z = 1;
  }
  return true;
}

GeometrySharedPtr parseGeometry(TiXmlElement *g)
{
  GeometrySharedPtr geom;
  if (!g) return geom;

  TiXmlElement *shape = g->FirstChildElement();
  if (!shape)
  {
    CONSOLE_BRIDGE_logError("Geometry tag contains no child element.");
    return geom;
  }

  std::string type_name = shape->ValueStr();
  if (type_name == "sphere")
  {
    Sphere *s = new Sphere();
    geom.reset(s);
    if (parseSphere(*s, shape))
      return geom;
  }
  else if (type_name == "box")
  {
    Box *b = new Box();
    geom.reset(b);
    if (parseBox(*b, shape))
      return geom;
  }
  else if (type_name == "cylinder")
  {
    Cylinder *c = new Cylinder();
    geom.reset(c);
    if (parseCylinder(*c, shape))
      return geom;
  }
  else if (type_name == "mesh")
  {
    Mesh *m = new Mesh();
    geom.reset(m);
    if (parseMesh(*m, shape))
      return geom;    
  }
  else
  {
    CONSOLE_BRIDGE_logError("Unknown geometry type '%s'", type_name.c_str());
    return geom;
  }
  
  return GeometrySharedPtr();
}

bool parseInertial(Inertial &i, TiXmlElement *config)
{
  i.clear();

  // Origin
  TiXmlElement *o = config->FirstChildElement("origin");
  if (o)
  {
    if (!parsePose(i.origin, o))
      return false;
  }

  TiXmlElement *mass_xml = config->FirstChildElement("mass");
  if (!mass_xml)
  {
    CONSOLE_BRIDGE_logError("Inertial element must have a mass element");
    return false;
  }
  if (!mass_xml->Attribute("value"))
  {
    CONSOLE_BRIDGE_logError("Inertial: mass element must have value attribute");
    return false;
  }

  try {
    i.mass = strToDouble(mass_xml->Attribute("value"));
  } catch(std::runtime_error &) {
    std::stringstream stm;
    stm << "Inertial: mass [" << mass_xml->Attribute("value")
        << "] is not a float";
    CONSOLE_BRIDGE_logError(stm.str().c_str());
    return false;
  }

  TiXmlElement *inertia_xml = config->FirstChildElement("inertia");
  if (!inertia_xml)
  {
    CONSOLE_BRIDGE_logError("Inertial element must have inertia element");
    return false;
  }

  std::vector<std::pair<std::string, double>> attrs{
      std::make_pair("ixx", 0.0),
      std::make_pair("ixy", 0.0),
      std::make_pair("ixz", 0.0),
      std::make_pair("iyy", 0.0),
      std::make_pair("iyz", 0.0),
      std::make_pair("izz", 0.0)
  };

  for (auto& attr : attrs)
  {
    if (!inertia_xml->Attribute(attr.first))
    {
      std::stringstream stm;
      stm << "Inertial: inertia element missing " << attr.first << " attribute";
      CONSOLE_BRIDGE_logError(stm.str().c_str());
      return false;
    }

    try {
      attr.second = strToDouble(inertia_xml->Attribute(attr.first.c_str()));
    } catch(std::runtime_error &) {
      std::stringstream stm;
      stm << "Inertial: inertia element " << attr.first << " is not a valid double";
      CONSOLE_BRIDGE_logError(stm.str().c_str());
      return false;
    }
  }

  i.ixx = attrs[0].second;
  i.ixy = attrs[1].second;
  i.ixz = attrs[2].second;
  i.iyy = attrs[3].second;
  i.iyz = attrs[4].second;
  i.izz = attrs[5].second;

  return true;
}

bool parseVisual(Visual &vis, TiXmlElement *config)
{
  vis.clear();

  // Origin
  TiXmlElement *o = config->FirstChildElement("origin");
  if (o) {
    if (!parsePose(vis.origin, o))
      return false;
  }

  // Geometry
  TiXmlElement *geom = config->FirstChildElement("geometry");
  vis.geometry = parseGeometry(geom);
  if (!vis.geometry)
    return false;

  const char *name_char = config->Attribute("name");
  if (name_char)
    vis.name = name_char;

  // Material
  TiXmlElement *mat = config->FirstChildElement("material");
  if (mat) {
    // get material name
    if (!mat->Attribute("name")) {
      CONSOLE_BRIDGE_logError("Visual material must contain a name attribute");
      return false;
    }
    vis.material_name = mat->Attribute("name");
    
    // try to parse material element in place
    vis.material.reset(new Material());
    if (!parseMaterial(*vis.material, mat, true))
    {
      vis.material.reset();
    }
  }
  
  return true;
}

bool parseCollision(Collision &col, TiXmlElement* config)
{  
  col.clear();

  // Origin
  TiXmlElement *o = config->FirstChildElement("origin");
  if (o) {
    if (!parsePose(col.origin, o))
      return false;
  }
  
  // Geometry
  TiXmlElement *geom = config->FirstChildElement("geometry");
  col.geometry = parseGeometry(geom);
  if (!col.geometry)
    return false;

  const char *name_char = config->Attribute("name");
  if (name_char)
    col.name = name_char;

  return true;
}

bool parseLink(Link &link, TiXmlElement* config)
{
  
  link.clear();

  const char *name_char = config->Attribute("name");
  if (!name_char)
  {
    CONSOLE_BRIDGE_logError("No name given for the link.");
    return false;
  }
  link.name = std::string(name_char);

  // Inertial (optional)
  TiXmlElement *i = config->FirstChildElement("inertial");
  if (i)
  {
    link.inertial.reset(new Inertial());
    if (!parseInertial(*link.inertial, i))
    {
      CONSOLE_BRIDGE_logError("Could not parse inertial element for Link [%s]", link.name.c_str());
      return false;
    }
  }

  // Multiple Visuals (optional)
  for (TiXmlElement* vis_xml = config->FirstChildElement("visual"); vis_xml; vis_xml = vis_xml->NextSiblingElement("visual"))
  {

    VisualSharedPtr vis;
    vis.reset(new Visual());
    if (parseVisual(*vis, vis_xml))
    {
      link.visual_array.push_back(vis);
    }
    else
    {
      vis.reset();
      CONSOLE_BRIDGE_logError("Could not parse visual element for Link [%s]", link.name.c_str());
      return false;
    }
  }

  // Visual (optional)
  // Assign the first visual to the .visual ptr, if it exists
  if (!link.visual_array.empty())
    link.visual = link.visual_array[0];
  
  // Multiple Collisions (optional)
  for (TiXmlElement* col_xml = config->FirstChildElement("collision"); col_xml; col_xml = col_xml->NextSiblingElement("collision"))
  {
    CollisionSharedPtr col;
    col.reset(new Collision());
    if (parseCollision(*col, col_xml))
    {      
      link.collision_array.push_back(col);
    }
    else
    {
      col.reset();
      CONSOLE_BRIDGE_logError("Could not parse collision element for Link [%s]",  link.name.c_str());
      return false;
    }
  }
  
  // Collision (optional)  
  // Assign the first collision to the .collision ptr, if it exists
  if (!link.collision_array.empty())
    link.collision = link.collision_array[0];

  return true;
}

/* exports */
bool exportPose(Pose &pose, TiXmlElement* xml);

bool exportMaterial(Material &material, TiXmlElement *xml)
{
  TiXmlElement *material_xml = new TiXmlElement("material");
  material_xml->SetAttribute("name", material.name);

  TiXmlElement* texture = new TiXmlElement("texture");
  if (!material.texture_filename.empty())
    texture->SetAttribute("filename", material.texture_filename);
  material_xml->LinkEndChild(texture);

  TiXmlElement* color = new TiXmlElement("color");
  color->SetAttribute("rgba", urdf_export_helpers::values2str(material.color));
  material_xml->LinkEndChild(color);
  xml->LinkEndChild(material_xml);
  return true;
}

bool exportSphere(Sphere &s, TiXmlElement *xml)
{
  // e.g. add <sphere radius="1"/>
  TiXmlElement *sphere_xml = new TiXmlElement("sphere");
  sphere_xml->SetAttribute("radius", urdf_export_helpers::values2str(s.radius));
  xml->LinkEndChild(sphere_xml);
  return true;
}

bool exportBox(Box &b, TiXmlElement *xml)
{
  // e.g. add <box size="1 1 1"/>
  TiXmlElement *box_xml = new TiXmlElement("box");
  box_xml->SetAttribute("size", urdf_export_helpers::values2str(b.dim));
  xml->LinkEndChild(box_xml);
  return true;
}

bool exportCylinder(Cylinder &y, TiXmlElement *xml)
{
  // e.g. add <cylinder radius="1"/>
  TiXmlElement *cylinder_xml = new TiXmlElement("cylinder");
  cylinder_xml->SetAttribute("radius", urdf_export_helpers::values2str(y.radius));
  cylinder_xml->SetAttribute("length", urdf_export_helpers::values2str(y.length));
  xml->LinkEndChild(cylinder_xml);
  return true;
}

bool exportMesh(Mesh &m, TiXmlElement *xml)
{
  // e.g. add <mesh filename="my_file" scale="1 1 1"/>
  TiXmlElement *mesh_xml = new TiXmlElement("mesh");
  if (!m.filename.empty())
    mesh_xml->SetAttribute("filename", m.filename);
  mesh_xml->SetAttribute("scale", urdf_export_helpers::values2str(m.scale));
  xml->LinkEndChild(mesh_xml);
  return true;
}

bool exportGeometry(GeometrySharedPtr &geom, TiXmlElement *xml)
{
  TiXmlElement *geometry_xml = new TiXmlElement("geometry");
  if (urdf::dynamic_pointer_cast<Sphere>(geom))
  {
    exportSphere((*(urdf::dynamic_pointer_cast<Sphere>(geom).get())), geometry_xml);
  }
  else if (urdf::dynamic_pointer_cast<Box>(geom))
  {
    exportBox((*(urdf::dynamic_pointer_cast<Box>(geom).get())), geometry_xml);
  }
  else if (urdf::dynamic_pointer_cast<Cylinder>(geom))
  {
    exportCylinder((*(urdf::dynamic_pointer_cast<Cylinder>(geom).get())), geometry_xml);
  }
  else if (urdf::dynamic_pointer_cast<Mesh>(geom))
  {
    exportMesh((*(urdf::dynamic_pointer_cast<Mesh>(geom).get())), geometry_xml);
  }
  else
  {
    CONSOLE_BRIDGE_logError("geometry not specified, I'll make one up for you!");
    Sphere *s = new Sphere();
    s->radius = 0.03;
    geom.reset(s);
    exportSphere((*(urdf::dynamic_pointer_cast<Sphere>(geom).get())), geometry_xml);
  }

  xml->LinkEndChild(geometry_xml);
  return true;
}

bool exportInertial(Inertial &i, TiXmlElement *xml)
{
  // adds <inertial>
  //        <mass value="1"/>
  //        <pose xyz="0 0 0" rpy="0 0 0"/>
  //        <inertia ixx="1" ixy="0" />
  //      </inertial>
  TiXmlElement *inertial_xml = new TiXmlElement("inertial");

  TiXmlElement *mass_xml = new TiXmlElement("mass");
  mass_xml->SetAttribute("value", urdf_export_helpers::values2str(i.mass));
  inertial_xml->LinkEndChild(mass_xml);

  exportPose(i.origin, inertial_xml);

  TiXmlElement *inertia_xml = new TiXmlElement("inertia");
  inertia_xml->SetAttribute("ixx", urdf_export_helpers::values2str(i.ixx));
  inertia_xml->SetAttribute("ixy", urdf_export_helpers::values2str(i.ixy));
  inertia_xml->SetAttribute("ixz", urdf_export_helpers::values2str(i.ixz));
  inertia_xml->SetAttribute("iyy", urdf_export_helpers::values2str(i.iyy));
  inertia_xml->SetAttribute("iyz", urdf_export_helpers::values2str(i.iyz));
  inertia_xml->SetAttribute("izz", urdf_export_helpers::values2str(i.izz));
  inertial_xml->LinkEndChild(inertia_xml);

  xml->LinkEndChild(inertial_xml);
  
  return true;
}

bool exportVisual(Visual &vis, TiXmlElement *xml)
{
  // <visual group="default">
  //   <origin rpy="0 0 0" xyz="0 0 0"/>
  //   <geometry>
  //     <mesh filename="mesh.dae"/>
  //   </geometry>
  //   <material name="Grey"/>
  // </visual>
  TiXmlElement * visual_xml = new TiXmlElement("visual");

  exportPose(vis.origin, visual_xml);

  exportGeometry(vis.geometry, visual_xml);

  if (vis.material)
    exportMaterial(*vis.material, visual_xml);

  xml->LinkEndChild(visual_xml);

  return true;
}

bool exportCollision(Collision &col, TiXmlElement* xml)
{  
  // <collision group="default">
  //   <origin rpy="0 0 0" xyz="0 0 0"/>
  //   <geometry>
  //     <mesh filename="mesh.dae"/>
  //   </geometry>
  //   <material name="Grey"/>
  // </collision>
  TiXmlElement * collision_xml = new TiXmlElement("collision");

  exportPose(col.origin, collision_xml);

  exportGeometry(col.geometry, collision_xml);

  xml->LinkEndChild(collision_xml);

  return true;
}

bool exportLink(Link &link, TiXmlElement* xml)
{
  TiXmlElement * link_xml = new TiXmlElement("link");
  link_xml->SetAttribute("name", link.name);

  if (link.inertial)
    exportInertial(*link.inertial, link_xml);
  for (std::size_t i = 0 ; i < link.visual_array.size() ; ++i)
    exportVisual(*link.visual_array[i], link_xml);
  for (std::size_t i = 0 ; i < link.collision_array.size() ; ++i)
    exportCollision(*link.collision_array[i], link_xml);

  xml->LinkEndChild(link_xml);

  return true;
}

}
