#pragma once

#include <string>

namespace cupoch {

namespace utility {
class IJsonConvertible;
}

namespace io {

/// The general entrance for reading an IJsonConvertible from a file
/// The function calls read functions based on the extension name of filename.
/// \return return true if the read function is successful, false otherwise.
bool ReadIJsonConvertible(const std::string &filename,
                          utility::IJsonConvertible &object);

/// The general entrance for writing an IJsonConvertible to a file
/// The function calls write functions based on the extension name of filename.
/// \return return true if the write function is successful, false otherwise.
bool WriteIJsonConvertible(const std::string &filename,
                           const utility::IJsonConvertible &object);

bool ReadIJsonConvertibleFromJSON(const std::string &filename,
                                  utility::IJsonConvertible &object);

bool WriteIJsonConvertibleToJSON(const std::string &filename,
                                 const utility::IJsonConvertible &object);

bool ReadIJsonConvertibleFromJSONString(const std::string &json_string,
                                        utility::IJsonConvertible &object);

bool WriteIJsonConvertibleToJSONString(std::string &json_string,
                                       const utility::IJsonConvertible &object);

}  // namespace io
}  // namespace cupoch