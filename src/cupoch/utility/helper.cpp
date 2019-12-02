#include "cupoch/utility/helper.h"

using namespace cupoch;
using namespace cupoch::utility;

void cupoch::utility::SplitString(std::vector<std::string>& tokens,
                                  const std::string& str,
                                  const std::string& delimiters /* = " "*/,
                                  bool trim_empty_str /* = true*/) {
    std::string::size_type pos = 0, new_pos = 0, last_pos = 0;
    while (pos != std::string::npos) {
        pos = str.find_first_of(delimiters, last_pos);
        new_pos = (pos == std::string::npos ? str.length() : pos);
        if (new_pos != last_pos || !trim_empty_str) {
            tokens.push_back(str.substr(last_pos, new_pos - last_pos));
        }
        last_pos = new_pos + 1;
    }
}