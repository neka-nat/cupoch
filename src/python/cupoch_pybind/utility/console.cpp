#include "cupoch/utility/console.h"

#include "cupoch_pybind/cupoch_pybind.h"
#include "cupoch_pybind/docstring.h"

using namespace cupoch;

void pybind_console(py::module &m) {
    py::enum_<utility::VerbosityLevel> vl(m, "VerbosityLevel", py::arithmetic(),
                                          "VerbosityLevel");
    vl.value("Error", utility::VerbosityLevel::Error)
            .value("Warning", utility::VerbosityLevel::Warning)
            .value("Info", utility::VerbosityLevel::Info)
            .value("Debug", utility::VerbosityLevel::Debug)
            .export_values();
    // Trick to write docs without listing the members in the enum class again.
    vl.attr("__doc__") = docstring::static_property(
            py::cpp_function([](py::handle arg) -> std::string {
                return "Enum class for VerbosityLevel.";
            }),
            py::none(), py::none(), "");

    m.def("set_verbosity_level", &utility::SetVerbosityLevel,
          "Set global verbosity level of Open3D", py::arg("verbosity_level"));
    docstring::FunctionDocInject(
            m, "set_verbosity_level",
            {{"verbosity_level",
              "Messages with equal or less than ``verbosity_level`` verbosity "
              "will be printed."}});

    m.def("get_verbosity_level", &utility::GetVerbosityLevel,
          "Get global verbosity level of Cupoch");
    docstring::FunctionDocInject(m, "get_verbosity_level");
}
