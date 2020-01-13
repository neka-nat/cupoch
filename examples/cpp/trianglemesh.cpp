#include "cupoch/cupoch.h"

int main(int argc, char **argv) {
    using namespace cupoch;

    utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
    if (argc < 2) {utility::LogInfo("Need an argument of mesh file name."); return 0;}
    auto mesh = io::CreateMeshFromFile(argv[1]);
    utility::LogDebug("Vertices size : {:d}", mesh->vertices_.size());
    visualization::DrawGeometries({mesh});
}