file(GLOB_RECURSE VISUALIZATION_ALL_SOURCE_FILES "*.cpp")

# create object library
add_library(Visualization OBJECT
            ${VISUALIZATION_ALL_SOURCE_FILES})