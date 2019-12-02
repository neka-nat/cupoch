file(GLOB_RECURSE CLASS_IO_SOURCE_FILES "class_io/*.cpp")
file(GLOB_RECURSE FILE_FORMAT_SOURCE_FILES "file_format/*.cu")
set(IO_ALL_SOURCE_FILES ${CLASS_IO_SOURCE_FILES} ${FILE_FORMAT_SOURCE_FILES})

# Create object library
cuda_add_library(cupoch_io ${IO_ALL_SOURCE_FILES})
target_link_libraries(cupoch_io cupoch_geometry cupoch_utility liblzf)