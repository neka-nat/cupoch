file(GLOB_RECURSE KINFU_SOURCE_FILES "*.cpp")

# Create object library
cuda_add_library(cupoch_kinfu ${KINFU_SOURCE_FILES})
target_link_libraries(cupoch_kinfu cupoch_integration
                      cupoch_registration
                      ${3RDPARTY_LIBRARIES})