set(PACKAGE_NAME cupoch)

file(GLOB_RECURSE PY_ALL_SOURCE_FILES "cupoch_pybind/*.cpp")

add_library(${PACKAGE_NAME} SHARED
    ${PY_ALL_SOURCE_FILES}
)

cuda_add_library(cupoch_wrapper cupoch_pybind/device_vector_wrapper.cu
    cupoch_pybind/device_map_wrapper.cu)
target_include_directories(cupoch_wrapper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(cupoch_wrapper cupoch_geometry)

target_include_directories(${PACKAGE_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
# Suppress Pybind11 warnings
target_include_directories(${PACKAGE_NAME} SYSTEM PRIVATE
    ${PYBIND11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS}
)

if (${PYTHON_VERSION_MAJOR} EQUAL 2)
    target_compile_definitions(${PACKAGE_NAME} PRIVATE PYTHON_2_FALLBACK)
endif ()

target_link_libraries(${PACKAGE_NAME} PRIVATE cupoch_registration
                      cupoch_visualization cupoch_io cupoch_odometry cupoch_geometry cupoch_utility
                      cupoch_wrapper
                      ${3RDPARTY_LIBRARIES} ${CUDA_LIBRARIES})

set_target_properties(${PACKAGE_NAME} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
set_target_properties(${PACKAGE_NAME} PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")
set_target_properties(${PACKAGE_NAME} PROPERTIES CXX_VISIBILITY_PRESET "hidden")
set_target_properties(${PACKAGE_NAME} PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
target_compile_options(${PACKAGE_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${PYBIND11_CPP_STANDARD}>)

# At `make`: cupoch.so (or the equivalents) will be created at
# PYTHON_COMPILED_MODULE_DIR. The default locaiton is `build/lib/python`
set(PYTHON_COMPILED_MODULE_DIR "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/python")
set_target_properties(${PACKAGE_NAME} PROPERTIES
                      FOLDER "python"
                      LIBRARY_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}"
                      ARCHIVE_OUTPUT_DIRECTORY "${PYTHON_COMPILED_MODULE_DIR}")

# Use `make python-package` to create the python package in the build directory
# The python package will be created at PYTHON_PACKAGE_DIR. It contains:
# 1) Pure-python code and misc files, copied from src/python/package
# 2) The compiled python-C++ module, i.e. cupoch.so (or the equivalents)
# 3) Configured files and supporting files
# Note: `make python-package` clears PYTHON_COMPILED_MODULE_DIR first every time
set(PYTHON_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/lib/python_package")
message(STATUS "PYPI_PACKAGE_NAME: ${PYPI_PACKAGE_NAME}")

add_custom_target(python-package
    COMMAND ${CMAKE_COMMAND}
            -DPYTHON_PACKAGE_SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
            -DPYTHON_VERSION=${PYTHON_VERSION}
            -DPYTHON_COMPILED_MODULE_PATH=$<TARGET_FILE:${PACKAGE_NAME}>
            -DENABLE_JUPYTER=${ENABLE_JUPYTER}
            -DPROJECT_EMAIL=${PROJECT_EMAIL}
            -DPROJECT_HOME=${PROJECT_HOME}
            -DPROJECT_DOCS=${PROJECT_DOCS}
            -DPROJECT_CODE=${PROJECT_CODE}
            -DPROJECT_ISSUES=${PROJECT_ISSUES}
            -DPROJECT_VERSION=${PROJECT_VERSION}
            -DPROJECT_VERSION_THREE_NUMBER=${PROJECT_VERSION_THREE_NUMBER}
            -DPYPI_PACKAGE_NAME=${PYPI_PACKAGE_NAME}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/make_python_package.cmake
)

# Use `make pip-package` to create the pip package in the build directory
add_custom_target(pip-package
    COMMAND ${PYTHON_EXECUTABLE} setup.py bdist_wheel --dist-dir pip_package
    COMMAND echo "pip wheel created at ${PYTHON_PACKAGE_DST_DIR}/pip_package"
    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
    DEPENDS python-package
)

# Use `make install-pip-package` to install pip wheel package to the current
# python environment.
add_custom_target(install-pip-package
    COMMAND ${CMAKE_COMMAND}
            -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/make_install_pip_package.cmake
    DEPENDS pip-package
)
