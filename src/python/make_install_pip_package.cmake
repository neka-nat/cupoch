file(GLOB WHEEL_FILE "${PYTHON_PACKAGE_DST_DIR}/pip_package/*.whl")
execute_process(COMMAND pip install ${WHEEL_FILE} -U)