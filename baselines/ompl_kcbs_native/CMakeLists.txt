cmake_minimum_required(VERSION 3.16)
project(stgcs_ompl_kcbs_native LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
find_package(ompl REQUIRED)

pybind11_add_module(_ompl_kcbs_native ompl_kcbs_native.cpp)
target_compile_features(_ompl_kcbs_native PRIVATE cxx_std_17)

if(TARGET ompl::ompl)
  target_link_libraries(_ompl_kcbs_native PRIVATE ompl::ompl)
elseif(TARGET ompl)
  target_link_libraries(_ompl_kcbs_native PRIVATE ompl)
else()
  target_include_directories(_ompl_kcbs_native PRIVATE ${OMPL_INCLUDE_DIRS})
  target_link_libraries(_ompl_kcbs_native PRIVATE ${OMPL_LIBRARIES})
endif()

set_target_properties(
  _ompl_kcbs_native
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
