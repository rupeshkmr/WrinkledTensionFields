# Collect all library sources
file(GLOB EXECUTABLES *.cpp)
file(GLOB SFFSOURCES {PROJECT_SOURCE_DIR}/SecondFundamentalForm/*.cpp)
file(GLOB MESHLIBSOURCES {PROJECT_SOURCE_DIR}/MeshLib/*.cpp)

# Build static library from sources
add_library(TFWLib STATIC ${EXECUTABLES})
target_link_libraries(TFWLib PRIVATE SFF)
target_link_libraries(TFWLib PRIVATE MeshLib)
target_link_libraries(TFWLib PRIVATE Eigen3::Eigen igl::core)
# Make library headers accessible
target_include_directories(TFWLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

