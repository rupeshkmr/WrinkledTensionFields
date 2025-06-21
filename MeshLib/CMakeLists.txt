# Build helpers as a static library
add_library(MeshLib STATIC 
    GeometryDerivatives.cpp
    IntrinsicGeometry.cpp
    MeshConnectivity.cpp
    MeshGeometry.cpp
)
target_link_libraries(MeshLib 
    Eigen3::Eigen# Eigen target
    igl::core      # libigl core module
)
# Expose headers to dependents (like libs)
target_include_directories(MeshLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

