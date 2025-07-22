# Build helpers as a static library
add_library(WRAPPERS STATIC 
    CoMISoWrapper.cpp
    EigenNASOQ.cpp
)


# Set include and library directories
set(NASOQ_INC ${NASOQ_ROOT_DIR}/include)
set(NASOQ_EIGEN_INC ${NASOQ_ROOT_DIR}/eigen_interface/include)
set(NASOQ_LIB ${NASOQ_ROOT_DIR}/lib/)
include_directories(${NASOQ_INC} ${NASOQ_EIGEN_INC})
link_directories(${NASOQ_LIB})
target_link_libraries(WRAPPERS
    nasoq
    Eigen3::Eigen# Eigen target
    igl::core      # libigl core module
    ${NASOQ_LIB}
    igl_copyleft::comiso
    nasoq_eigen
)
# Expose headers to dependents (like libs)
target_include_directories(WRAPPERS PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

