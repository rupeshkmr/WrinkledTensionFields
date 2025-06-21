# Build helpers as a static library
add_library(SFF STATIC 
    MidedgeAngleSinFormulation.cpp
    MidedgeAngleTanFormulation.cpp
    MidedgeAverageFormulation.cpp
    SecondFundamentalFormDiscretization.cpp
)

# Expose headers to dependents (like libs)
target_link_libraries(SFF Eigen3::Eigen igl::core TBB::tbb)
target_include_directories(SFF PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
