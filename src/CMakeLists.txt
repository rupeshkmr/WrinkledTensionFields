add_executable(main main.cpp)
target_link_libraries(main PRIVATE TFWLib)
target_link_libraries(main PRIVATE Eigen3::Eigen igl::core ${SUITESPARSE_LIBRARIES} polyscope)
        

