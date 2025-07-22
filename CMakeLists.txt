cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(tfwshell)
# Find Eigen and libigl
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_LIST_DIR}/cmake")
include(FetchContent)

# Declare libigl with specific commit for stability
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG 7b6cc27  # Lock to a specific commit
)

# Find TBB package
find_package(TBB REQUIRED)

# Skip if already available
if(NOT TARGET igl::core)
    # Disable ALL optional modules
    set(LIBIGL_GLFW                OFF CACHE BOOL "" FORCE)
    set(LIBIGL_PNG                 OFF CACHE BOOL "" FORCE)
    set(LIBIGL_IMGUI               OFF CACHE BOOL "" FORCE)
    set(LIBIGL_OPENGL              OFF CACHE BOOL "" FORCE)
    set(LIBIGL_PREDICATES          OFF CACHE BOOL "" FORCE)
    set(LIBIGL_COPYLEFT_COMISO     ON  CACHE BOOL "" FORCE)
    set(LIBIGL_COPYLEFT_CORE       OFF CACHE BOOL "" FORCE)
    set(LIBIGL_COPYLEFT_TETGEN     OFF CACHE BOOL "" FORCE)
    set(LIBIGL_COPYLEFT_CGAL       OFF CACHE BOOL "" FORCE)
    set(LIBIGL_RESTRICTED_TRIANGLE OFF CACHE BOOL "" FORCE)

    # Make libigl available
    FetchContent_MakeAvailable(libigl)
    
    message(STATUS "libigl core available")
endif()
#suite-sparse
message("build with suite-sparse")
find_package(SuiteSparse REQUIRED)
include_directories(${SUITESPARSE_INCLUDE_DIRS})

add_subdirectory(vendor)
add_subdirectory(src)
add_subdirectory(SecondFundamentalForm)
add_subdirectory(MeshLib)
add_subdirectory(TFWShell)
add_subdirectory(Wrappers)


