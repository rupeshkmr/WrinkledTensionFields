add_subdirectory(polyscope)
add_subdirectory(lbfgspp)

# NASOQ, refer https://github.com/cheshmi/nasoq-example.git
# set NASOQ properties such BLAS Type, etc
set(NASOQ_BLAS_BACKEND "OpenBLAS" CACHE STRING "BLAS implementation for NASOQ to use")

if (MSVC)
 set(NASOQ_USE_CLAPACK  ON CACHE STRING "Use CLAPACK as the LAPACK implementaiton")
endif()

add_subdirectory(nasoq)

if(NOT DEFINED NASOQ_ROOT)
    set(NASOQ_ROOT_DIR "nasoq") #assumed the example folder is within nasoq
else()
    set(NASOQ_ROOT_DIR "${NASOQ_ROOT}")
endif()


