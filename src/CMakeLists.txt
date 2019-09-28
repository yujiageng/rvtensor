
FILE(GLOB_RECURSE RVTENSOR_SRCS
    "${CMAKE_CURRENT_LIST_DIR}/*/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/*/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/*/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/*/*.c"
    "${CMAKE_CURRENT_LIST_DIR}/*/*.s"
    "${CMAKE_CURRENT_LIST_DIR}/*/*.S"
    )

include_directories(${THIRD_PARTY_DIR}/hdf5-1.10.5-release/include)
#link_directories(${THIRD_PARTY_DIR}/hdf5-1.10.5-release/lib)

add_library(RVTensor STATIC ${RVTENSOR_SRCS})
