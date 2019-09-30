
FILE(GLOB_RECURSE RVTENSOR_SRCS
    "${CMAKE_CURRENT_LIST_DIR}/*/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/*/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/*/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/*/*.c"
    "${CMAKE_CURRENT_LIST_DIR}/*/*.s"
    "${CMAKE_CURRENT_LIST_DIR}/*/*.S"
    )

if(RISCV)
    include_directories(${THIRD_PARTY_DIR}/hdf5-1.10.5-release/include)
elseif(X86)
    include_directories(${THIRD_PARTY_DIR}/hdf5-1.10.5-release/include)
endif()


add_library(RVTensor STATIC ${RVTENSOR_SRCS})
