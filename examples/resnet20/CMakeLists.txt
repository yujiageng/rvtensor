set(QEMU_RESNET20_NAME qemu_resnet20)

FILE(GLOB_RECURSE QEMU_RESNET20_SRCS
    "${CMAKE_CURRENT_LIST_DIR}/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/*.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/*.c"
    "${CMAKE_CURRENT_LIST_DIR}/*.s"
    "${CMAKE_CURRENT_LIST_DIR}/*.S"
    )
message(STATUS "CMAKE_CURRENT_BINARY_DIR = ${CMAKE_CURRENT_BINARY_DIR}")
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${THIRD_PARTY_DIR}/hdf5-1.10.5-release/include)
link_directories(${THIRD_PARTY_DIR}/hdf5-1.10.5-release/lib)

add_executable(${QEMU_RESNET20_NAME} ${QEMU_RESNET20_SRCS})
target_link_libraries(${QEMU_RESNET20_NAME} RVTensor
                -Wl,--start-group
                -Wl,--gc-sections
                m atomic c stdc++ hdf5
                m atomic c stdc++
                -Wl,--end-group
                )
set_target_properties(${QEMU_RESNET20_NAME} PROPERTIES LINKER_LANGUAGE C)
