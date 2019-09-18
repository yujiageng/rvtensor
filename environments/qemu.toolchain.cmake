# Standard settings
set (CMAKE_SYSTEM_NAME Linux)
# set (CMAKE_SYSTEM_VERSION 1)
set (RISCV True)

set(TOOLCHAIN_DIR "${FREEDOM_U_SDK}/work/buildroot_initramfs/host")
set(_CMAKE_TOOLCHAIN_PREFIX riscv64-sifive-linux-gnu)
set(SYSROOT_DIR "${FREEDOM_U_SDK}/work/buildroot_initramfs/host/${_CMAKE_TOOLCHAIN_PREFIX}/sysroot")

set(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/bin/${_CMAKE_TOOLCHAIN_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/bin/${_CMAKE_TOOLCHAIN_PREFIX}-g++)

set(CMAKE_FIND_ROOT_PATH ${TOOLCHAIN_DIR}
    ${TOOLCHAIN_DIR}/${_CMAKE_TOOLCHAIN_PREFIX}/include/c++/8.3.0
    ${TOOLCHAIN_DIR}/${_CMAKE_TOOLCHAIN_PREFIX}/lib
    ${SYSROOT_DIR}
    ${SYSROOT_DIR}/usr
    ${SYSROOT_DIR}/usr/lib
    ${SYSROOT_DIR}/usr/include)

set(CMAKE_AR  ${TOOLCHAIN_DIR}/bin/${_CMAKE_TOOLCHAIN_PREFIX}-ar CACHE FILEPATH "Archiver")
set(CMAKE_LD  ${TOOLCHAIN_DIR}/bin/${_CMAKE_TOOLCHAIN_PREFIX}-ld CACHE string "linker")
set(CMAKE_NM  ${TOOLCHAIN_DIR}/bin/${_CMAKE_TOOLCHAIN_PREFIX}-nm CACHE string "nm")
set(CMAKE_STRIP ${TOOLCHAIN_DIR}/bin/${_CMAKE_TOOLCHAIN_PREFIX}-strip CACHE string "strip")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NERVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

link_directories(${TOOLCHAIN_DIR}/bin)
link_directories(${SYSROOT_DIR}/usr/include)

add_compile_options(-ffunction-sections)
add_compile_options(-fdata-sections)
add_compile_options(-mcmodel=medany)
add_compile_options(-march=rv64imafdc)
add_compile_options(-mabi=lp64d)
add_compile_options(-fno-common)
#add_compile_options(-T "${SDK_DIR}/lds/kendryte.ld")
#set(CMAKE_CXX_FLAGS "-O0 -std=gnu++11 ${CMAKE_CXX_FLAGS}")
#set(CMAKE_C_FLAGS "-std=gnu11 ${CMAKE_C_FLAGS}")

