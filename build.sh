#!/bin/bash

##### RISCV
if [ -d "build-qemu-riscv" ];then
    rm -fr build-qemu-riscv
fi
mkdir -p build-qemu-riscv
pushd build-qemu-riscv
cmake -DCMAKE_TOOLCHAIN_FILE=../environments/qemu.toolchain.cmake\
      -DFREEDOM_U_SDK="/opt/freedom-u-sdk" \
      -DBUILD_DIR="../build-qemu-riscv" \
      -DRISCV=ON \
      -DX86=OFF \
      ..
make -j 8 VERBOSE=1
popd


##### X86
# if [ -d "build-x86-linux" ];then
#     rm -fr build-x86-linux
# fi
# mkdir -p build-x86-linux
# pushd build-x86-linux
# cmake -DBUILD_DIR="../build-x86-linux" \
#       -DRISCV=OFF \
#       -DX86=ON \
#       ..
# make -j 8 VERBOSE=1
# popd


