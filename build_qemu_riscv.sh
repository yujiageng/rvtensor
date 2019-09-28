#!/bin/bash

##### KENDRYTE
if [ -d "build-qemu-riscv" ];then
    rm -fr build-qemu-riscv
fi
mkdir -p build-qemu-riscv
pushd build-qemu-riscv
cmake -DCMAKE_TOOLCHAIN_FILE=../environments/qemu.toolchain.cmake\
      -DFREEDOM_U_SDK="/opt/freedom-u-sdk" \
      -DBUILD_DIR="../build-qemu-riscv" \
      ..
make -j 8 VERBOSE=1
popd
