#!/bin/bash

NAME=rvtensor

##### package linux x86 resnet20
LINUXX86NAME=${NAME}-linux-x86
rm -fr $LINUXX86NAME
mkdir -p $LINUXX86NAME
cp data/cifar_resnet/resnet20.h5 $LINUXX86NAME/
cp data/cifar-10-batches-bin/test_batch.bin $LINUXX86NAME/
cp third_party/hdf5-1.10.5-x86/lib/libhdf5.so.* $LINUXX86NAME/
cp build-x86-linux/examples/resnet20/qemu_resnet20 $LINUXX86NAME/
rm -fr $LINUXX86NAME.zip
zip -9 -r $LINUXX86NAME.zip $LINUXX86NAME

##### package SiFive lib benchmark and examples
#SIFIVENAME=${NAME}-qemu-riscv
#rm -fr $SIFIVENAME
#cp -a build-qemu-riscv/examples/resnet20/qemu-resnet20 $LINUXX86NAME/
#cp -a data/cifar_resnet/resnet20.h5 $LINUXX86NAME/
#cp -a data/cifar-10-batches-bin/test_batch.bin $LINUXX86NAME/
#cp -a third_party/hdf5-1.10.5-release/lib/libhdf5.so.* $LINUXX86NAME/
#rm -fr $SIFIVENAME.zip
#zip -9 -r $SIFIVENAME.zip $SIFIVENAME
