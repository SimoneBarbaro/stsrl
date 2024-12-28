#!/bin/bash
cd ./sts_lightspeed
cmake -G Ninja -S$PWD -B$PWD/build
cd build
ninja
mv slaythespire*.so ../../
cd ../
rm -r build
cd ../
