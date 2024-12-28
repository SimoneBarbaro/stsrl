#!/bin/bash
cd ./sts_lightspeed || exit
cmake -G Ninja -S$PWD -B$PWD/build || exit
cd build || exit
ninja || exit
mv slaythespire*.so ../../
cd ../
rm -r build
cd ../
