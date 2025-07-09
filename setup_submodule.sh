# Initialize and update all submodules
git submodule update --init --recursive

cd external/cnpy
mkdir build
cd build
cmake ..
make -j 32

cd ../../..
