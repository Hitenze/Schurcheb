wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz
tar -xzf parmetis-4.0.3.tar.gz
mkdir ./EXTERNAL
mv parmetis-4.0.3 ./EXTERNAL/parmetis
rm parmetis-4.0.3.tar.gz
cd ./EXTERNAL/parmetis;
make config;
make -j;
cd ../..;
make -j;
cd TESTS/;
make -j;
