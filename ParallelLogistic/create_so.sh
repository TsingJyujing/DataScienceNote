clear
gcc parallel_logistic_kernel.c -lm -lpthread -fPIC -shared -o liblogistic.so
sudo cp liblogistic.so /lib/liblogistic.so
gcc unit_test_so.c -L -l liblogistic.so -lm -o exce_so.o
./exce_so.o
