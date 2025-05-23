 #!/bin/bash
#SBATCH -J JOB
#SBATCH -p ty_xd
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=dcu:1
#SBATCH --mem=90G

module list

module purge
module load compiler/dtk/24.04

module list

make clean
make

#Preliminary round
#./conv2dfp16demo 64  256  14 14 256  3 3 1 1 1 1 
# ./conv2dfp16demo 256 192  14 14 192  3 3 1 1 1 1 
# ./conv2dfp16demo 16  256  26 26 512  3 3 1 1 1 1 
# ./conv2dfp16demo 32  256  14 14 256  3 3 1 1 1 1 
# ./conv2dfp16demo 2   1280 16 16 1280 3 3 1 1 1 1 
#./conv2dfp16demo 2   960  64 64 32   3 3 1 1 1 1 

# ./conv2dfp16demo 16   64    32 32 64    3 3 1 1 1 1 

./conv2dfp16demo 16 128 64 64 27 3 3 1 1 1 1 
./conv2dfp16demo 16 256 32 32 256 3 3 1 1 1 1 
./conv2dfp16demo 16 64 128 128 64 3 3 1 1 1 1 
./conv2dfp16demo 2 1920 32 32 640 3 3 1 1 1 1 
./conv2dfp16demo 2 640 64 64 640 3 3 1 1 1 1 
./conv2dfp16demo 2 320 64 64 4 3 3 1 1 1 1 

# MIOpenDriver conv -n 256 -c 192  -H 14 -W 14 -k 192  -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -F 1 -t 1
# MIOpenDriver conv -n 16  -c 256  -H 26 -W 26 -k 512  -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -F 1 -t 1
# MIOpenDriver conv -n 32  -c 256  -H 14 -W 14 -k 256  -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -F 1 -t 1 
# MIOpenDriver conv -n 2   -c 1280 -H 16 -W 16 -k 1280 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -F 1 -t 1
# MIOpenDriver conv -n 2   -c 960  -H 64 -W 64 -k 32   -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -F 1 -t 1
