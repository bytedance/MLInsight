#LD_PRELOAD=`realpath ../build/Runtime/libmlinsight.so` python3 ./oom_insufficient.py
CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=`realpath ../build/Runtime/libmlinsight.so` python3 ./leak.py
