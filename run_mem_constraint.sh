# Usage: bash scripts/run_mem_constraint.sh MEM CMD
sudo bash /home/huangkz/clean_mem.sh # Clean memory
echo $$  # Print PID
sudo echo $$  >  /sys/fs/cgroup/limit/cgroup.procs # Add PID to cgroup
cat /sys/fs/cgroup/limit/cgroup.procs # Print PID
echo $1  # Memory limit
echo $1 > /sys/fs/cgroup/limit/memory.max # Set memory limit
cat /sys/fs/cgroup/limit/memory.max # Print memory limit
echo "${@:2}" # Print command
eval "${@:2}" # Run command
