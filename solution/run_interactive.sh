srun --nodes=1 \
--time=00:20:00 \
--partition=gpu \
--qos=short \
--account=tc062-staff \
--gres=gpu:4 \
--pty /usr/bin/bash \
--login