#!/bin/bash
# Installed into the cloned env as etc/conda/activate.d/zz_cuda_headers.sh
#
# Triton / torch.inductor compile a tiny launcher (cuda_utils.c) with the SYSTEM
# gcc, which needs the CUDA driver-API header cuda.h. The conda env and Triton's
# bundled include dir do NOT ship cuda.h, so torch.compile dies with
#   "cuda.h: No such file or directory"
# Point gcc at the system CUDA toolkit's include dir via CPATH (gcc searches it
# automatically). Pick the highest cuda-13.* available so a minor bump still works.
_CUDA_INC=""
for d in /apps/cuda/cuda-13.*/targets/x86_64-linux/include; do
  [ -f "$d/cuda.h" ] && _CUDA_INC="$d"
done
if [ -n "$_CUDA_INC" ]; then
  export CPATH="${_CUDA_INC}${CPATH:+:$CPATH}"
fi
unset _CUDA_INC d
