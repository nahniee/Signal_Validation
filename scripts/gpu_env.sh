# Source from the project root before running CLAM scripts:  . scripts/gpu_env.sh
# Exposes the pip-installed CUDA libraries to TensorFlow and keeps the JIT-compiled
# kernels cached (the TF wheel ships no prebuilt SASS for Blackwell / compute 12.0).
_NV="$(.venv/bin/python -c 'import nvidia, os; print(nvidia.__path__[0])')"
export LD_LIBRARY_PATH="$(ls -d "$_NV"/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export CUDA_CACHE_MAXSIZE=4294967296
export TF_CPP_MIN_LOG_LEVEL=1
