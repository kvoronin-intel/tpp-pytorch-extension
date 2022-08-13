#!/bin/bash

set -e
HERE=$(cd "$(dirname "$0")" && pwd -P)
CONDA_INSTALL_DIR=`realpath ./miniconda3`
ENV_NAME=pt1120

while (( "$#" )); do
  case "$1" in
    -n)
      ENV_NAME=$2
      shift 2
      ;;
    -p)
      CONDA_INSTALL_DIR=$2
      CONDA_INSTALL_DIR=`realpath $CONDA_INSTALL_DIR`
      shift 2
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      echo "Error: Unsupported argument $1" >&2
      exit 1
      ;;
  esac
done

if ! test -f Miniconda3-latest-Linux-x86_64.sh ; then 
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi
if ! test -d ${CONDA_INSTALL_DIR} ; then 
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p ${CONDA_INSTALL_DIR}
${CONDA_INSTALL_DIR}/bin/conda create -y -n ${ENV_NAME} python=3.8
source ${CONDA_INSTALL_DIR}/bin/activate ${ENV_NAME}
conda install -y pytorch==1.12.0 torchvision torchaudio cpuonly intel-openmp gperftools ninja setuptools tqdm future cmake numpy pyyaml scikit-learn pydot -c pytorch -c intel -c conda-forge 

# for bert
conda install -y h5py onnx tensorboardx -c anaconda -c conda-forge

# for unbuffer
#conda install -y -c eumetsat expect

# for development (code formatting)
conda install -y black=22.6.0 clang-format=5.0.1 -c sarcasm -c conda-forge

fi

echo "Writing an env.sh file..."
cat <<EOF > env.sh
#!/bin/bash

source ${CONDA_INSTALL_DIR}/bin/activate ${ENV_NAME}
torch_ccl_path=\$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))" 2> /dev/null)
if test -f \$torch_ccl_path/env/setvars.sh ; then
  source \$torch_ccl_path/env/setvars.sh
fi

NUM_THREADS=\$(lscpu | grep "Core(s) per socket" | awk '{print \$NF}')
export OMP_NUM_THREADS=\${NUM_THREADS}
export KMP_AFFINITY=compact,1,granularity=fine
export KMP_BLOCKTIME=1
export LD_PRELOAD=\${CONDA_PREFIX}/lib/libtcmalloc.so:\${CONDA_PREFIX}/lib/libiomp5.so
EOF
chmod +x env.sh
echo "env.sh file created..."

