#!/bin/bash


# ==============================================
#   🧠 EOGS Project — Installer Script
# ==============================================

PROJECT_NAME=EOGS
PYTHON_VERSION=3.8
CUDA_SUPPORTED=(11.8 12.0 12.1 11.6 11.4)
REPO_URL="git@gitlab-student.centralesupelec.fr:test-groupe/EOGS.git"

echo
echo "_____________________________________________"
echo "           ⚙️ EOGS Environment Setup          "
echo "_____________________________________________"
echo

# ---- Detect CUDA version ----
# echo "⭐ Detecting installed CUDA..."
# CUDA_VERSION=$(nvcc --version | grep release | sed 's/.* release //' | sed 's/, .*//')
# CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d. -f1)
# CUDA_MINOR=$(echo ${CUDA_VERSION} | cut -d. -f2)
# if [[ ! " ${CUDA_SUPPORTED[*]} " =~ " ${CUDA_VERSION} " ]]; then
#   echo "❌ Unsupported CUDA ${CUDA_VERSION}. Supported: ${CUDA_SUPPORTED[*]}"
#   exit 1
# else
#   echo "✅ Found supported CUDA ${CUDA_VERSION}"
# fi

CUDA_VERSIon=12.1


# ---- Set libmamba solver for speed ----
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba

# ---- Install PyTorch ----
echo "⭐ Installing PyTorch..."
if [[ "$CUDA_VERSION" == "11.8" ]]; then
    conda install pytorch==2.2.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == "12.1" ]]; then
    conda install pytorch==2.2.0 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
else
    conda install pytorch torchvision torchaudio pytorch-cuda=${CUDA_MAJOR}${CUDA_MINOR} -c pytorch -c nvidia -y
fi
echo "✅ PyTorch installed"
echo



# ---- Install Python dependencies ----
echo "⭐ Installing Python dependencies..."

# install with conda C++ needed dependencies
conda install -c conda-forge libpng libwebp libtiff zlib glm ninja
sudo apt-get install libtiff-dev libpng-dev libwebp-dev libglm-dev

pip install -r requirements.txt

echo "✅ Python dependencies installed"
echo


echo " Building EOGS package..."
pip install src/gaussiansplatting/submodules/diff-gaussian-rasterization
pip install src/gaussiansplatting/submodules/simple-knn

echo "🚀 EOGS environment successfully installed and configured!"

