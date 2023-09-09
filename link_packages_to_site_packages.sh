#!/bin/bash

# 공통 코드를 포함한 디렉터리 경로 설정
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
common_code_dir="$script_dir/kangnam_packages"

# 현재 활성화된 Conda 환경 이름 얻기
# conda_env_name=$(conda env list | grep '^\*' | awk '{print $2}') # 동작 안함
conda_env_name="$CONDA_DEFAULT_ENV"

# which conda
# echo "DEBUG: conda info --envs output:"
# conda info --envs
# echo "DEBUG: conda_env_name = $conda_env_name"

if [ -z "$conda_env_name" ]; then
    echo "Error: No Conda environment is currently activated."
    exit 1
fi

# 현재 활성화된 Conda 환경의 site-packages 디렉터리로 심볼릭 링크 생성
env_site_packages="$CONDA_PREFIX/lib/python$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')/site-packages"

if [ -L "$env_site_packages/kangnam_packages" ] || [ -d "$env_site_packages/kangnam_packages" ]; then
    echo "Error: There's already a symbolic link or directory named 'kangnam_packages' in the site-packages."
    exit 1
fi

ln -s $common_code_dir $env_site_packages
if [ $? -ne 0 ]; then
    echo "Error: Failed to create a symbolic link. Check your permissions or other issues."
    exit 1
fi

echo "Symbolic link created in the active Conda environment: $conda_env_name"
