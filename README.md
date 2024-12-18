# 2024_ASOC_GNN_COVID19_Forecast

## Manual Installation (macOS M1):
```bash
conda install pkg-config libuv
conda install cmake ninja
conda install -y clang_osx-arm64 clangxx_osx-arm64 gfortran_osx-arm64
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
# or (alternative with pip)
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124/torch/ --no-cache-dir --verbose
pip install torch-geometric==2.4.0 torch-geometric-temporal --no-cache-dir --verbose
pip install pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cu124.html --no-cache-dir --verbose

conda install xlsx2csv pypickle polars
# or (alternative with pip) 
pip install xlsx2csv pypickle polars
```

## Using a conda environment (recommended):
```bash
conda env create -f environment_mac_m1.yml
or
conda env create -f environment_windows_11.yml
```

## Using requirements.txt:
```bash
pip install -r requirements.txt
```



