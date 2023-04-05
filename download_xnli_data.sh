echo Downloading XNLI
wget -P data/ https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip
unzip data/XNLI-1.0.zip -d data/
rm -rf data/__MACOSX
rm data/XNLI-1.0.zip