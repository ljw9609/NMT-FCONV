#wget --no-check-certificate https://dl.fbaipublicfiles.com/fairseq/models/dynamicconv/wmt17.zh-en.lightconv-glu.tar.gz -P ./FCONV
#tar -xzvf ./FCONV/wmt17.zh-en.lightconv-glu.tar.gz
python -m nltk.downloader all

gunicorn --config ./conf/gunicorn_config.py src:app
