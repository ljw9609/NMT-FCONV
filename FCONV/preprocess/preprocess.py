#!/usr/bin/env python3 -u
import sys
import io

import prepare


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN_ENZH = {
    "url": "http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz",
    "source": "training-parallel-nc-v13/news-commentary-v13.zh-en.en",
    "target": "training-parallel-nc-v13/news-commentary-v13.zh-en.zh",
    "data_source": "train.en",
    "data_target": "train.zh",
}

VALID_ENZH = {
    "url": "http://data.statmt.org/wmt18/translation-task/dev.tgz",
    "source": "dev/newsdev2017-zhen-ref.en.sgm",
    "target": "dev/newsdev2017-zhen-src.zh.sgm",
    "data_source": "valid.en",
    "data_target": "valid.zh",
}

TEST_ZHEN = {
    "url": "http://data.statmt.org/wmt18/translation-task/test.tgz",
    "source": "test/newstest2018-zhen-ref.en.sgm",
    "target": "test/newstest2018-zhen-src.zh.sgm",
    "data_source": "test.en",
    "data_target": "test.zh",
}

DATA_DIR = "data/wmt18_en_zh/"
TMP_DIR = "tmp/wmt18_en_zh/"

if __name__ == '__main__':
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True, line_buffering=True)

    for ds in [TRAIN_ENZH, VALID_ENZH, TEST_ZHEN]:
        prepare.prepare_dataset(DATA_DIR, TMP_DIR, ds)
