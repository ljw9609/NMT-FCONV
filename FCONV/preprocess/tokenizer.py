import os
import jieba
import nltk

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

jieba.initialize()


def _preprocess_sgm(line, is_sgm):
    """Preprocessing to strip tags in SGM files."""
    if not is_sgm:
        return line
    # In SGM files, remove <srcset ...>, <p>, <doc ...> lines.
    if line.startswith("<srcset") or line.startswith("</srcset"):
        return ""
    if line.startswith("<refset") or line.startswith("</refset"):
        return ""
    if line.startswith("<doc") or line.startswith("</doc"):
        return ""
    if line.startswith("<p>") or line.startswith("</p>"):
        return ""
    # Strip <seg> tags.
    line = line.strip()
    if line.startswith("<seg") and line.endswith("</seg>"):
        i = line.index(">")
        return line[i + 1:-6]  # Strip first <seg ...> and last </seg>.


def tokenize(line, is_sgm=False, is_zh=False, lower_case=True, delim=' '):
    # strip sgm tags if any
    _line = _preprocess_sgm(line, is_sgm)
    # replace non-breaking whitespace
    _line = _line.replace("\xa0", " ").strip()
    # tokenize
    _tok = jieba.cut(_line.rstrip('\r\n')) if is_zh else nltk.word_tokenize(_line)
    _tokenized = delim.join(_tok)
    # lowercase. ignore if chinese.
    _tokenized = _tokenized.lower() if lower_case and not is_zh else _tokenized
    return _tokenized


def tokenize_file(filepath, lower_case=True, delim=' '):
    filename = os.path.basename(filepath)
    is_sgm = filename.endswith(".sgm")
    is_zh = filename.endswith(".zh") or filename.endswith(".zh.sgm")

    tokenized = ''
    f = open(filepath, 'rb')
    for i, line in enumerate(f):
        line = line.decode('utf-8')  # decode
        if i % 2000 == 0:
            _tokenizer_name = "jieba" if is_zh else "nltk.word_tokenize"
            logger.info("     [%d] %s: %s" % (i, _tokenizer_name, line))

        # tokenize
        _tokenized = tokenize(line, is_sgm, is_zh, lower_case, delim)
        # if len(_tokenized) < 2:
        #     logger.info("     [%d] (blank): @@%s >> %s@@" % (i, _tokenized, line))

        # append
        tokenized += "%s\n" % _tokenized
    f.close()
    return tokenized
