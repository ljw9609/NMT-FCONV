from flask import request, jsonify, make_response
from src import app
from fairseq.models.fconv import FConvModel

import os

PATH = os.path.join(os.path.dirname(__file__), '../FCONV/model')

zh2en = FConvModel.from_pretrained(
    PATH,
    checkpoint_file='model_zh2en.pt',
    data_name_or_path=PATH,
    tokenizer='moses',
    bpe='subword_nmt',
    bpe_codes=PATH + '/zh.code'
)

en2zh = FConvModel.from_pretrained(
    PATH,
    checkpoint_file='model_en2zh.pt',
    data_name_or_path=PATH,
    tokenizer='moses',
    bpe='subword_nmt',
    bpe_codes=PATH + '/en.code'
)


def translate_passage(src, translator):
    def translate_paragraph(_para_):
        sents = _para_.split('.')
        tgt_sents = []
        for sent in sents:
            if len(sent) == 0 or sent in ['', ' ', '\t']:
                continue
            tgt_sents.append(translate_sentence(sent))
        return ''.join(tgt_sents)

    def translate_sentence(_sent_):
        return translator.translate(_sent_.lower()).capitalize()

    paras = src.split('\n')
    outputs = []
    for para in paras:
        if len(para) == 0 or para in ['', ' ', '\t']:
            continue
        outputs.append(translate_paragraph(para))
    return '\n'.join(outputs)


@app.route('/translate', methods=['POST'])
def translate():
    params = request.json
    s_lang = params.get('s_lang').lower()
    s_text = params.get('s_text').lower()

    if s_lang == 'en':
        # t_text = en2zh.translate(s_text)
        t_text = translate_passage(s_text, en2zh)
    elif s_lang == 'zh':
        # t_text = zh2en.translate(s_text)
        t_text = translate_passage(s_text, zh2en)
    else:
        t_text = 'Wrong source language!'
    json_obj = {'s_text': s_text,
                't_text': t_text}
    response = make_response(jsonify(json_obj), 200)
    return response
