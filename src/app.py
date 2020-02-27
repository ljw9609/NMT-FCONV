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


@app.route('/translate', methods=['POST'])
def translate_sentence():
    params = request.json
    s_lang = params.get('s_lang')
    s_text = params.get('s_text')

    if s_lang == 'en':
        t_text = en2zh.translate(s_text)
    elif s_lang == 'zh':
        t_text = zh2en.translate(s_text)
    else:
        t_text = 'Wrong source language!'
    json_obj = {'s_text': s_text,
                't_text': t_text}
    response = make_response(jsonify(json_obj), 200)
    return response
