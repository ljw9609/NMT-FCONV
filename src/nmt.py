from fairseq.models.lightconv import LightConvModel
from fairseq.models.fconv import FConvModel

# PATH = '/Users/jiawei.lyu/Documents/20winter/cs496/NMT-FCONV/FCONV/wmt17.zh-en.lightconv-glu'
#
# zh2en = LightConvModel.from_pretrained(
#     PATH,
#     checkpoint_file='model.pt',
#     data_name_or_path=PATH,
#     tokenizer='moses',
#     bpe='subword_nmt',
#     bpe_codes=PATH + '/zh.code'
# )
# print(zh2en.translate('你好 世界!'))
#
# print(zh2en.translate('首先，也许是最重要的，1989年的革命和随后的苏联解体结束了全球的两极化'))
#
# print(zh2en.translate('尽管房利美和房地美的确存在一些丑闻，但它们在引发金融危机中扮演的角色微不足道：绝大部分的不良贷款来自私人贷款发放者。'))
#
# print(zh2en.translate('13秒的那次是他自从四年前受伤以来跑出的最快成绩。'))

#print(zh2en.translate('赵立坚在推特上严厉批评了西方22国对中国新疆政策的抹黑，引起美国前驻联合国大使苏珊·赖斯的激烈回应，她在推特上要求中国驻美大使崔天凯将赵立坚送回国。有趣的是，赵立坚当时担任的是驻巴基斯坦使馆公使衔参赞，并非赖斯以为的驻美外交官。赵立坚据实回怼，称自己常驻伊斯兰堡，只不过10年前曾在华盛顿居住而已，直言赖斯才是“惊人地无知”。此次交锋引发了全世界的围观，BBC指出，中国外交官的语言风格更加直接强硬。'))

# PATH = '/Users/jiawei.lyu/Documents/20winter/cs496/NMT-FCONV/FCONV/'
#
# DATA = '/Users/jiawei.lyu/Documents/20winter/cs496/NMT-FCONV/FCONV/data-bin/wmt18_en_zh'
#
# BPE = '/Users/jiawei.lyu/Documents/20winter/cs496/NMT-FCONV/FCONV/data/wmt18_en_zh/codes.32000.bpe.en'
#
# en2zh = LightConvModel.from_pretrained(
#     PATH,
#     checkpoint_file='checkpoint_best.pt',
#     data_name_or_path=DATA,
#     tokenizer='moses',
#     bpe='subword_nmt',
#     bpe_codes=BPE
# )
# print(en2zh.translate('Hello world'))

# PATH = '/Users/jiawei.lyu/Documents/20winter/cs496/NMT-FCONV/FCONV'
# DATA = '/Users/jiawei.lyu/Documents/20winter/cs496/NMT-FCONV/FCONV/data-bin/wmt18_en_zh'
# BPE = '/Users/jiawei.lyu/Documents/20winter/cs496/NMT-FCONV/FCONV/data/wmt18_en_zh/codes.32000.bpe.en'
#
# en2zh = FConvModel.from_pretrained(
#     PATH,
#     checkpoint_file='checkpoint_best.pt',
#     data_name_or_path=DATA,
#     tokenizer='moses',
#     bpe='subword_nmt',
#     bpe_codes=BPE,
#     source_lang='en',
#     target_lang='zh',
#     memory_efficient_fp16=False
# )
#
# print(en2zh.translate('slowly but not without struggle'))

PATH = '/Users/jiawei.lyu/Documents/20winter/cs496/NMT-FCONV/FCONV/model'

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

TEST = '/Users/jiawei.lyu/Documents/20winter/cs496/NMT-FCONV/FCONV/data'

print("\n----------")

# with open(TEST + '/valid.en') as f1:
#     inputs = [next(f1) for i in range(5)]
#
# for input in inputs:
#     output = en2zh.translate(input.lower())
#     print(f'Source: {input}Target: {output}\n--------------')

# with open(TEST + '/valid.zh') as f2:
#     inputs2 = [next(f2) for i in range(5)]
#
# for input in inputs2:
#     output = zh2en.translate(input)
#     print(f'Source: {input}Target: {output}\n--------------')

# print(en2zh.translate('Further development of the central cell mainly involved changes in the orientation of the polar nuclei and the distribution of the cytoplasm.'))
#
# print(zh2en.translate('了解员工真实的内心需求，促进员工的自我激励，加强内在因素的激励作用。'))

src = '"In our view, we think a Sanders nomination would tilt the election more toward Trump, to the point where the ratings would reflect him as something of a favorite," write Kyle Kondik and J. Miles Coleman of the map.\n"However, we would not put Trump over 270 electoral votes in our ratings, at least not initially and based on the information we have now."'

# print(f'Source: {src}\nTarget: {en2zh.translate(src.lower())}\n')


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
        return translator.translate(_sent_.lower())

    paras = src.split('\n')
    outputs = []
    for para in paras:
        if len(para) == 0 or para in ['', ' ', '\t']:
            continue
        outputs.append(translate_paragraph(para))
    return '\n'.join(outputs)
#
#
# print(f'Source: {src}\nTarget: {translate_passage(src, en2zh)}\n')

src = 'Dithering is a technique that blends your colors together, making them look smoother, or just creating interesting textures.'
print(en2zh.translate(src.lower()))
print(translate_passage(src, en2zh))