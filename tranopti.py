from datetime import datetime
import json
import numpy as np
import os
import pickle
from protes import protes
import subprocess
import sys
import time
from time import perf_counter as tpc
import torch
from transformers import GPT2LMHeadModel
from transformers import GPT2TokenizerFast


os.environ['TOKENIZERS_PARALLELISM'] = 'false'


LEVELS = [1, 2, 3]
ENGINES = ['deepl', 'google', 'yandex']
SYMBOLS_INP = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
SYMBOLS_OUT = 'abcdefghijklmnopqrstuvwxyz'
SYMBOLS_REPL = ['.', ',', ':', ';', '"', "'", 'ʼ']
SYMBOLS_STOP = ['ü', 'ø', 'ö', 'š', 'æ']
LEN_OUT_MIN = 5


def tranopti(engine, d=7, evals=1.E+3, k=50, k_top=5, level=1, seed=0):
    _time = tpc()
    n = len(SYMBOLS_INP)
    print(f'\n\n---> Start for level "{level}" and engine "{engine}"\n')

    score = _score_build()
    results = []

    if level > 1:
        with open(f'result/{engine}/data_{engine}_l{level-1}.pkl', 'rb') as f:
            results_prev = pickle.load(f)
        eng = 'deepl' if engine == 'all' else engine
        phrase_list = [res[eng]['inp'] for res in results_prev[:n]]

    _log(engine, d, evals, k, k_top, results, level)

    def index_to_phrase(i):
        if level == 1:
            return ''.join([SYMBOLS_INP[ii] for ii in i])
        return ' '.join([phrase_list[ii] for ii in i])

    def func(I):
        losses = []

        t_tr = tpc()
        iter = int(len(results) / k) + 1
        result = _translate([index_to_phrase(i) for i in I], engine)
        t_tr = (tpc() - t_tr) / len(I)

        t_sc = tpc()
        for num in range(len(result)):
            l_list = []
            for eng in result[num].keys():
                if len(result[num][eng]['out']) == 0:
                    p_inp = result[num][eng]['inp']
                    print(f'WRN : empty translation for "{p_inp}" ({eng})')
                if result[num][eng]['inp'] != result[num][eng]['inp_mod']:
                    result[num][eng]['s_inp'] = -1
                    result[num][eng]['s_out'] = -1
                else:
                    result[num][eng]['s_inp'] = score(result[num][eng]['inp'])
                    result[num][eng]['s_out'] = score(result[num][eng]['out'])
                result[num][eng]['s_rev'] = score(result[num][eng]['rev_out'])
                l = _loss(result[num][eng], level)
                result[num][eng]['l'] = l
                l_list.append(l)
            losses.append(np.max(l_list))
        t_sc = (tpc() - t_sc) / len(I)

        results.extend(result)

        t_all = tpc() - _time
        _log(engine, d, evals, k, k_top, results, level, t_tr, t_sc)

        with open(f'result/{engine}/data_{engine}_l{level}.pkl', 'wb') as f:
            pickle.dump(_sort(results), f)

        return losses

    i = protes(func, [n]*d, evals, k=k, k_top=k_top, seed=seed, log=True)[0]
    i = np.array(i) # From jax to numpy
    p = index_to_phrase(i)

    print(f'\n\n---> DONE for level {level}')
    print(f'Total time (sec) : {tpc() - _time}')
    print(f'Found optimum    : {p}\n')


def _log(engine, d, evals, k, k_top, results, level, t_tr=0., t_sc=0.):
    date = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    info = f'd={d:-3d}, evals={evals:-7.1e}, k={k:-3d}, k_top={k_top:-3d}'
    text = f'[{date}] {info}'
    text += '\n' + '=' * 21 + ' ' + '-' * len(info) + '\n\n'

    for num, result in enumerate(_sort(results), 1):
        l = np.max([result[eng]['l'] for eng in result.keys()])
        text += f'# {num:-5d} : loss = {l:-8.1e}\n'

        for eng, res in result.items():
            p_inp = res['inp'].replace('\n', '')
            p_out = res['out'].replace('\n', '')
            p_rev = res['rev_out'].replace('\n', '')

            if engine == 'all':
                text += '   >>> ' + eng + ' ' * max(0, 10-len(eng)) + ' :\n'

            text += f'   inp  : [s = {res["s_inp"]:-8.1e}] {p_inp}\n'
            text += f'   out  : [s = {res["s_out"]:-8.1e}] {p_out}\n'
            text += f'   rev  : [s = {res["s_rev"]:-8.1e}] {p_rev}\n'

            if res['inp'] != res['inp_mod']:
                text += f'   WRN  : inp != "{res["inp_mod"]}"\n'
            if res['out'] != res['rev_inp_mod']:
                text += f'   WRN  : rev != "{res["rev_inp_mod"]}"\n'

        text += '\n'

    if t_tr > 0 or t_sc > 0:
        text += f'\n\n' + '=' * 50 + '\n'
        text += f'Time / phrase for translate  (sec) : {t_tr:-12.4f}\n'
        text += f'Time / phrase for score      (sec) : {t_sc:-12.4f}\n'

    with open(f'result/{engine}/log_{engine}_l{level}.txt', 'w') as f:
        f.write(text + '\n')


def _loss(res, level=1, thr=1000.):
    p_inp = res['inp']
    p_out = res['out']
    loss = 0.

    # Subtract the score of the input only for the first level (symbols)
    if level == 1:
        loss -= res['s_inp']

    # Add score of the output
    loss += res['s_out']

    # Penalty for short outputs:
    if len(p_out) < LEN_OUT_MIN:
        loss += thr

    # Penalty for repeating 3 identical letters (definitely not a word):
    for s in SYMBOLS_OUT:
        if s+s+s in p_out:
            loss += thr * 2

    # Penalty for stop symbols (definitely not a word):
    for s in SYMBOLS_STOP:
        if s in p_out:
            loss += thr * 3

    # Penalty for the input with invalid score:
    if res['s_inp'] < 0:
        loss += thr * 10

    # Penalty for the output with invalid score:
    if res['s_inp'] < 0:
        loss += thr * 10

    return loss


def _score_build(device='cpu'):
    model = GPT2LMHeadModel.from_pretrained('gpt2-large').to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')

    def score(text):
        try:
            enc = tokenizer(text, return_tensors='pt')
            enc_ids = enc.input_ids.to(device)
            enc_ids_target = enc_ids.clone()
            enc_len = enc.input_ids.size(1)

            with torch.no_grad():
                outputs = model(enc_ids, labels=enc_ids_target)
                s = outputs.loss * enc_len

            s = s.numpy().item()

        except Exception as e:
            print(f'WRN : can not compute score for "{text}"')
            return -1.

        if np.isnan(s):
            return 0.

        if s < 0:
            print(f'WRN : negative score for "{text}"')
            s = -1.

        return s

    return score


def _sort(results):
    def sort_func(result):
        return np.max([result[eng]['l'] for eng in result.keys()])
    return sorted(results, key=sort_func)


def _translate(phrases, engine):
    """Translation of the provided list of texts.

    This function should translate the provided list "phrases" of words /
    phrases / texts using a translator "engine" (it may be "deepl", "google",
    "yandex" or "all") and return a list of the translations (including backward
    translations, i.e. "inp -> out -> rev_out", where "inp" is a Russian text,
    "out" is its translation into English and "rev_out" is the backward
    translation of the "out" into Russian). A demonstration of the output
    ("result") format is given below in the body of the function.

    Note that we do not provide our code for requests to the translators, since,
    on the one hand, it is private, and on the other hand, it can be easily
    implemented independently within the framework of the corresponding online
    translators.

    """
    result = []
    for i in range(len(phrases)):
        res = {}
        for eng in (ENGINES if engine == 'all' else [engine]):
            res[eng] = {
                'inp': phrases[i],
                'inp_mod': 'it is equal for phrases[i] if translator did not modify the input sequence',
                'out': 'translation of the phrases[i]',
                'rev_inp': 'input for backward translation (= out)',
                'rev_inp_mod': 'it is equal for rev_inp if translator did not modify the sequence for reverse translation',
                'rev_out': 'translation of the "out" into base language',
            }
        result.append(res)

    return result


if __name__ == '__main__':
    np.random.seed(42)

    level = int(sys.argv[1] if len(sys.argv) > 1 else 1)
    assert level in LEVELS

    engine = sys.argv[2] if len(sys.argv) > 2 else 'all'
    assert engine in (['all'] + ENGINES)

    os.makedirs(f'result', exist_ok=True)
    os.makedirs(f'result/{engine}', exist_ok=True)

    for l in range(level, LEVELS[-1]+1):
        tranopti(engine, level=l)
