import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils.vocab import load_asmfiles, init_tokenizer, generate_embeddings, check_vocab
from utils.model import load_weights_uniasm
from utils.misc import evaluate_performance, filter_similar_pairs

import tensorflow as tf
print("Tensorflow version " + tf.__version__)


#    '''
#    Embedding two ASM functions and calculate the similarity score.
#    '''
if __name__=='__main__':
    config_json = """
{
    "hidden_act": "gelu",
    "hidden_size": 516,
    "intermediate_size": 3072,
    "max_position_embeddings": 256,
    "num_hidden_layers": 4,
    "vocab_size": 21000,
    "num_attention_heads": 12
}
"""

    dir1 = 'out-elfedit-gcc-o0'
    dir2 = 'out-elfedit-gcc-o1'

    modelfile = './data/uniasm_base.h5'
    vocabfile = './data/vocab_base.txt'

    # load asm and create vocabulary
    funcs1 = load_asmfiles('./testdata/', [dir1])
    funcs2 = load_asmfiles('./testdata/', [dir2])
    check_vocab(vocabfile, funcs1)
    check_vocab(vocabfile, funcs2)

    # similar function pair
    funcs_left, funcs_right = filter_similar_pairs(funcs1, funcs2)
    
    # load the model
    model = load_weights_uniasm(modelfile, config_json)    
    tokenizer = init_tokenizer(vocabfile)

    # generate the embeddings for function
    generate_embeddings(tokenizer, model, funcs_left, 'liner', b_num=True)
    generate_embeddings(tokenizer, model, funcs_right, 'liner', b_num=True)

    # calculate recall@k
    evaluate_performance(funcs_left, funcs_right, topk=10)

        
