import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.vocab import load_asmfile, init_tokenizer, generate_embeddings, check_vocab
from utils.model import load_weights_uniasm

import tensorflow as tf

print("Tensorflow version " + tf.__version__)

#    '''
#    Embedding a input ASM function
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

    n = 1
    if len(sys.argv) < 2:
        print("Usage:")
        print("  {} input.asm".format(sys.argv[0]))
        exit()

    inputasm = 'test.asm'
    if len(sys.argv) > n:
        inputasm = sys.argv[n]
        n+=1

    modelfile = './data/uniasm_base.h5'
    vocabfile = './data/vocab_base.txt'

    # load asm and create vocabulary
    funcs = load_asmfile(inputasm)
    if len(funcs) != 1:
        print('check asm file!')
        exit()
    check_vocab(vocabfile, funcs)
    
    # load the model
    model = load_weights_uniasm(modelfile, config_json)
    tokenizer = init_tokenizer(vocabfile)

    # generate the embeddings for function
    generate_embeddings(tokenizer, model, funcs, 'liner', b_num=True)

    # show the embedding vector
    print(funcs[0].embedding)

        
