import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils.vocab import load_asmfile, init_tokenizer, generate_embeddings, check_vocab
from utils.model import load_weights_uniasm
from utils.misc import cosine_similarity

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

    n = 1
    if len(sys.argv) < 3:
        print("Usage:")
        print("  {} input_1.asm input_2.asm".format(sys.argv[0]))
        exit()

    inputasm1 = 'test1.asm'
    if len(sys.argv) > n:
        inputasm1 = sys.argv[n]
        n+=1

    inputasm2 = 'test2.asm'
    if len(sys.argv) > n:
        inputasm2 = sys.argv[n]
        n+=1

    modelfile = './data/uniasm_base.h5'
    vocabfile = './data/vocab_base.txt'

    # load asm and create vocabulary
    funcs = []
    funcs += load_asmfile(inputasm1)
    funcs += load_asmfile(inputasm2)
    if len(funcs) != 2:
        print('check asm files!')
        exit()
    check_vocab(vocabfile, funcs)
    
    # load the model
    model = load_weights_uniasm(modelfile, config_json)    
    tokenizer = init_tokenizer(vocabfile)

    # generate the embeddings for function
    generate_embeddings(tokenizer, model, funcs, 'liner', b_num=True)

    # calculate the similarity
    sim_score = cosine_similarity(funcs[0].embedding, funcs[1].embedding)
    print(sim_score)

        
