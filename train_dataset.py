import sys
from typing import Counter
from random import random
import threading
import multiprocessing
import tensorflow as tf
from typing import List
import tqdm
from transformers import BertTokenizer
import collections
import sys
from itertools import combinations_with_replacement
from utils.asmfunction import *
from utils.vocab import *

sys.setrecursionlimit(10000)

#dirs = ['binutils', 'coreutils', 'diffutils', 'findutils', 'curl', 'tcpdump', 'gmp']
dirs = ['testdata']
sub_dirs = [
    'out-elfedit-gcc-o0',
    'out-elfedit-gcc-o1',
    #'gcc-o0',
    #'gcc-o1',
    #'gcc-o2',
    #'gcc-o3',
    #'clang-o0',
    #'clang-o1',
    #'clang-o2',
    #'clang-o3',
    ]

# config
MAX_SEQ_LEN = 256

g_func_trancate1 = 0
g_func_trancate2 = 0
g_func_same1 = 0
g_func_same2 = 0
g_func_liner1 = 0
g_func_liner2 = 0

def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature

# type = 'liner': Serialized read instructions
# type = 'xpath': Use the path in CFG
def gen_unilm_ids(func1:Function, func2:Function, type, n_xpath = 1, b_num = False):
    global g_func_trancate1
    global g_func_trancate2
    global g_func_same1
    global g_func_same2
    global g_func_liner1
    global g_func_liner2

    batch_token_ids, batch_segment_ids = [], []
    idss1 = gen_func_sents(TOKENIZER, func1, type, MAX_SEQ_LEN//2 - 2, n_xpath, b_num=b_num)
    b_merge = False
    for ids in idss1:
        if len(ids) < int(MAX_SEQ_LEN * 0.75):
            b_merge = True
            break
    if b_merge and type == 'xpath':
        ids_liner = gen_func_sents(TOKENIZER, func1, 'liner', MAX_SEQ_LEN//2 - 2, b_num=b_num)
        idss1 = ids_liner 
        g_func_liner1+=1

    idss2 = gen_func_sents(TOKENIZER, func2, type, MAX_SEQ_LEN//2 - 2, n_xpath, b_num=b_num)
    b_merge = False
    for ids in idss2:
        if len(ids) < int(MAX_SEQ_LEN * 0.75):
            b_merge = True
            break
    if b_merge and type == 'xpath':
        ids_liner = gen_func_sents(TOKENIZER, func2, 'liner', MAX_SEQ_LEN//2 - 2, b_num=b_num)
        idss2 = ids_liner   
        g_func_liner2+=1
    
    for ids1 in idss1:
        if len(ids1) > MAX_SEQ_LEN//2 - 2:
            raise Exception('bad sent size')
        if len(ids1) == MAX_SEQ_LEN//2 - 2:
            g_func_trancate1+=1        

        for ids2 in idss2:       
            if len(ids2) > MAX_SEQ_LEN//2 - 2:
                raise Exception('bad sent size')
            if len(ids2) == MAX_SEQ_LEN//2 - 2:
                g_func_trancate2+=1

            padsize = MAX_SEQ_LEN - 3 - len(ids1) - len(ids2)

            token_ids = [CLS_ID] + ids1 + [SEP_ID] + ids2 + [SEP_ID] + [PAD_ID] * padsize
            segment_ids = [0] * (len(ids1) + 2) + [1] * (len(ids2) + 1)  + [0] * padsize

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)

            token_ids = [CLS_ID] + ids2 + [SEP_ID] + ids1 + [SEP_ID] + [PAD_ID] * padsize
            segment_ids = [0] * (len(ids2) + 2) + [1] * (len(ids1) + 1)  + [0] * padsize

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)

    return batch_token_ids, batch_segment_ids

def gen_features(funcs1:List[Function], funcs2:List[Function], desc, type, n_xpath = 1, b_num=False):
    all_ids = []
    all_seg = []
    for f1 in tqdm.tqdm(funcs1, desc='Generate similar ' + desc):
        for f2 in funcs2:           
            if f1.meta['name'] == f2.meta['name']:
                ids, seg = gen_unilm_ids(f1, f2, type=type, n_xpath=n_xpath, b_num=b_num)
                all_ids += ids
                all_seg += seg
    return all_ids, all_seg

def create_datasets(out_file, all_ids, all_seg):
    print("Write data to {}".format(os.path.abspath(out_file)))
    with tf.io.TFRecordWriter(os.path.abspath(out_file)) as file_writer:
        for i in range(len(all_ids)):
            features = collections.OrderedDict()
            features["id"] = create_int_feature(all_ids[i])
            features["seg"] = create_int_feature(all_seg[i])
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            file_writer.write(tf_example.SerializeToString())

threadlock = threading.Lock()
class gen_features_thread(threading.Thread):
    def __init__(self, all_ids, all_seg, funcs1:List[Function], funcs2:List[Function], desc, type, n_xpath = 1, b_num=False):
        threading.Thread.__init__(self)
        self.all_ids = all_ids
        self.all_seg = all_seg
        self.funcs1 = funcs1
        self.funcs2 = funcs2
        self.desc = desc
        self.type = type
        self.n_xpath = n_xpath
        self.b_num = b_num

    def run(self):
        ids, seg = gen_features(self.funcs1, self.funcs2, self.desc, self.type, self.n_xpath, self.b_num)
        threadlock.acquire()
        self.all_ids += ids
        self.all_seg += seg
        threadlock.release()

proclock = multiprocessing.Lock()
def gen_features_proc(all_ids, all_seg, funcs1:List[Function], funcs2:List[Function], desc, type, n_xpath = 1, b_num=False):
    ids, seg = gen_features(funcs1, funcs2, desc, type, n_xpath, b_num)
    proclock.acquire()
    all_ids += ids
    all_seg += seg
    proclock.release()

if __name__ == '__main__':

    ASM_PATH = './'
    VOCAB_FILE = './testdata/vocab.txt'

    # load all the asm files
    func_all = []
    funcss = dict()
    for dir in dirs:
        for subdir in sub_dirs:
            print("Processing {}-{}".format(dir, subdir))
            funcs = load_asmfiles(ASM_PATH, [dir, subdir], False)        
            func_all+=funcs
            funcss['{}_{}'.format(dir, subdir)] = funcs

    # create vocab
    functions_summary(func_all)
    VOCABER = create_vocabs(VOCAB_FILE, func_all)
    freqs:Counter = VOCABER.freqs
    for word in freqs.most_common(10):
        print(word)

    TOKENIZER = BertTokenizer(VOCAB_FILE, do_lower_case=False)
    VOCAB_SIZE=len(TOKENIZER.vocab.items())
    MASK_ID = TOKENIZER.vocab["[MASK]"]
    CLS_ID = TOKENIZER.vocab["[CLS]"]
    SEP_ID = TOKENIZER.vocab["[SEP]"]
    PAD_ID = TOKENIZER.vocab["[PAD]"]

    # generate input sequences
    all_ids = []
    all_seg = []
    for t in dirs:
        for c in combinations_with_replacement(sub_dirs, 2):
            if c[0] == c[1]:
                continue
            ids, seg = gen_features(funcss['{}_{}'.format(t, c[0])], funcss['{}_{}'.format(t, c[1])], '{}-{}/{}'.format(t, c[0], c[1]), type='liner', b_num=True)
            all_ids += ids
            all_seg += seg

    print("total count: {}".format(len(all_ids)))
    print("Func trancate: {} {}".format(g_func_trancate1, g_func_trancate2))

    if len(all_ids) != len(all_seg) or len(all_ids) % 2 != 0:
        raise(Exception("Error: size of data may be wrong!")) 
    
    # shuffle the data
    indexs = [i for i in range(round(len(all_ids) / 2))]
    random.shuffle(indexs)

    # we should keep the similar function pair
    rand_ids = []
    rand_seg = []
    for i in range(round(len(all_ids) / 2)):
        rand_ids.append(all_ids[indexs[i] * 2])
        rand_seg.append(all_seg[indexs[i] * 2])
        rand_ids.append(all_ids[indexs[i] * 2 + 1])
        rand_seg.append(all_seg[indexs[i] * 2 + 1])

    # generate the training dataset
    train_size = round(len(rand_ids) * 0.9) # 90% for train
    if train_size % 2 != 0:
        train_size += 1
    valid_size = len(rand_ids) - train_size
    if valid_size % 2 != 0:
        valid_size -= 1
    create_datasets("./testdata/train-{}.tfrec".format(train_size), rand_ids[:train_size], rand_seg[:train_size])
    create_datasets("./testdata/validate-{}.tfrec".format(valid_size), rand_ids[train_size:train_size + valid_size], rand_seg[train_size:train_size + valid_size])
