from typing import List
from utils.asmfunction import *
from transformers import BertTokenizer
import tensorflow as tf
import random

MASK_ID=0
CLS_ID=0
SEP_ID=0 
PAD_ID=0 
UNK_ID=0 

# Building Vocab with ASM files
# word: mov_eax_2
# sentence: push_eax push_ebx mov_eax_2 ret 
from bert_pytorch.dataset import WordVocab
class ASMVocab(WordVocab):
    def __init__(self, funcs:List[Function], max_size=None, min_freq=1):
        sentences = []
        func:Function
        for func in tqdm.tqdm(funcs):
            for bb in func.blocks:     
                sentence = ""           
                for i in bb.insts:  
                    txt = i.text.strip().replace(',', ' ').replace(':', ' ')
                    txt = re.sub(r'[ ]+', '_', txt)
                    i.vocab = txt
                    sentence += txt + " "
                    for n in i.nums:
                        sentence += str(n) + ' '
                sentences.append(sentence.strip())
        super().__init__(sentences, max_size=max_size, min_freq=min_freq)

def create_vocabs(vocabfile, funcs, max_size = None, min_freq = 1):
    print("Save vocab to {}".format(vocabfile))
    vocab = ASMVocab(funcs, max_size, min_freq)
    specials = ['[UNK]', '[CLS]', '[SEP]', '[MASK]', '[PAD]']
    with open(vocabfile, 'w') as f:
        f.writelines([line + '\n' for line in specials])
        f.writelines([line + '\n' for line in vocab.stoi])
    with open(vocabfile + '.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("Vocab size: ", len(vocab) + len(specials))
    return vocab

def check_vocab(vocabfile, funcs:List[Function]):
    #print("Check vocabs in {}".format(vocabfile))
    tokenizer = BertTokenizer(vocabfile, do_lower_case=False)
    unk_count = 0
    for func in funcs:
        for inst in func.insts:
            try:                
                txt = inst.text.strip().replace(',', ' ').replace(':', ' ')
                txt = re.sub(r'[ ]+', '_', txt)
                inst.vocab = txt
                tokenizer.vocab[txt]
            except KeyError as e:
                #print("Unkown vocab: " + e.args[0])
                unk_count += 1

    #print('Unknown vocab count: {}'.format(unk_count))

def calc_bb_path_len(bb:BasicBlock, bbs_path:List[BasicBlock], add = 1, deep = 0):
    # skip the very long path
    if deep > 20 or len(bbs_path) > 20:
        return

    if bb in bbs_path:
        return
        
    if add:
        for i in range(len(bbs_path)):
            if bbs_path[i].n_path_len < len(bbs_path) - i:
                bbs_path[i].n_path_len = len(bbs_path) - i

    bbs_path.append(bb)
    add = 1
    deep += 1
    for sbb in bb.successors:
        calc_bb_path_len(sbb, bbs_path, add, deep)
        add = 0
    bbs_path.pop()
    deep -= 1

def gen_max_path(b:BasicBlock, bbs:set, sent:List[Instruction], max_sent_len, id = 0, n_total = 1):
    if b in bbs:
        return
    bbs.add(b)

    if id >= n_total:
        raise NotImplementedError()

    if len(sent) + len(b.insts) > max_sent_len:
        sent += b.insts[:max_sent_len-len(sent)]
        return

    sent += b.insts

    bbslist:List[BasicBlock] = list(b.successors)
    if len(bbslist) > 0:
        select:List[BasicBlock] = [bbslist[0]]
        select_size = 1
        
        n_same = 0 
        for sub in bbslist[1:]:
            if sub in bbs or sub.n_path_len == 0:
                continue

            if select_size < n_total:
                select.append(sub)
                select_size+=1
                continue

            for i in range(select_size):          
                if sub.n_path_len > select[i].n_path_len:
                    select[i] = sub
                    break
                elif sub.n_path_len == select[i].n_path_len:
                    n_same+=1
                    if id == n_same:
                        select[i] = sub
                        n_same = 0
                        break

        tmpid = id
        if select_size <= id:
            tmpid = select_size - 1
        gen_max_path(select[tmpid], bbs, sent, max_sent_len, id, n_total)
        

def gen_func_sents(tokenizer, func:Function, func_type, max_sent_len, n_xpath = 1, b_num = False, skip_small = False):
    # skip small functions
    if skip_small:
        if len(func.insts) < 10 or len(func.blocks) < 3:
            return []

    if 'liner' in func_type or 'basicblock' in func_type: 
        sents = []        
        instrs = func.insts

        sent = []
        for instr in instrs:
            sent.append(instr.vocab)
            if b_num:
                for n in instr.nums:
                    sent.append(str(n))

            if len(sent) >= max_sent_len:
                sent = sent[:max_sent_len]
                break

        sent = tokenizer.convert_tokens_to_ids(sent)
        sents.append(sent)
        return sents
    elif 'xpath' in func_type:
        sents = []
        for i in range(n_xpath):
            instrs = []
            bbs = set()     
            gen_max_path(func.blocks[0], bbs, instrs, max_sent_len, i, n_xpath)
            
            sent = []
            for instr in instrs:
                sent.append(instr.vocab)
                if b_num:
                    for n in instr.nums:
                        sent.append(str(n))

                if len(sent) >= max_sent_len:
                    sent = sent[:max_sent_len]
                    break

            sent = tokenizer.convert_tokens_to_ids(sent)
            sents.append(sent)
        return sents
    else:
        raise NotImplementedError

def gen_func_sent(tokenizer, func:Function, func_type, max_sent_len, b_num = False, skip_small = False):
    # skip small functions
    if skip_small:
        if len(func.insts) < 10 or len(func.blocks) < 3:
            return []

    instrs = [] # vocab ids list of instructions for current function
    if 'randomwalk' in func_type:
        bb_end = func.blocks[0]
        while True:
            if len(instrs) + len(bb_end.insts) > max_sent_len:
                instrs += bb_end.insts[:max_sent_len-len(instrs)]
                break

            instrs += bb_end.insts
            if len(bb_end.successors) == 0:
                break
            else:
                bb_end = random.choice(list(bb_end.successors))
    elif 'liner' in func_type or 'basicblock' in func_type: 
        sent = []
        for instr in func.insts:
            sent.append(instr.vocab)
            if b_num:
                for n in instr.nums:
                    sent.append(str(n))

            if len(sent) >= max_sent_len:
                sent = sent[:max_sent_len]
                break

        sent = tokenizer.convert_tokens_to_ids(sent)
        return sent
    elif 'xpath' in func_type:   
        bbs = set()     
        gen_max_path(func.blocks[0], bbs, instrs, max_sent_len)

    sent = []
    for instr in instrs:
        sent.append(instr.vocab)
        if b_num:
            for n in instr.nums:
                sent.append(str(n))

        if len(sent) >= max_sent_len:
            sent = sent[:max_sent_len]
            break

    return tokenizer.convert_tokens_to_ids(sent)

def init_tokenizer(vocabfile):
    global MASK_ID,CLS_ID,SEP_ID,PAD_ID,UNK_ID
    tokenizer = BertTokenizer(vocabfile, do_lower_case=False)
    MASK_ID = tokenizer.vocab["[MASK]"]
    CLS_ID = tokenizer.vocab["[CLS]"]
    SEP_ID = tokenizer.vocab["[SEP]"]
    PAD_ID = tokenizer.vocab["[PAD]"]
    UNK_ID = tokenizer.vocab["[UNK]"]
    return tokenizer

def generate_embeddings(tokenizer, model, funcs, func_type, max_seq = 256, b_num = False):
    funcs_emb = []
    ids_batch = []
    mask_batch = []
    for f in funcs:
        ids = gen_func_sent(tokenizer,f,func_type,max_seq - 2, b_num=b_num)
        ids = [CLS_ID] + ids + [SEP_ID]
        mask = [0] * len(ids) + [1] * (max_seq - len(ids))
        ids = ids + [PAD_ID] * (max_seq - len(ids))

        ids_batch.append(ids)
        mask_batch.append(mask)

    n_total = len(ids_batch)
    n_batch = 3
    n_c = n_total // n_batch
    n_r = n_total % n_batch
    
    for i in tqdm.tqdm(range(n_c)):
        bert_out, _ = model((tf.constant(ids_batch[i*n_batch:(i+1)*n_batch]), tf.constant(mask_batch[i*n_batch:(i+1)*n_batch])))
        funcs_emb.extend(bert_out.numpy())
    if n_r:
        bert_out, _ = model((tf.constant(ids_batch[-n_r:]), tf.constant(mask_batch[-n_r:])))
        funcs_emb.extend(bert_out.numpy())

    for i in range(len(funcs)):
        funcs[i].embedding = funcs_emb[i]