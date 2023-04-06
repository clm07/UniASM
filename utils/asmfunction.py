from typing import List
import re
import pickle
import tqdm
import os
from pathlib import Path


class Instruction:
    def __init__(self, text, op, args, nums):
        self.text = text
        self.vocab = ''
        self.op = op
        self.args = args
        self.nums = nums
    def __str__(self):
        return f'{self.op} {", ".join([str(arg) for arg in self.args if str(arg)])}'
    @classmethod
    def load(cls, text):
        text = text.strip() # get rid of BND prefix
        op, _, args = text.strip().partition(' ')
        if args:
            args = [arg.strip() for arg in args.split(',')]
        else:
            args = []
        #args = (args + ['', ''])[:2]
        return cls(text, op, args, [])
    def tokens(self):
        return [self.op] + self.args
    def is_cjmp(self):
        return self.op != 'jmp' and (self.op == 'cjmp' or self.op[0] == 'j')
    def is_jmp(self):
        return self.op == 'jmp'
    def is_jmp_or_cjmp(self):
        return self.op == 'cjmp' or self.op[0] == 'j'
    def is_call(self):
        return self.op == 'call'
    def is_ret(self):
        return self.op == 'ret'

class BasicBlock:
    def __init__(self):
        self.insts:List[Instruction] = []
        self.successors = set()
        self.n_path_len = 0
    def add(self, inst:Instruction):
        self.insts.append(inst)
    def end(self):
        inst = self.insts[-1]
        return inst.is_jmp_or_cjmp() or inst.op == 'ret'
    def back(self):
        return self.insts[-1]

class Function:
    def __init__(self, insts:List[Instruction], blocks:List[BasicBlock], meta, addrs):
        self.insts = insts
        self.blocks = blocks
        self.meta = meta
        self.calls_addrs = addrs
        self.addr = int(meta['offset'], 16)
        self.calls:set[Function] = set()
        self.embedding = []
        self.target = ''
        self.compiler = ''

        self.n_insts = len(insts)
        self.n_insts_args3 = 0 # count of instructions with more than 2 args
        for inst in insts:
            if len(inst.args) > 2:
                self.n_insts_args3 += 1

        self.n_blocks = len(blocks)
        self.n_blocks_list = [ 0 for i in range(0, 1024)]
        self.n_blocks_lg1024 = 0
        for bb in blocks:
            if len(bb.insts) < 1024:
                self.n_blocks_list[len(bb.insts)] += 1
            else:
                self.n_blocks_lg1024 += 1

    @classmethod
    def load(self, lines:str):
        '''
        gcc -S format compatiable
        '''
        label, labels = None, {}
        insts:List[Instruction] = []
        blocks:List[BasicBlock] = []
        meta = {}
        addrs = []
        for line in lines.strip('\n').split('\n'):
            line = line.strip().lower()
            if line == '':
                continue
            
            # meta data
            if line[0] == '.':
                key, _, value = line[1:].strip().partition(' ')
                meta[key] = value
                continue
            
            # LABLE
            if line[-1] == ':':                
                label = line.partition(':')[0]
                continue

            # calls
            if line.startswith('call'):
                call_addr = line.strip().split(' ')[-1]
                if call_addr.startswith('0x') and not call_addr.endswith(']'):
                    try:
                        addr = int(call_addr,16)
                    except:
                        addr = 0
                    if addr:
                        addrs.append(addr)        
            
            # instr normalize 
            num_int16 = []
            num_int10 = []
            line = line.replace(' - ', ' + ')
            line = re.sub(r'[xyz]mm[0-9][0-5]*', 'XMM', line)

            if line.find('[') != -1:
                if line.startswith('call') or 'jmp' in line:
                    line = re.sub(r'0x[0-9a-f]+', 'NUM', line)
                    line = re.sub(r' [0-9]+', ' NUM', line)
                else:
                    num_int16 += re.findall(r'0x[0-9a-f]+', line)
                    #num_int10 += re.findall(r' [0-9]+', line)        

                line = re.sub(r'\[[er]ip .*\]', 'PTR', line)
                line = re.sub(r'\[[er]sp .*\]', 'SSP', line)
                line = re.sub(r'\[[er]bp .*\]', 'SBP', line)
                line = re.sub(r'\[.*\]', 'MEM', line)     

                line = re.sub(r'0x[0-9a-f]+', 'NUM', line)         
                line = re.sub(r' [0-9]+', ' NUM', line)
            else:
                if line.startswith('call') or 'jmp' in line:
                    line = re.sub(r'0x[0-9a-f]+', 'NUM', line)
                    line = re.sub(r' [0-9]+', ' NUM', line)
                else:
                    num_int16 += re.findall(r'0x[0-9a-f]+', line)
                    line = re.sub(r'0x[0-9a-f]+', 'NUM', line)
                    num_int10 += re.findall(r' [0-9]+', line)                
                    line = re.sub(r' [0-9]+', ' NUM', line)                                            

            # load Instruction
            inst = Instruction.load(line)
            for num in num_int16:
                n = int(num, 16)
                if n < 0x1000:
                    inst.nums.append(n)
            for num in num_int10:
                n = int(num)
                if n < 0x1000:
                    inst.nums.append(n)

            insts.append(inst)
            if len(blocks) == 0 or blocks[-1].end():
                blocks.append(BasicBlock())
                # link prev and next block
                if len(blocks) > 1 and blocks[-2].back().is_cjmp():
                    blocks[-2].successors.add(blocks[-1])
            if label:
                labels[label], label = blocks[-1], None
            blocks[-1].add(inst)               

        # link label
        for block in blocks:
            inst = block.insts[-1]
            if inst.is_jmp_or_cjmp() and labels.get(inst.args[0]):
                block.successors.add(labels[inst.args[0]])

        # normalize instruction
        for inst in insts:            
            assert(inst.text.find('[') == -1)
            assert(inst.text.find('0x') == -1)

            # replace LABLES with CONST
            for i, arg in enumerate(inst.args):
                if labels.get(arg):
                    inst.args[i] = 'REL'
                    inst.text = inst.text.replace(arg, 'REL')           
            
        return self(insts, blocks, meta, addrs)
    def tokens(self):
        return [token for inst in self.insts for token in inst.tokens()]

def create_functions(filenames, func_file = None) -> List[Function]:
    # prepare data    
    functions:dict[int,Function] = {}

    if func_file and os.path.exists(func_file):
        print("File already exist: {}".format(func_file))
        return []
    
    if not filenames:
        print('need asm files to load')
        return []

    #print("Loading raw asm files in " + os.path.dirname(filenames[0]))
    for filename in tqdm.tqdm(filenames):
        with open(filename) as f:
            fn = Function.load(f.read())
            functions[fn.addr]=fn

    for addr in functions:
        for call_addr in functions[addr].calls_addrs:
            try:
                functions[addr].calls.add(functions[call_addr])
            except:
                pass

    if func_file:
        #print("Dump functions to {}".format(func_file))
        with open(func_file, "wb") as f:
            pickle.dump([i for i in functions.values()], f)

    return [i for i in functions.values()]

def load_functions(func_file) -> List[Function]:
    #print("Load functions from {}".format(func_file))
    functions:List[Function] = []
    with open(func_file, "rb") as f:
        functions = pickle.load(f)

    return functions

def functions_summary(functions:List[Function]):
    print("      total functions: {} (blocks: {} = 1, {} in [2,4), {} in [4,8), {} in [8,16), {} in [16,32), {} in [32,64), {} in [64,128), {} in [128,256), {} in [256,512), {} in [512,1024), {} >= 1024)".format(
        len(functions),
        sum([1 for f in functions if len(f.blocks) == 1]),
        sum([1 for f in functions if len(f.blocks) >= 2 and len(f.blocks) < 4]),
        sum([1 for f in functions if len(f.blocks) >= 4 and len(f.blocks) < 8]),
        sum([1 for f in functions if len(f.blocks) >= 8 and len(f.blocks) < 16]),
        sum([1 for f in functions if len(f.blocks) >= 16 and len(f.blocks) < 32]),
        sum([1 for f in functions if len(f.blocks) >= 32 and len(f.blocks) < 64]),
        sum([1 for f in functions if len(f.blocks) >= 64 and len(f.blocks) < 128]),
        sum([1 for f in functions if len(f.blocks) >= 128 and len(f.blocks) < 256]),
        sum([1 for f in functions if len(f.blocks) >= 256 and len(f.blocks) < 512]),
        sum([1 for f in functions if len(f.blocks) >= 512 and len(f.blocks) < 1024]),
        sum([1 for f in functions if len(f.blocks) > 1024])
        ))
    print("      total functions: {} (instrs: {} = 1, {} in [2,4), {} in [4,8), {} in [8,16), {} in [16,32), {} in [32,64), {} in [64,128), {} in [128,256), {} in [256,512), {} in [512,1024), {} >= 1024)".format(
        len(functions),
        sum([1 for f in functions if len(f.insts) == 1]),
        sum([1 for f in functions if len(f.insts) >= 2 and len(f.insts) < 4]),
        sum([1 for f in functions if len(f.insts) >= 4 and len(f.insts) < 8]),
        sum([1 for f in functions if len(f.insts) >= 8 and len(f.insts) < 16]),
        sum([1 for f in functions if len(f.insts) >= 16 and len(f.insts) < 32]),
        sum([1 for f in functions if len(f.insts) >= 32 and len(f.insts) < 64]),
        sum([1 for f in functions if len(f.insts) >= 64 and len(f.insts) < 128]),
        sum([1 for f in functions if len(f.insts) >= 128 and len(f.insts) < 256]),
        sum([1 for f in functions if len(f.insts) >= 256 and len(f.insts) < 512]),
        sum([1 for f in functions if len(f.insts) >= 512 and len(f.insts) < 1024]),
        sum([1 for f in functions if len(f.insts) > 1024])
        ))
    print("         total blocks: {} (instrs for block: {} = 1, {} in [2,4), {} in [4,8), {} in [8,16), {} in [16,32), {} in [32,64), {} in [64,128), {} in [128,256), {} in [256,512), {} in [512,1024), {} >= 1024)".format(
        sum([f.n_blocks for f in functions]),
        sum([sum(f.n_blocks_list[:1]) for f in functions]),
        sum([sum(f.n_blocks_list[1:3]) for f in functions]),
        sum([sum(f.n_blocks_list[3:7]) for f in functions]),
        sum([sum(f.n_blocks_list[7:15]) for f in functions]),
        sum([sum(f.n_blocks_list[15:31]) for f in functions]),
        sum([sum(f.n_blocks_list[31:63]) for f in functions]),
        sum([sum(f.n_blocks_list[63:127]) for f in functions]),
        sum([sum(f.n_blocks_list[127:255]) for f in functions]),
        sum([sum(f.n_blocks_list[255:511]) for f in functions]),
        sum([sum(f.n_blocks_list[511:1023]) for f in functions]),
        sum([f.n_blocks_lg1024 for f in functions])
        ))
    print("   total instructions: {} ({} have >=3 args)".format(
        sum([f.n_insts for f in functions]),
        sum([f.n_insts_args3 for f in functions])))

def load_asmfiles(datahome, subdirs:List[str], b_save = True):
    inpath = datahome
    for dir in subdirs:
        inpath = Path(inpath) / dir

    if os.path.isdir(inpath):
        picklefile = 'out'
        for dir in subdirs:
            picklefile = picklefile + '-' + dir
        picklefile = datahome + picklefile + '.pkl'     

        funcs = []                
        if os.path.lexists(picklefile):
            funcs += load_functions(picklefile)            
        else:
            filenames = [] 
            for filename in sorted(os.listdir(inpath)):
                if os.path.isfile(Path(inpath) / filename):
                    filenames.append(Path(inpath) / filename)
                else:
                    funcs += load_asmfiles(datahome, subdirs + [filename])
            if len(filenames):
                if b_save:                 
                    funcs += create_functions(filenames, picklefile)
                else:
                    funcs += create_functions(filenames)
                    
        #functions_summary(funcs)
        return funcs
    else:
        raise(Exception('file'))

def save_asmfile(asmfile, funcs):
    #print("Dump functions to {}".format(asmfile))
    with open(asmfile, "wb") as f:
        pickle.dump(funcs, f)

def load_asmfile(asmfile):
    funcs = create_functions([asmfile])
    return funcs