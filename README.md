# UniASM

A pre-train Language model for Binary Code Similarity Detection (BCSD) tasks. Currently supported platforms: x86_64.

You can find the pre-trained model [here](https://github.com/clm07/UniASM/releases/download/v2.0/uniasm_base.h5). 



## Acknowledgement:

This implementation is based on [bert4keras](https://github.com/bojone/bert4keras), [SimBERT](https://github.com/ZhuiyiTechnology/simbert) and [Asm2Vec-pytorch](https://github.com/oalieno/asm2vec-pytorch).



## Requirements:

- Tensorflow >= 2.4
- python3
- Radare2 (optional, for ASM files generation)



We use the following commands to initialize the conda environment:

```
conda create -n uniasm python=3.8
conda activate uniasm
pip install numpy==1.19.5 tensorflow==2.5.0 bert_pytorch scikit-learn==1.2.0 pandas==1.4.4 transformers==4.25.1 matplotlib==3.6.2 tqdm
```



## 1. Generate ASM files

```
python bin2asm.py -i testdata\bin\elfedit-gcc-o0 -o testdata\out-elfedit-gcc-o0
```



## 2. Generate embedding for ASM function

```
python embedding.py testdata\out-elfedit-gcc-o0\dbg.adjust_relative_path
```

outputs:

```
100%|███████████████████████████████████| 1/1 [00:00<?, ?it/s]
[ 0.7137607   0.9913202  -0.85646147  0.92895794  0.6742756   0.20073773
  0.99648523 -0.9999651  -0.99792427 -0.9964269   0.6482426  -0.99827206
  ...
 -0.38122523 -0.9998596   0.9994266   0.80819637  0.97346854  0.92341983
 -0.5838027   0.9094241   0.99129903 -0.99984354 -0.05967474  0.96710646]
```



## 3. Calculate the similarity of two ASM functions

```
python similarity.py testdata\out-elfedit-gcc-o0\dbg.adjust_relative_path testdata\out-elfedit-gcc-o1\dbg.adjust_relative_path
```

outputs:

```
100%|███████████████████████████████████| 1/1 [00:00<00:00, 977.01it/s]
100%|███████████████████████████████████| 1/1 [00:00<00:00, 972.25it/s]
0.91649896
```



## 4. Evaluate Recall@k for BCSD searching task

```
python evaluate_recall@k.py
```

outputs
```
Tensorflow version 2.5.0
100%|███████████████████████████████████| 16/16 [00:01<00:00,  9.00it/s]
100%|███████████████████████████████████| 16/16 [00:01<00:00, 12.77it/s] 
calc top-10: 100%|███████████████████████████████████| 48/48 [00:00<00:00, 2327.77it/s] 
Top-10 recall:
0.85 0.90 0.92 0.98 0.98 0.98 0.98 0.98 0.98 0.98
```



## 5. Re-train UniASM with new training dataset

**Step 1**, create the training dataset.

```
python train_dataset.py
```



**Step 2**, train the model.

```
python train.py
```

