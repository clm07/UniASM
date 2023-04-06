
import tensorflow as tf

from utils.model import define_uniasm_model

print("Tensorflow version " + tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE
strategy = tf.distribute.get_strategy()
    
try:
    num_replicas_in_sync = strategy.num_replicas_in_sync 
except:
    num_replicas_in_sync = 1
print("num_replicas_in_sync: " + str(num_replicas_in_sync))

OUT_HOME = './testdata/'
TRAIN_SIZE = 86 # items in train dataset
TRAIN_FILENAMES = tf.io.gfile.glob(OUT_HOME + "train-86.tfrec")
VAL_SIZE = 10 # items in validate dataset
VAL_FILENAMES = tf.io.gfile.glob(OUT_HOME + "validate-10.tfrec")
OUT_MODEL_FILE = OUT_HOME + 'pretrain_model.h5'
OUT_HISTORY_FILE = OUT_HOME + 'pretrain_history.csv'
OUT_WEIGHTS_FILE = OUT_HOME + 'pretrain_weights.h5'

# params 
VOCAB_SIZE=21000
MAX_SEQ_LEN = 256
config_json = """
{
    "hidden_act": "gelu",
    "hidden_size": 516,
    "intermediate_size": 3072,
    "max_position_embeddings": yyy,
    "num_hidden_layers": 4,
    "vocab_size": xxx,
    "num_attention_heads": 12
}
"""
config_json = config_json.replace('xxx', str(VOCAB_SIZE))
config_json = config_json.replace('yyy', str(MAX_SEQ_LEN))


# train params
EPOCHS = 20
BATCH_SIZE = 8 * num_replicas_in_sync
STEPS_EPOCH = TRAIN_SIZE // BATCH_SIZE # steps for one epoch
STEPS_VALID = VAL_SIZE // BATCH_SIZE # steps for one validation 
TOKEN_PAD = 4
TOKEN_MASK = 3

def read_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "id": tf.io.FixedLenFeature([MAX_SEQ_LEN], tf.int64), 
        "seg": tf.io.FixedLenFeature([MAX_SEQ_LEN], tf.int64),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    return (example['id'], example['seg']),0
    
def load_dataset(files):
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=1)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=1)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset

trainning_dataset = load_dataset(TRAIN_FILENAMES)
validation_dataset = load_dataset(VAL_FILENAMES)

# print some train data
# dataset = trainning_dataset.enumerate()
# i = 0
# c = 0
# for data in dataset:
#     if i == c:
#         print(data[1][0][0][0])
#         print(data[1][0][1][0])
#         print(data[1][0][2][0])
#         print(data[1][0][3][0])
#         break
#     i += 1
#     i += 1

with strategy.scope():
    train_model = define_uniasm_model(config_json)
    train_model.summary()

# learning rate
from matplotlib import pyplot as plt
LR_START = 0.00005
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = 0.00005
LR_RAMPUP_EPOCHS = 4
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = 0.8
def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
#plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


import os
import csv
class StoreModelHistory(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(StoreModelHistory, self).__init__(**kwargs)
        self.wh = False
        
    def on_train_begin(self, logs=None):
        try:
            os.remove(OUT_HISTORY_FILE)
        except:
            pass

    def on_epoch_end(self, batch, logs=None):
        if ('lr' not in logs.keys()):
            logs.setdefault('lr',0)
            logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

        if not self.wh:
            self.wh = True
            with open(OUT_HISTORY_FILE,'a') as f:
                y=csv.DictWriter(f,logs.keys())
                y.writeheader()

        with open(OUT_HISTORY_FILE,'a') as f:
            y=csv.DictWriter(f,logs.keys())
            y.writerow(logs)
            
class ModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self):
        self.lowest = 1e10
        
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(OUT_HOME + 'model_ep{}.h5'.format(epoch))
        if logs['val_loss'] <= self.lowest:
            self.lowest = logs['val_loss']
            self.model.save_weights(OUT_HOME + 'best_model.h5')

csv_logger = tf.keras.callbacks.CSVLogger(OUT_HOME + 'training.log')

train_model.fit(
    trainning_dataset,
    validation_data=validation_dataset, 
    validation_steps=STEPS_VALID, 
    steps_per_epoch=STEPS_EPOCH,
    epochs=EPOCHS,
    callbacks=[lr_callback, ModelCheckpoint(), StoreModelHistory(), csv_logger]
)
