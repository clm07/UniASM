import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras import initializers
from tensorflow.keras import optimizers
from tensorflow.keras import activations
from tensorflow.python.util import nest, tf_inspect
from tensorflow.python.eager import tape
import numpy as np
import json
import re

do_recompute = False #strtobool(os.environ.get('RECOMPUTE', '0'))

def infinity():
    """返回默认的代表无穷大的数值
    """
    return tf.keras.utils.get_custom_objects().get('infinity', 1e12)

def recompute_grad(call):
    """重计算装饰器（用来装饰Keras层的call函数）
    关于重计算，请参考：https://arxiv.org/abs/1604.06174
    """
    if not do_recompute:
        return call

    def inner(self, inputs, **kwargs):
        """定义需要求梯度的函数以及重新定义求梯度过程
        （参考自官方自带的tf.recompute_grad函数）
        """
        flat_inputs = nest.flatten(inputs)
        call_args = tf_inspect.getfullargspec(call).args
        for key in ['mask', 'training']:
            if key not in call_args and key in kwargs:
                del kwargs[key]

        def kernel_call():
            """定义前向计算
            """
            return call(self, inputs, **kwargs)

        def call_and_grad(*inputs):
            """定义前向计算和反向计算
            """
            with tape.stop_recording():
                outputs = kernel_call()
                outputs = tf.identity(outputs)

            def grad_fn(doutputs, variables=None):
                watches = list(inputs)
                if variables is not None:
                    watches += list(variables)
                with tf.GradientTape() as t:
                    t.watch(watches)
                    with tf.control_dependencies([doutputs]):
                        outputs = kernel_call()
                grads = t.gradient(
                    outputs, watches, output_gradients=[doutputs]
                )
                del t
                return grads[:len(inputs)], grads[len(inputs):]

            return outputs, grad_fn

        outputs, grad_fn = call_and_grad(*flat_inputs)
        flat_outputs = nest.flatten(outputs)

        def actual_grad_fn(*doutputs):
            grads = grad_fn(*doutputs, variables=self.trainable_weights)
            return grads[0] + grads[1]

        watches = flat_inputs + self.trainable_weights
        watches = [tf.convert_to_tensor(x) for x in watches]
        tape.record_operation(
            call.__name__, flat_outputs, watches, actual_grad_fn
        )
        return outputs

    return inner

def insert_arguments(**arguments):
    """装饰器，为类方法增加参数
    （主要用于类的__init__方法）
    """
    def actual_decorator(func):
        def new_func(self, *args, **kwargs):
            for k, v in arguments.items():
                if k in kwargs:
                    v = kwargs.pop(k)
                setattr(self, k, v)
            return func(self, *args, **kwargs)

        return new_func

    return actual_decorator

def orthogonally_resize(a, new_shape, window=2):
    """简单的正交化缩放矩阵
    """
    assert a.ndim == len(new_shape)
    slices, a_norm, w = [], np.linalg.norm(a), window
    for i, (d1, d2) in enumerate(zip(a.shape, new_shape)):
        if d1 != d2:
            k = d2 // d1 + int(d2 % d1 != 0)
            if k > 1:
                assert d1 % w == 0
                a = a.reshape(a.shape[:i] + (d1 // w, w) + a.shape[i + 1:])
                a = np.repeat(a, k, axis=i)
                a = a.reshape(a.shape[:i] + (d1 * k,) + a.shape[i + 2:])
        slices.append(np.s_[:d2])
    a = a[tuple(slices)]
    return a / np.linalg.norm(a) * a_norm

def is_one_of(x, ys):
    """判断x是否在ys之中
    等价于x in ys，但有些情况下x in ys会报错
    """
    for y in ys:
        if x is y:
            return True
    return False

def string_matching(s, keywords):
    """判断s是否至少包含keywords中的至少一个字符串
    """
    for k in keywords:
        if re.search(k, s):
            return True
    return False

def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, str)

def export_to_custom_objects(base_extend_with):
    """装饰器，用来将优化器放到custom_objects中
    """
    def new_extend_with(BaseOptimizer, name=None):
        NewOptimizer = base_extend_with(BaseOptimizer)

        if is_string(name):
            NewOptimizer.__name__ = name

        name = NewOptimizer.__name__
        tf.keras.utils.get_custom_objects()[name] = NewOptimizer
        
        return NewOptimizer

    return new_extend_with

class Adam(optimizers.Optimizer):
    """重新定义Adam优化器，便于派生出新的优化器
    （tensorflow的optimizer_v2类）
    """
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        bias_correction=True,
        **kwargs
    ):
        kwargs['name'] = kwargs.get('name') or 'Adam'
        super(Adam, self).__init__(**kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or K.epislon()
        self.bias_correction = bias_correction

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')

    def _resource_apply(self, grad, var, indices=None):
        # 准备变量
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = K.cast(self.epsilon, var_dtype)
        local_step = K.cast(self.iterations + 1, var_dtype)
        beta_1_t_power = K.pow(beta_1_t, local_step)
        beta_2_t_power = K.pow(beta_2_t, local_step)

        # 更新公式
        if indices is None:
            m_t = K.update(m, beta_1_t * m + (1 - beta_1_t) * grad)
            v_t = K.update(v, beta_2_t * v + (1 - beta_2_t) * K.square(grad))
        else:
            mv_ops = [K.update(m, beta_1_t * m), K.update(v, beta_2_t * v)]
            with tf.control_dependencies(mv_ops):
                m_t = self._resource_scatter_add(
                    m, indices, (1 - beta_1_t) * grad
                )
                v_t = self._resource_scatter_add(
                    v, indices, (1 - beta_2_t) * K.square(grad)
                )

        # 返回算子
        with tf.control_dependencies([m_t, v_t]):
            if self.bias_correction:
                m_t = m_t / (1.0 - beta_1_t_power)
                v_t = v_t / (1.0 - beta_2_t_power)
            var_t = var - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
            return K.update(var, var_t)

    def _resource_apply_dense(self, grad, var):
        return self._resource_apply(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        return self._resource_apply(grad, var, indices)

    def get_config(self):
        config = {
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
            'bias_correction': self.bias_correction,
        }
        base_config = super(Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

@export_to_custom_objects
def extend_with_weight_decay(BaseOptimizer):
    """返回新的优化器类，加入权重衰减
    """
    class NewOptimizer(BaseOptimizer):
        """带有权重衰减的优化器
        """
        @insert_arguments(weight_decay_rate=0.01, exclude_from_weight_decay=[])
        def __init__(self, *args, **kwargs):
            super(NewOptimizer, self).__init__(*args, **kwargs)
            if not hasattr(self, 'learning_rate'):
                self.learning_rate = self.lr

        def get_updates(self, loss, params):
            old_update = K.update

            def new_update(x, new_x):
                if is_one_of(x, params) and self._do_weight_decay(x):
                    new_x = new_x - self.learning_rate * self.weight_decay_rate * x
                return old_update(x, new_x)

            K.update = new_update
            updates = super(NewOptimizer, self).get_updates(loss, params)
            K.update = old_update

            return updates

        def _do_weight_decay(self, w):
            return (not string_matching(w.name, self.exclude_from_weight_decay))

        def get_config(self):
            config = {
                'weight_decay_rate': self.weight_decay_rate,
                'exclude_from_weight_decay': self.exclude_from_weight_decay,
            }
            base_config = super(NewOptimizer, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    return NewOptimizer

def integerize_shape(func):
    """装饰器，保证input_shape一定是int或None
    """
    def convert(item):
        if hasattr(item, '__iter__'):
            return [convert(i) for i in item]
        elif hasattr(item, 'value'):
            return item.value
        else:
            return item

    def new_func(self, input_shape):
        input_shape = convert(input_shape)
        return func(self, input_shape)

    return new_func

def sequence_masking(x, mask, value=0, axis=None):
    """为序列条件mask的函数
    mask: 形如(batch_size, seq_len)的0-1矩阵；
    value: mask部分要被替换成的值，可以是'-inf'或'inf'；
    axis: 序列所在轴，默认为1；
    """
    if mask is None:
        return x
    else:
        x_dtype = K.dtype(x)
        if x_dtype == 'bool':
            x = K.cast(x, 'int32')
        if K.dtype(mask) != K.dtype(x):
            mask = K.cast(mask, K.dtype(x))
        if value == '-inf':
            value = -infinity()
        elif value == 'inf':
            value = infinity()
        if axis is None:
            axis = 1
        elif axis < 0:
            axis = K.ndim(x) + axis
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        value = K.cast(value, K.dtype(x))
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        x = x * mask + value * (1 - mask)
        if x_dtype == 'bool':
            x = K.cast(x, 'bool')
        return x

class Embedding(layers.Embedding):
    """拓展Embedding层
    """
    def compute_mask(self, inputs, mask=None):
        """为了适配T5，保证第一个token不被mask
        """
        if K.ndim(inputs) == 2:
            mask = super(Embedding, self).compute_mask(inputs, mask)
            if mask is not None:
                mask1 = K.ones_like(mask[:, :1], dtype='bool')
                mask2 = mask[:, 1:]
                return K.concatenate([mask1, mask2], 1)
        else:
            return mask

    def call(self, inputs, mode='embedding'):
        """新增mode参数，可以为embedding或dense。如果为embedding，
        则等价于普通Embedding层；如果为dense，则等价于无bias的Dense层。
        """
        if mode == 'embedding':
            return super(Embedding, self).call(inputs)
        else:
            kernel = K.transpose(self.embeddings)
            return K.dot(inputs, kernel)

    def compute_output_shape(self, input_shape):
        """关于判据，本来是通过缓存call时的mode参数来判断的，但是后来发现
        Keras在使用compute_output_shape的时候不一定配套调用了call函数，
        所以缓存的mode可能是不准的，因此只能出此下策。
        """
        if len(input_shape) == 2:
            return super(Embedding, self).compute_output_shape(input_shape)
        else:
            return input_shape[:2] + (K.int_shape(self.embeddings)[0],)

class BiasAdd(Layer):
    """加上偏置项
    """
    @integerize_shape
    def build(self, input_shape):
        super(BiasAdd, self).build(input_shape)
        output_dim = input_shape[-1]
        self.bias = self.add_weight(
            name='bias', shape=(output_dim,), initializer='zeros'
        )

    def call(self, inputs):
        return K.bias_add(inputs, self.bias)

class Concatenate1D(Layer):
    """1维序列拼接层
    说明：本来该功能可以直接通过Concatenate层来实现，无奈Keras
          自带的Concatenate层的compute_mask写得不合理，导致一个
          带mask的序列与一个不带mask的序列拼接会报错，因此干脆
          自己重写一个好了。
    """
    def call(self, inputs):
        return K.concatenate(inputs, axis=1)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            masks = []
            for i, m in enumerate(mask):
                if m is None:
                    m = K.ones_like(inputs[i][..., 0], dtype='bool')
                masks.append(m)
            return K.concatenate(masks, axis=1)

    def compute_output_shape(self, input_shape):
        if all([shape[1] for shape in input_shape]):
            seq_len = sum([shape[1] for shape in input_shape])
            return (input_shape[0][0], seq_len, input_shape[0][2])
        else:
            return (input_shape[0][0], None, input_shape[0][2])

class FeedForward(Layer):
    """FeedForward层
    如果activation不是一个list，那么它就是两个Dense层的叠加；如果activation是
    一个list，那么第一个Dense层将会被替换成门控线性单元（Gated Linear Unit）。
    参考论文: https://arxiv.org/abs/2002.05202
    """
    def __init__(
        self,
        units,
        activation='relu',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(FeedForward, self).__init__(**kwargs)
        self.units = units
        if not isinstance(activation, list):
            activation = [activation]
        self.activation = [activations.get(act) for act in activation]
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)

    @integerize_shape
    def build(self, input_shape):
        super(FeedForward, self).build(input_shape)
        output_dim = input_shape[-1]

        for i, activation in enumerate(self.activation):
            i_dense = Dense(
                units=self.units,
                activation=activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer
            )
            setattr(self, 'i%s_dense' % i, i_dense)

        self.o_dense = Dense(
            units=output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    @recompute_grad
    def call(self, inputs):
        x = self.i0_dense(inputs)
        for i in range(1, len(self.activation)):
            x = x * getattr(self, 'i%s_dense' % i)(inputs)
        x = self.o_dense(x)
        return x

    def get_config(self):
        config = {
            'units': self.units,
            'activation': [
                activations.serialize(act) for act in self.activation
            ],
            'use_bias': self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class PositionEmbedding(Layer):
    """定义可训练的位置Embedding
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        merge_mode='add',
        hierarchical=None,
        embeddings_initializer='zeros',
        custom_position_ids=False,
        **kwargs
    ):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.hierarchical = hierarchical
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.custom_position_ids = custom_position_ids

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if 'int' not in K.dtype(position_ids):
                position_ids = K.cast(position_ids, 'int32')
        else:
            input_shape = K.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = K.arange(0, seq_len, dtype='int32')[None]

        if self.hierarchical:
            alpha = 0.4 if self.hierarchical is True else self.hierarchical
            embeddings = self.embeddings - alpha * self.embeddings[:1]
            embeddings = embeddings / (1 - alpha)
            embeddings_x = K.gather(embeddings, position_ids // self.input_dim)
            embeddings_y = K.gather(embeddings, position_ids % self.input_dim)
            embeddings = alpha * embeddings_x + (1 - alpha) * embeddings_y
        else:
            if self.custom_position_ids:
                embeddings = K.gather(self.embeddings, position_ids)
            else:
                embeddings = self.embeddings[None, :seq_len]

        if self.merge_mode == 'add':
            return inputs + embeddings
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0)
        elif self.merge_mode == 'zero':
            return embeddings
        else:
            if not self.custom_position_ids:
                embeddings = K.tile(embeddings, [batch_size, 1, 1])
            return K.concatenate([inputs, embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode in ['add', 'mul', 'zero']:
            return input_shape[:2] + (self.output_dim,)
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'hierarchical': self.hierarchical,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer),
            'custom_position_ids': self.custom_position_ids,
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class LayerNormalization(Layer):
    """(Conditional) Layer Normalization
    hidden_*系列参数仅为有条件输入时(conditional=True)使用
    """
    def __init__(
        self,
        center=True,
        scale=True,
        epsilon=None,
        conditional=False,
        hidden_units=None,
        hidden_activation='linear',
        hidden_initializer='glorot_uniform',
        **kwargs
    ):
        super(LayerNormalization, self).__init__(**kwargs)
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)
        self.epsilon = epsilon or 1e-12

    def compute_mask(self, inputs, mask=None):
        if self.conditional:
            masks = mask if mask is not None else []
            masks = [m[None] for m in masks if m is not None]
            if len(masks) == 0:
                return None
            else:
                return K.all(K.concatenate(masks, axis=0), axis=0)
        else:
            return mask

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)

        if self.conditional:
            shape = (input_shape[0][-1],)
        else:
            shape = (input_shape[-1],)

        if self.center:
            self.beta = self.add_weight(
                shape=shape, initializer='zeros', name='beta'
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape, initializer='ones', name='gamma'
            )

        if self.conditional:

            if self.hidden_units is not None:
                self.hidden_dense = Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer
                )

            if self.center:
                self.beta_dense = Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )
            if self.scale:
                self.gamma_dense = Dense(
                    units=shape[0], use_bias=False, kernel_initializer='zeros'
                )

    @recompute_grad
    def call(self, inputs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是condition
        """
        if self.conditional:
            inputs, cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond, 1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = K.mean(outputs, axis=-1, keepdims=True)
            outputs = outputs - mean
        if self.scale:
            variance = K.mean(K.square(outputs), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            outputs = outputs / std * gamma
        if self.center:
            outputs = outputs + beta

        return outputs

    def compute_output_shape(self, input_shape):
        if self.conditional:
            return input_shape[0]
        else:
            return input_shape

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': activations.serialize(self.hidden_activation),
            'hidden_initializer':
                initializers.serialize(self.hidden_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MultiHeadAttention(Layer):
    """多头注意力机制
    """
    def __init__(
        self,
        heads,
        head_size,
        out_dim=None,
        key_size=None,
        use_bias=True,
        attention_scale=True,
        attention_dropout=None,
        return_attention_scores=False,
        kernel_initializer='glorot_uniform',
        **kwargs
    ):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = out_dim or heads * head_size
        self.key_size = key_size or head_size
        self.use_bias = use_bias
        self.attention_scale = attention_scale
        self.attention_dropout = attention_dropout
        self.return_attention_scores = return_attention_scores
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        super(MultiHeadAttention, self).build(input_shape)
        self.q_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.k_dense = Dense(
            units=self.key_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.v_dense = Dense(
            units=self.head_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.o_dense = Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )

    @recompute_grad
    def call(self, inputs, mask=None, **kwargs):
        """实现多头注意力
        q_mask: 对输入的query序列的mask。
                主要是将输出结果的padding部分置0。
        v_mask: 对输入的value序列的mask。
                主要是防止attention读取到padding信息。
        """
        q, k, v = inputs[:3]
        q_mask, v_mask = None, None
        if mask is not None:
            q_mask, v_mask = mask[0], mask[2]
        # 线性变换
        qw = self.q_dense(q)
        kw = self.k_dense(k)
        vw = self.v_dense(v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
        # Attention
        qkv_inputs = [qw, kw, vw] + inputs[3:]
        qv_masks = [q_mask, v_mask]
        o, a = self.pay_attention_to(qkv_inputs, qv_masks, **kwargs)
        # 完成输出
        o = K.reshape(o, (-1, K.shape(o)[1], self.head_size * self.heads))
        o = self.o_dense(o)
        # 返回结果
        if self.return_attention_scores:
            return [o, a]
        else:
            return o

    def pay_attention_to(self, inputs, mask=None, **kwargs):
        """实现标准的乘性多头注意力
        a_bias: 对attention矩阵的bias。
                不同的attention bias对应不同的应用。
        p_bias: 在attention里的位置偏置。
                一般用来指定相对位置编码的种类。
        说明: 这里单独分离出pay_attention_to函数，是为了方便
              继承此类来定义不同形式的atttention；此处要求
              返回o.shape=(batch_size, seq_len, heads, head_size)。
        """
        (qw, kw, vw), n = inputs[:3], 3
        q_mask, v_mask = mask
        a_bias, p_bias = kwargs.get('a_bias'), kwargs.get('p_bias')
        if a_bias:
            a_bias = inputs[n]
            n += 1
        if p_bias == 'rotary':
            cos_pos = K.repeat_elements(inputs[n][..., None, 1::2], 2, -1)
            sin_pos = K.repeat_elements(inputs[n][..., None, ::2], 2, -1)
            qw2 = K.stack([-qw[..., 1::2], qw[..., ::2]], 4)
            qw2 = K.reshape(qw2, K.shape(qw))
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = K.stack([-kw[..., 1::2], kw[..., ::2]], 4)
            kw2 = K.reshape(kw2, K.shape(kw))
            kw = kw * cos_pos + kw2 * sin_pos
        # Attention
        a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
        # 处理位置编码
        if p_bias == 'typical_relative':
            position_bias = inputs[n]
            a = a + tf.einsum('bjhd,jkd->bhjk', qw, position_bias)
        elif p_bias == 't5_relative':
            position_bias = K.permute_dimensions(inputs[n], (2, 0, 1))
            a = a + K.expand_dims(position_bias, 0)
        # Attention（续）
        if self.attention_scale:
            a = a / self.key_size**0.5
        if a_bias is not None:
            a = a + a_bias
        a = sequence_masking(a, v_mask, '-inf', -1)
        A = K.softmax(a)
        if self.attention_dropout:
            A = Dropout(self.attention_dropout)(A)
        # 完成输出
        o = tf.einsum('bhjk,bkhd->bjhd', A, vw)
        if p_bias == 'typical_relative':
            o = o + tf.einsum('bhjk,jkd->bjhd', A, position_bias)
        return o, a

    def compute_output_shape(self, input_shape):
        o_shape = (input_shape[0][0], input_shape[0][1], self.out_dim)
        if self.return_attention_scores:
            a_shape = (
                input_shape[0][0], self.heads, input_shape[0][1],
                input_shape[1][1]
            )
            return [o_shape, a_shape]
        else:
            return o_shape

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.return_attention_scores:
                return [mask[0], None]
            else:
                return mask[0]

    def get_config(self):
        config = {
            'heads': self.heads,
            'head_size': self.head_size,
            'out_dim': self.out_dim,
            'key_size': self.key_size,
            'use_bias': self.use_bias,
            'attention_scale': self.attention_scale,
            'attention_dropout': self.attention_dropout,
            'return_attention_scores': self.return_attention_scores,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Transformer(object):
    """模型基类
    """
    def __init__(
        self,
        vocab_size,  # 词表大小
        hidden_size,  # 编码维度
        num_hidden_layers,  # Transformer总层数
        num_attention_heads,  # Attention的头数
        intermediate_size,  # FeedForward的隐层维度
        hidden_act,  # FeedForward隐层的激活函数
        dropout_rate=None,  # Dropout比例
        attention_dropout_rate=None,  # Attention矩阵的Dropout比例
        embedding_size=None,  # 是否指定embedding_size
        attention_head_size=None,  # Attention中V的head_size
        attention_key_size=None,  # Attention中Q,K的head_size
        sequence_length=None,  # 是否固定序列长度
        keep_tokens=None,  # 要保留的词ID列表
        compound_tokens=None,  # 扩展Embedding
        residual_attention_scores=False,  # Attention矩阵加残差
        ignore_invalid_weights=False,  # 允许跳过不存在的权重
        autoresize_weights=False,  # 自动变换形状不匹配的权重
        layers=None,  # 外部传入的Keras层
        prefix=None,  # 层名前缀
        name=None,  # 模型名称
        **kwargs
    ):
        if keep_tokens is not None:
            vocab_size = len(keep_tokens)
        if compound_tokens is not None:
            vocab_size += len(compound_tokens)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size or hidden_size // num_attention_heads
        self.attention_key_size = attention_key_size or self.attention_head_size
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate or 0
        self.attention_dropout_rate = attention_dropout_rate or 0
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size or hidden_size
        self.sequence_length = sequence_length
        self.keep_tokens = keep_tokens
        self.compound_tokens = compound_tokens
        self.attention_bias = None
        self.position_bias = None
        self.attention_scores = None
        self.residual_attention_scores = residual_attention_scores
        self.ignore_invalid_weights = ignore_invalid_weights
        self.autoresize_weights = autoresize_weights
        self.layers = {} if layers is None else layers
        self.prefix = prefix or ''
        self.name = name
        self.built = False

    def build(
        self,
        attention_caches=None,
        layer_norm_cond=None,
        layer_norm_cond_hidden_size=None,
        layer_norm_cond_hidden_act=None,
        additional_input_layers=None,
        **kwargs
    ):
        """模型构建函数
        attention_caches：为Attention的K,V的缓存序列字典，格式为
                         {Attention层名: [K缓存, V缓存]}；
        layer_norm_*系列参数：实现Conditional Layer Normalization时使用，
                            用来实现以“固定长度向量”为条件的条件Bert。
        """
        if self.built:
            return None
        # Input
        inputs = self.get_inputs()
        self.set_inputs(inputs, additional_input_layers)
        # Other
        self.attention_caches = attention_caches or {}
        self.layer_norm_conds = [
            layer_norm_cond,
            layer_norm_cond_hidden_size,
            layer_norm_cond_hidden_act or 'linear',
        ]
        # Call
        outputs = self.call(inputs)
        self.set_outputs(outputs)
        # Model
        self.model = models.Model(self.inputs, self.outputs, name=self.name)
        self.built = True

    def call(self, inputs):
        """定义模型的执行流程
        """
        # Embedding
        outputs = self.apply_embeddings(inputs)
        # Main
        for i in range(self.num_hidden_layers):
            outputs = self.apply_main_layers(outputs, i)
        # Final
        outputs = self.apply_final_layers(outputs)
        return outputs

    def prefixed(self, name):
        """给名字加前缀
        """
        if name is not None:
            return self.prefix + name

    def apply(self, inputs=None, layer=None, arguments=None, **kwargs):
        """通过apply调用层会自动重用同名层
        inputs: 上一层的输出；
        layer: 要调用的层类名；
        arguments: 传递给layer.call的参数；
        kwargs: 传递给层初始化的参数。
        """
        if layer is Dropout and self.dropout_rate == 0:
            return inputs

        if layer is MultiHeadAttention and self.residual_attention_scores:
            kwargs['return_attention_scores'] = True

        arguments = arguments or {}
        if layer is Lambda:
            kwargs['arguments'] = arguments
            arguments = {}

        name = self.prefixed(kwargs.get('name'))
        kwargs['name'] = name
        if name not in self.layers:
            layer = layer(**kwargs)
            name = layer.name
            self.layers[name] = layer

        if inputs is None:
            return self.layers[name]
        else:
            if isinstance(self.layers[name], MultiHeadAttention):
                if name in self.attention_caches:
                    # 如果检测到Cache的传入，那么自动在Key,Value处拼接起来
                    k_cache, v_cache = self.attention_caches[name]
                    k_name, v_name = name + '-Cached-Key', name + '-Cached-Value'
                    k = Concatenate1D(name=k_name)([k_cache, inputs[1]])
                    v = Concatenate1D(name=v_name)([v_cache, inputs[2]])
                    inputs = inputs[:1] + [k, v] + inputs[3:]
                if self.residual_attention_scores:
                    # 如果使用残差Attention矩阵，则给每个Attention矩阵加上前上一层的Attention
                    # 矩阵，这对应RealFormer设计（https://arxiv.org/abs/2012.11747）。目前
                    # 该实现还相对粗糙，可能欠缺通用性。
                    if self.attention_scores is not None:
                        if arguments.get('a_bias'):
                            a_bias = Add(name=name + '-Attention-Bias'
                                        )([inputs[3], self.attention_scores])
                            inputs = inputs[:3] + [a_bias] + inputs[4:]
                        else:
                            a_bias = self.attention_scores
                            inputs = inputs[:3] + [a_bias] + inputs[3:]
                        arguments['a_bias'] = True
                    o, a = self.layers[name](inputs, **arguments)
                    self.attention_scores = a
                    return o
            return self.layers[name](inputs, **arguments)

    def get_inputs(self):
        raise NotImplementedError

    def apply_embeddings(self, inputs):
        raise NotImplementedError

    def apply_main_layers(self, inputs, index):
        raise NotImplementedError

    def apply_final_layers(self, inputs):
        raise NotImplementedError

    def compute_attention_bias(self, inputs=None):
        """定义每一层的Attention Bias
        """
        return self.attention_bias

    def compute_position_bias(self, inputs=None):
        """定义每一层的Position Bias（一般相对位置编码用）
        """
        return self.position_bias

    def set_inputs(self, inputs, additional_input_layers=None):
        """设置input和inputs属性
        """
        if inputs is None:
            inputs = []
        elif not isinstance(inputs, list):
            inputs = [inputs]

        inputs = inputs[:]
        if additional_input_layers is not None:
            if not isinstance(additional_input_layers, list):
                additional_input_layers = [additional_input_layers]
            inputs.extend(additional_input_layers)

        self.inputs = inputs
        if len(inputs) > 1:
            self.input = inputs
        else:
            self.input = inputs[0]

    def set_outputs(self, outputs):
        """设置output和oututs属性
        """
        if not isinstance(outputs, list):
            outputs = [outputs]

        outputs = outputs[:]
        self.outputs = outputs
        if len(outputs) > 1:
            self.output = outputs
        else:
            self.output = outputs[0]

    @property
    def initializer(self):
        """默认使用截断正态分布初始化
        """
        return tf.keras.initializers.TruncatedNormal(stddev=0.02)

    def simplify(self, inputs):
        """将list中的None过滤掉
        """
        inputs = [i for i in inputs if i is not None]
        if len(inputs) == 1:
            inputs = inputs[0]

        return inputs

    def load_embeddings(self, embeddings):
        """处理Embedding层权重
        """
        embeddings = embeddings.astype(K.floatx())  # 防止np.average报错

        if self.keep_tokens is not None:
            embeddings = embeddings[self.keep_tokens]

        if self.compound_tokens is not None:
            ext_embeddings = []
            for item in self.compound_tokens:
                if isinstance(item, list):
                    item = (item, [1] * len(item))
                ext_embeddings.append(
                    np.average(embeddings[item[0]], 0, item[1])
                )
            embeddings = np.concatenate([embeddings, ext_embeddings], 0)

        return embeddings

    def load_variable(self, checkpoint, name):
        """加载单个变量的函数
        """
        if isinstance(checkpoint, dict):
            return checkpoint[name]
        else:
            return tf.train.load_variable(checkpoint, name)

    def create_variable(self, name, value, dtype=None):
        """创建一个变量
        """
        dtype = dtype or K.floatx()
        return K.variable(
            self.initializer(value.shape, dtype), dtype, name=name
        ), value

    def variable_mapping(self):
        """构建keras层与checkpoint的变量名之间的映射表
        """
        return {}

    def load_weights_from_checkpoint(self, checkpoint, mapping=None):
        """根据mapping从checkpoint加载权重
        """
        mapping = mapping or self.variable_mapping()
        mapping = {self.prefixed(k): v for k, v in mapping.items()}
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        weight_value_pairs = []
        for layer, variables in mapping.items():
            layer = self.layers[layer]
            weights, values = [], []

            for w, v in zip(layer.trainable_weights, variables):  # 允许跳过不存在的权重
                try:
                    values.append(self.load_variable(checkpoint, v))
                    weights.append(w)
                except Exception as e:
                    if self.ignore_invalid_weights:
                        print('%s, but ignored.' % e.message)
                    else:
                        raise e

            for i, (w, v) in enumerate(zip(weights, values)):
                if v is not None:
                    w_shape, v_shape = K.int_shape(w), v.shape
                    if self.autoresize_weights and w_shape != v_shape:
                        v = orthogonally_resize(v, w_shape)
                        if isinstance(layer, MultiHeadAttention):
                            count = 2
                            if layer.use_bias:
                                count += 2
                            if layer.attention_scale and i < count:
                                scale = 1.0 * w_shape[-1] / v_shape[-1]
                                v = v * scale**0.25
                        if isinstance(layer, FeedForward):
                            count = 1
                            if layer.use_bias:
                                count += 1
                            if self.hidden_act in ['relu', 'leaky_relu']:
                                count -= 2
                            if i < count:
                                v *= np.sqrt(1.0 * w_shape[-1] / v_shape[-1])
                            else:
                                v *= np.sqrt(1.0 * v_shape[0] / w_shape[0])

                    weight_value_pairs.append((w, v))

        K.batch_set_value(weight_value_pairs)

    def save_weights_as_checkpoint(self, filename, mapping=None, dtype=None):
        """根据mapping将权重保存为checkpoint格式
        """
        mapping = mapping or self.variable_mapping()
        mapping = {self.prefixed(k): v for k, v in mapping.items()}
        mapping = {k: v for k, v in mapping.items() if k in self.layers}

        with tf.Graph().as_default():
            all_variables, all_values = [], []
            for layer, variables in mapping.items():
                layer = self.layers[layer]
                values = K.batch_get_value(layer.trainable_weights)
                for name, value in zip(variables, values):
                    variable, value = self.create_variable(name, value, dtype)
                    all_variables.append(variable)
                    all_values.append(value)
            with tf.Session() as sess:
                K.batch_set_value(zip(all_variables, all_values))
                saver = tf.train.Saver()
                saver.save(sess, filename)


class BERT(Transformer):
    """构建BERT模型
    """
    def __init__(
        self,
        max_position,  # 序列最大长度
        segment_vocab_size=2,  # segment总数目
        with_pool=False,  # 是否包含Pool部分
        with_rsp=False,  # 是否包含RSP部分（函数关系预测）
        with_alg=False,  # 是否包含ALG部分（seq2seq生成预测）
        with_nsp=False,  # 是否包含NSP部分
        with_mlm=False,  # 是否包含MLM部分
        hierarchical_position=None,  # 是否层次分解位置编码
        custom_position_ids=False,  # 是否自行传入位置id
        shared_segment_embeddings=False,  # 若True，则segment跟token共用embedding
        **kwargs  # 其余参数
    ):
        super(BERT, self).__init__(**kwargs)
        self.max_position = max_position
        self.segment_vocab_size = segment_vocab_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm
        self.with_alg = with_alg
        self.with_rsp = with_rsp
        self.hierarchical_position = hierarchical_position
        self.custom_position_ids = custom_position_ids
        self.shared_segment_embeddings = shared_segment_embeddings
        if self.with_nsp and not self.with_pool:
            self.with_pool = True

    def get_inputs(self):
        """BERT的输入是token_ids和segment_ids
        （但允许自行传入位置id，以实现一些特殊需求）
        """
        x_in = self.apply(
            layer=Input, shape=(self.sequence_length,), name='Input-Token'
        )
        inputs = [x_in]

        if self.segment_vocab_size > 0:
            s_in = self.apply(
                layer=Input,
                shape=(self.sequence_length,),
                name='Input-Segment'
            )
            inputs.append(s_in)

        if self.custom_position_ids:
            p_in = self.apply(
                layer=Input,
                shape=(self.sequence_length,),
                name='Input-Position'
            )
            inputs.append(p_in)

        return inputs

    def apply_embeddings(self, inputs):
        """BERT的embedding是token、position、segment三者embedding之和
        """
        inputs = inputs[:]
        x = inputs.pop(0)
        if self.segment_vocab_size > 0:
            s = inputs.pop(0)
        if self.custom_position_ids:
            p = inputs.pop(0)
        else:
            p = None
        z = self.layer_norm_conds[0]

        x = self.apply(
            inputs=x,
            layer=Embedding,
            input_dim=self.vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=self.initializer,
            mask_zero=True,
            name='Embedding-Token'
        )
        if self.segment_vocab_size > 0:
            if self.shared_segment_embeddings:
                name = 'Embedding-Token'
            else:
                name = 'Embedding-Segment'
            s = self.apply(
                inputs=s,
                layer=Embedding,
                input_dim=self.segment_vocab_size,
                output_dim=self.embedding_size,
                embeddings_initializer=self.initializer,
                name=name
            )
            x = self.apply(
                inputs=[x, s], layer=Add, name='Embedding-Token-Segment'
            )
        x = self.apply(
            inputs=self.simplify([x, p]),
            layer=PositionEmbedding,
            input_dim=self.max_position,
            output_dim=self.embedding_size,
            merge_mode='add',
            hierarchical=self.hierarchical_position,
            embeddings_initializer=self.initializer,
            custom_position_ids=self.custom_position_ids,
            name='Embedding-Position'
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='Embedding-Norm'
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='Embedding-Dropout'
        )
        if self.embedding_size != self.hidden_size:
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                kernel_initializer=self.initializer,
                name='Embedding-Mapping'
            )

        return x

    def apply_main_layers(self, inputs, index):
        """BERT的主体是基于Self-Attention的模块
        顺序：Att --> Add --> LN --> FFN --> Add --> LN
        """
        x = inputs
        z = self.layer_norm_conds[0]

        attention_name = 'Transformer-%d-MultiHeadSelfAttention' % index
        feed_forward_name = 'Transformer-%d-FeedForward' % index
        attention_mask = self.compute_attention_bias(index)

        # Self Attention
        xi, x, arguments = x, [x, x, x], {'a_bias': None}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.append(attention_mask)

        x = self.apply(
            inputs=x,
            layer=MultiHeadAttention,
            arguments=arguments,
            heads=self.num_attention_heads,
            head_size=self.attention_head_size,
            out_dim=self.hidden_size,
            key_size=self.attention_key_size,
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % attention_name
        )

        # Feed Forward
        xi = x
        x = self.apply(
            inputs=x,
            layer=FeedForward,
            units=self.intermediate_size,
            activation=self.hidden_act,
            kernel_initializer=self.initializer,
            name=feed_forward_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % feed_forward_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % feed_forward_name
        )
        x = self.apply(
            inputs=self.simplify([x, z]),
            layer=LayerNormalization,
            conditional=(z is not None),
            hidden_units=self.layer_norm_conds[1],
            hidden_activation=self.layer_norm_conds[2],
            hidden_initializer=self.initializer,
            name='%s-Norm' % feed_forward_name
        )

        return x

    def apply_final_layers(self, inputs):
        """根据剩余参数决定输出
        """
        x = inputs
        z = self.layer_norm_conds[0]
        outputs = [x]

        if self.with_pool:
            # Pooler部分（提取CLS向量）
            x = outputs[0]
            x = self.apply(
                inputs=x,
                layer=Lambda,
                function=lambda x: x[:, 0],
                name='Pooler'
            )
            pool_activation = 'tanh' if self.with_pool is True else self.with_pool
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.hidden_size,
                activation=pool_activation,
                kernel_initializer=self.initializer,
                name='Pooler-Dense'
            )
            if self.with_nsp:
                # Next Sentence Prediction部分
                x = self.apply(
                    inputs=x,
                    layer=Dense,
                    units=2,
                    activation='softmax',
                    kernel_initializer=self.initializer,
                    name='NSP-Proba'
                )
            outputs.append(x)

        if self.with_mlm:
            # Masked Language Model部分
            x = outputs[0]
            x = self.apply(
                inputs=x,
                layer=Dense,
                units=self.embedding_size,
                activation=self.hidden_act,
                kernel_initializer=self.initializer,
                name='MLM-Dense'
            )
            x = self.apply(
                inputs=self.simplify([x, z]),
                layer=LayerNormalization,
                conditional=(z is not None),
                hidden_units=self.layer_norm_conds[1],
                hidden_activation=self.layer_norm_conds[2],
                hidden_initializer=self.initializer,
                name='MLM-Norm'
            )
            x = self.apply(
                inputs=x,
                layer=Embedding,
                arguments={'mode': 'dense'},
                name='Embedding-Token'
            )
            x = self.apply(inputs=x, layer=BiasAdd, name='MLM-Bias')
            mlm_activation = 'softmax' if self.with_mlm is True else self.with_mlm
            x = self.apply(
                inputs=x,
                layer=Activation,
                activation=mlm_activation,
                name='MLM-Activation'
            )
            outputs.append(x)

        if len(outputs) == 1:
            outputs = outputs[0]
        elif len(outputs) == 2:
            outputs = outputs[1]
        else:
            outputs = outputs[1:]

        return outputs

class UniLM_Mask(object):
    """定义UniLM的Attention Mask（Seq2Seq模型用）
    其中source和target的分区，由segment_ids来表示。
    UniLM: https://arxiv.org/abs/1905.03197
    """
    def compute_attention_bias(self, inputs=None):
        """通过idxs序列的比较来得到对应的mask
        """
        if self.attention_bias is None:

            def unilm_mask(s):
                idxs = K.cumsum(s, axis=1)
                mask = idxs[:, None, :] <= idxs[:, :, None]
                mask = K.cast(mask, K.floatx())
                return -(1 - mask[:, None]) * infinity()

            self.attention_bias = self.apply(
                inputs=self.inputs[1],
                layer=Lambda,
                function=unilm_mask,
                name='Attention-UniLM-Mask'
            )

        return self.attention_bias

class UnifiedLanguageModel(UniLM_Mask, BERT):
        """带UniLM的Attention Mask的派生模型
        UniLM: https://arxiv.org/abs/1905.03197
        """
        def __init__(self, *args, **kwargs):
            super(UnifiedLanguageModel, self).__init__(*args, **kwargs)
            self.with_mlm = self.with_mlm or True

class Loss(Layer):
    """特殊的层，用来定义复杂loss
    """
    def __init__(self, output_axis=None, **kwargs):
        super(Loss, self).__init__(**kwargs)
        self.output_axis = output_axis

    def call(self, inputs, mask=None):
        loss = self.compute_loss(inputs, mask)
        self.add_loss(loss, inputs=inputs)
        if self.output_axis is None:
            return inputs
        elif isinstance(self.output_axis, list):
            return [inputs[i] for i in self.output_axis]
        else:
            return inputs[self.output_axis]

    def compute_loss(self, inputs, mask=None):
        raise NotImplementedError

    def compute_output_shape(self, input_shape):
        if self.output_axis is None:
            return input_shape
        elif isinstance(self.output_axis, list):
            return [input_shape[i] for i in self.output_axis]
        else:
            return input_shape[self.output_axis]

    def compute_mask(self, inputs, mask):
        if mask is not None:
            if self.output_axis is None:
                return mask
            elif isinstance(self.output_axis, list):
                return [mask[i] for i in self.output_axis]
            else:
                return mask[self.output_axis]

    def get_config(self):
        config = {
            'output_axis': self.output_axis,
        }
        base_config = super(Loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TotalLoss(Loss):
    def compute_loss(self, inputs, mask=None):
        loss1 = self.compute_loss_of_seq2seq(inputs, mask)
        loss2 = self.compute_loss_of_similarity(inputs, mask)
        self.add_metric(loss1, name='seq2seq_loss')
        self.add_metric(loss2, name='similarity_loss')
        return loss1 + loss2

    def compute_loss_of_seq2seq(self, inputs, mask=None):
        y_true, y_mask, _, y_pred = inputs
        y_true = y_true[:, 1:]  
        y_mask = y_mask[:, 1:]  
        y_pred = y_pred[:, :-1]  
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

    def compute_loss_of_similarity(self, inputs, mask=None):
        _, _, y_pred, _ = inputs
        y_true = self.get_labels_of_similarity(y_pred)
        y_pred = K.l2_normalize(y_pred, axis=1)  
        similarities = K.dot(y_pred, K.transpose(y_pred))  
        similarities = similarities - tf.eye(K.shape(y_pred)[0]) * 1e12  
        similarities = similarities * 30
        loss = K.categorical_crossentropy(y_true, similarities, from_logits=True) 
        return loss

    def get_labels_of_similarity(self, y_pred):
        idxs = K.arange(0, K.shape(y_pred)[0])
        idxs_1 = idxs[None, :]
        idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
        labels = K.equal(idxs_1, idxs_2)
        labels = K.cast(labels, K.floatx())
        return labels

def define_uniasm_model(json_data):
    configs = {}
    configs.update(json.loads(json_data))
    if 'max_position' not in configs:
        configs['max_position'] = configs.get('max_position_embeddings', 512)
    if 'dropout_rate' not in configs:
        configs['dropout_rate'] = configs.get('hidden_dropout_prob')
    if 'attention_dropout_rate' not in configs:
        configs['attention_dropout_rate'] = configs.get('attention_probs_dropout_prob')
    if 'segment_vocab_size' not in configs:
        configs['segment_vocab_size'] = configs.get('type_vocab_size', 2)
        
    configs['with_pool'] = True
    transformer = UnifiedLanguageModel(**configs)
    transformer.build(**configs)

    #if checkpoint_path is not None:
    #    transformer.load_weights_from_checkpoint(checkpoint_path)

    outputs = TotalLoss([2, 3])(transformer.model.inputs + transformer.model.outputs)
    train_model = models.Model(transformer.model.inputs, outputs)

    AdamW = extend_with_weight_decay(Adam, 'AdamW')
    optimizer = AdamW(learning_rate=1e-5, weight_decay_rate=0.01)
    train_model.compile(optimizer=optimizer, steps_per_execution=16)

    return train_model
    
# Load the pre-trained UniASM model
def load_weights_uniasm(weights_file, config_json):
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        pretrain_model = define_uniasm_model(config_json)
    if weights_file != "":
        pretrain_model.load_weights(weights_file)
    pretrain_model.trainable = False
    return pretrain_model
    