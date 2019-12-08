#-*-coding:utf-8-*-
import tensorflow as tf
import math
import numpy as np
from tensorflow.contrib import *

def noise_from_step_num():
  """Quantization noise equal to (phi * (step_num + 1)) mod 1.0.
  Not using random_uniform here due to a problem on TPU in that random seeds
  are not respected, which may cause the parameters on different replicas
  to go out-of-sync.
  Returns:
    a float32 scalar
  """
  step = tf.to_int32(tf.train.get_or_create_global_step()) + 1
  phi = ((5 ** 0.5) - 1) / 2
  # Naive computation tf.mod(phi * step, 1.0) in float32 would be disastrous
  # due to loss of precision when the step number gets large.
  # Computation in doubles does not work on TPU, so we use this complicated
  # alternative computation which does not suffer from these roundoff errors.
  ret = 0.0
  for i in range(30):
    ret += (((phi * (2 ** i)) % 1.0)  # double-precision computation in python
            * tf.to_float(tf.mod(step // (2 ** i), 2)))
  return tf.mod(ret, 1.0)

class AdafactorOptimizer(tf.train.Optimizer):
  """Optimizer that implements the Adafactor algorithm.
  Adafactor is described in https://arxiv.org/abs/1804.04235.
  Adafactor is most similar to Adam (Kingma and Ba), the major differences are:
  1. For a two-dimensional AxB weight matrix, Adafactor uses only A+B auxiliary
     parameters to maintain the second-moment estimator, instead of AB.
     This is advantageous on memory-limited systems.  In addition, beta1
     (momentum) is set to zero by default, saving an additional auxiliary
     parameter per weight.  Variables with >=3 dimensions are treated as
     collections of two-dimensional matrices - factorization is over the final
     two dimensions.
  2. Adafactor incorporates "update-clipping" - a scale-invariant analog of
     gradient clipping.  This adds stability
  3. Adafactor does not require an external "learning rate".  By default, it
     incorporates a relative-update-scale schedule, corresponding to
     inverse-square-root learning-rate-decay in ADAM.  We hope this works well
     for most applications.
  ALGORITHM:
  parameter -= absolute_update_scale * clip(grad / grad_scale)
  where:
    absolute_update_scale := relative_update_scale * parameter_scale
    relative_update_scale := min((step_num + 1)**-0.5, 1e-2)
    parameter_scale := max(rms(var)), epsilon2)
    clip(x) := x / max(1.0, rms(x))
    grad_scale := tf.sqrt(v)   (v is the second-moment estimator)
  The second-moment estimator v is maintained in a manner similar to Adam:
  We initialize
  ```
  if var is 2-dimensional:
    v_r <- zeros([num_rows])
    v_c <- zeros([num_cols])
  if var is 0-dimensional or 1-dimensional:
    v <- zeros(shape(var))
  ```
  The update rule is as follows:
  ```
  decay_rate = 1 - (step_num + 1) ^ -0.8
  grad_squared = tf.square(grad) + epsilon1
  if var is 2-dimensional:
    v_r <- decay_rate * v_r + (1 - decay_rate) * reduce_mean(grad_squared, 1)
    v_c <- decay_rate * v_c + (1 - decay_rate) * reduce_mean(grad_squared, 0)
    v = outer_prod(v_r, v_c) / reduce_mean(v_r)
  if var is 0-dimensional or 1-dimensional:
    v <- decay_rate * v + (1 - decay_rate) * grad_squared
  ```
  For variables with >=3 dimensions, we factorize the second-moment accumulator
  over the final 2 dimensions.  See the code for details.
  Several parts of this algorithm are configurable from the initializer.
    multiply_by_parameter_scale:  If True, then compute absolute_update_scale
      as described above.  If False, let absolute_update_scale be the externally
      supplied learning_rate.
    learning_rate: represents relative_update_scale if
      multiply_by_parameter_scale==True, or absolute_update_scale if
      multiply_by_parameter_scale==False.
    decay_rate: Decay rate of the second moment estimator (varies by step_num).
      This should be set to a function such that:
      1-1/(step_num + 1) <= decay_rate(step_num) < 1.0
    beta1: enables momentum, as in Adam.  Uses extra memory if nonzero.
    clipping_threshold: should be >=1.0 or None for no update clipping
    factored: whether to factor the second-moment estimator.  True means
      less memory usage.
  """

  def __init__(self,
               multiply_by_parameter_scale=True,
               learning_rate=None,
               decay_rate=None,
               beta1=0.0,
               clipping_threshold=1.0,
               factored=True,
               simulated_quantize_bits=None,
               parameter_encoding=None,
               use_locking=False,
               name="Adafactor",
               epsilon1=1e-30,
               epsilon2=1e-3):
    """Construct a new Adafactor optimizer.
    See class comment.
    Args:
      multiply_by_parameter_scale: a boolean
      learning_rate: an optional Scalar.
      decay_rate: an optional Scalar.
      beta1: a float value between 0 and 1
      clipping_threshold: an optional float >= 1
      factored: a boolean - whether to use factored second-moment estimator
        for 2d variables
      simulated_quantize_bits: train with simulated quantized parameters
        (experimental)
      parameter_encoding: a ParameterEncoding object to use in the case of
        bfloat16 variables.
      use_locking: If True use locks for update operations.
      name: Optional name for the operations created when applying gradients.
        Defaults to "AdafactorOptimizer".
      epsilon1: Regularization constant for squared gradient.
      epsilon2: Regularization constant for parameter scale.
    Raises:
      ValueError: if absolute_update_scale and relative_update_scale_fn are both
        present or both absent.
    """
    super(AdafactorOptimizer, self).__init__(use_locking, name)
    self._multiply_by_parameter_scale = multiply_by_parameter_scale
    if learning_rate is None:
      learning_rate = self._learning_rate_default(multiply_by_parameter_scale)
    self._learning_rate = learning_rate
    if decay_rate is None:
      decay_rate = self._decay_rate_default()
    self._decay_rate = decay_rate
    self._beta1 = beta1
    self._clipping_threshold = clipping_threshold
    self._factored = factored
    self._simulated_quantize_bits = simulated_quantize_bits
    self._parameter_encoding = parameter_encoding
    self._quantization_noise = noise_from_step_num()
    self._epsilon1 = epsilon1
    self._epsilon2 = epsilon2

  def _should_use_factored_second_moment_estimate(self, shape):
    """Should we use a factored second moment estimator.
    Based on the shape of the variable.
    Args:
      shape: a list of integers
    Returns:
      a boolean
    """
    return self._factored and len(shape) >= 2

  def _create_slots(self, var_list):
    for var in var_list:
      shape = var.get_shape().as_list()
      if self._beta1:
        self._zeros_slot(var, "m", self._name)
      if self._should_use_factored_second_moment_estimate(shape):
        r_val = tf.zeros(shape[:-1], dtype=tf.float32)
        c_val = tf.zeros(shape[:-2] + shape[-1:], dtype=tf.float32)
        self._get_or_make_slot(var, r_val, "vr", self._name)
        self._get_or_make_slot(var, c_val, "vc", self._name)
      else:
        v_val = tf.zeros(shape, dtype=tf.float32)
        self._get_or_make_slot(var, v_val, "v", self._name)

  def _apply_dense(self, grad, var):
    return self._resource_apply_dense(grad, var)

  def _apply_sparse(self, grad, var):
    return self._apply_dense(tf.convert_to_tensor(grad), var)

  def _resource_apply_sparse(self, grad, handle, indices):
    return self._resource_apply_dense(
        tf.convert_to_tensor(tf.IndexedSlices(grad, indices, tf.shape(handle))),
        handle)

  def _parameter_scale(self, var):
    """Estimate the scale of the parameters from the current values.
    We include a minimum value of 0.001 to give it a chance to escape 0
    if it was zero-initialized.
    Instead of using the value, we could impute the scale from the shape,
    as initializers do.
    Args:
      var: a variable or Tensor.
    Returns:
      a Scalar
    """
    return tf.maximum(reduce_rms(var), self._epsilon2)

  def _resource_apply_dense(self, grad, handle):
    var = handle
    grad = tf.to_float(grad)
    grad_squared = tf.square(grad) + self._epsilon1
    grad_squared_mean = tf.reduce_mean(grad_squared)
    decay_rate = self._decay_rate
    update_scale = self._learning_rate
    old_val = var
    if var.dtype.base_dtype == tf.bfloat16:
      old_val = tf.to_float(self._parameter_encoding.decode(old_val))
    if self._multiply_by_parameter_scale:
      update_scale *= tf.to_float(self._parameter_scale(old_val))
    # HACK: Make things dependent on grad.
    # This confounds the XLA rewriter and keeps it from fusing computations
    # across different variables.  This fusion is a bad for HBM usage, since
    # it causes the gradients to persist in memory.
    decay_rate += grad_squared_mean * 1e-30
    update_scale += grad_squared_mean * 1e-30
    # END HACK
    mixing_rate = 1.0 - decay_rate
    shape = var.get_shape().as_list()
    updates = []
    if self._should_use_factored_second_moment_estimate(shape):
      grad_squared_row_mean = tf.reduce_mean(grad_squared, -1)
      grad_squared_col_mean = tf.reduce_mean(grad_squared, -2)
      vr = self.get_slot(var, "vr")
      new_vr = (decay_rate * vr + mixing_rate * grad_squared_row_mean)
      vc = self.get_slot(var, "vc")
      new_vc = (decay_rate * vc + mixing_rate * grad_squared_col_mean)
      vr_update = tf.assign(vr, new_vr, use_locking=self._use_locking)
      vc_update = tf.assign(vc, new_vc, use_locking=self._use_locking)
      updates = [vr_update, vc_update]
      long_term_mean = tf.reduce_mean(new_vr, -1, keepdims=True)
      r_factor = tf.rsqrt(new_vr / long_term_mean)
      c_factor = tf.rsqrt(new_vc)
      x = grad * tf.expand_dims(r_factor, -1) * tf.expand_dims(c_factor, -2)
    else:
      v = self.get_slot(var, "v")
      new_v = decay_rate * v + mixing_rate * grad_squared
      v_update = tf.assign(v, new_v, use_locking=self._use_locking)
      updates = [v_update]
      x = grad * tf.rsqrt(new_v)
    if self._clipping_threshold is not None:
      clipping_denom = tf.maximum(1.0, reduce_rms(x) / self._clipping_threshold)
      x /= clipping_denom
    subtrahend = update_scale * x
    if self._beta1:
      m = self.get_slot(var, "m")
      new_m = self._beta1 * tf.to_float(m) + (1.0 - self._beta1) * subtrahend
      subtrahend = new_m
      new_m = common_layers.cast_like(new_m, var)
      updates.append(tf.assign(m, new_m, use_locking=self._use_locking))
    new_val = tf.to_float(old_val) - subtrahend
    if var.dtype.base_dtype == tf.bfloat16:
      new_val = self._parameter_encoding.encode(
          new_val, self._quantization_noise)
    if self._simulated_quantize_bits:
      new_val = quantization.simulated_quantize(
          var - subtrahend, self._simulated_quantize_bits,
          self._quantization_noise)
    var_update = tf.assign(var, new_val, use_locking=self._use_locking)
    updates = [var_update] + updates
    return tf.group(*updates)

  def _decay_rate_default(self):
    return adafactor_decay_rate_pow(0.8)

  def _learning_rate_default(self, multiply_by_parameter_scale):
    learning_rate = tf.minimum(tf.rsqrt(step_num() + 1.0), 0.01)
    if not multiply_by_parameter_scale:
      learning_rate *= 0.05
    return learning_rate


def adafactor_decay_rate_adam(beta2):
  """Second-moment decay rate like Adam, subsuming the correction factor.
  Args:
    beta2: a float between 0 and 1
  Returns:
    a scalar
  """
  t = tf.to_float(tf.train.get_or_create_global_step()) + 1.0
  decay = beta2 * (1.0 - tf.pow(beta2, t - 1.0)) / (1.0 - tf.pow(beta2, t))
  # decay = tf.cond(tf.equal(t, 1.0), lambda: beta2, lambda: decay)
  return decay


def adafactor_decay_rate_pow(exponent):
  """Second moment decay rate where memory-length grows as step_num^exponent.
  Args:
    exponent: a float between 0 and 1
  Returns:
    a scalar
  """
  return 1.0 - tf.pow((step_num() + 1.0), -exponent)


def step_num():
  return tf.to_float(tf.train.get_or_create_global_step())


def adafactor_optimizer_from_hparams(hparams, lr):
  """Create an Adafactor optimizer based on model hparams.
  Args:
    hparams: model hyperparameters
    lr: learning rate scalar.
  Returns:
    an AdafactorOptimizer
  Raises:
    ValueError: on illegal values
  """
  if hparams.optimizer_adafactor_decay_type == "adam":
    decay_rate = adafactor_decay_rate_adam(
        hparams.optimizer_adafactor_beta2)
  elif hparams.optimizer_adafactor_decay_type == "pow":
    decay_rate = adafactor_decay_rate_pow(
        hparams.optimizer_adafactor_memory_exponent)
  else:
    raise ValueError("unknown optimizer_adafactor_decay_type")
  if hparams.weight_dtype == "bfloat16":
    parameter_encoding = quantization.EighthPowerEncoding()
  else:
    parameter_encoding = None
  return AdafactorOptimizer(
      multiply_by_parameter_scale=(
          hparams.optimizer_adafactor_multiply_by_parameter_scale),
      learning_rate=lr,
      decay_rate=decay_rate,
      beta1=hparams.optimizer_adafactor_beta1,
      clipping_threshold=hparams.optimizer_adafactor_clipping_threshold,
      factored=hparams.optimizer_adafactor_factored,
      simulated_quantize_bits=getattr(
          hparams, "simulated_parameter_quantize_bits", 0),
      parameter_encoding=parameter_encoding,
      use_locking=False,
      name="Adafactor")


def reduce_rms(x):
  return tf.sqrt(tf.reduce_mean(tf.square(x)))

class code_gen_model:
    def gelu(self, x):                                    
        #return tf.nn.tanh(x)
        cdf = 0.5 * (1.0 + tf.tanh(                                                                        
            (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))                                                          
        return x * cdf

    def weights_nonzero(self, labels):
    #"""Assign weight 1.0 to all labels except for padding (id=0)."""
        return tf.to_float(tf.not_equal(labels, 0))

    def weights_zero(self, labels):
    #"""Assign weight 1.0 to all labels except for padding (id=0)."""
        return tf.to_float(tf.equal(labels, 0))


    def mask_from_embedding(self, emb):
    #"""Input embeddings -> padding mask.
    #We have hacked symbol_modality to return all-zero embeddings for padding.
    #Returns a mask with 0.0 in the padding positions and 1.0 elsewhere.
    #Args:
    #    emb: a Tensor with shape [batch, width, height, depth].
    #Returns:
    #    a 0.0/1.0 Tensor with shape [batch, width, height, 1].
    #"""
        return self.weights_nonzero(tf.reduce_sum(tf.abs(emb), axis=3, keepdims=True))

    def headattention(self, Q, K, V, mask_k, flag, antimask):
        d = int(Q.shape[2])
        d = math.sqrt(float(d))
        matrix = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / d
        if flag == False:
            mask = tf.expand_dims(mask_k, -2)
            #mask = tf.tile(mask, [1, 1, 1])
        else:
            if len(antimask.get_shape()) != 3:
                antimask = tf.expand_dims(antimask, 0)
            mask = antimask
        a = matrix * mask
        ma = self.weights_zero(a) * -1e18
        a += ma
        a = tf.nn.softmax(a)
        a *= mask
        return tf.matmul(a, V)

    def mul(self, Q, K):
        return tf.einsum("ijk,kl->ijl", Q, K)
    
    
    def multiheadattention_QKV_2(self, Query, Keys, Values, mask, flag=False, antimask=""):
        #m = int(Values.shape[1])
        d = int(Query.shape[2])
        list_concat = []
        heads = 8
        qd = math.sqrt(float(d // heads))
        for i in range(heads):
            W_q = tf.layers.dense(Query, d//heads, name="qkv2headq" + str(i), use_bias=False)#self.weight_variable(shape=[d, d // k])
            W_kv = tf.layers.dense(Keys, d//heads, name="qkv2headkv" + str(i), use_bias=False)
            W_k = tf.layers.dense(Keys, d//heads, name="qkv2headk" + str(i), use_bias=False)#self.weight_variable(shape=[d, d // k])
            W_v = tf.layers.dense(Values, d//heads, name="qkv2headv" + str(i), use_bias=False)#self.weight_variable(shape=[d, d // k])
            W_vv = tf.layers.dense(Values, d//heads, name="qkv2headvv" + str(i), use_bias=False)#self.weight_variable(shape=[d, d // k])
            QK = tf.reduce_sum(W_q * W_k, -1, keepdims=True) / qd
            QV = tf.reduce_sum(W_q * W_v, -1, keepdims=True) / qd
            QK_1 = QK - tf.maximum(QK, QV)
            QV_1 = QV - tf.maximum(QK, QV)
            self.probe = QV
            QK = tf.exp(QK_1)
            QV = tf.exp(QV_1)
            QK_S = QK / (QK + QV)
            QV_S = QV / (QK + QV)
            QK_S *= W_kv
            QV_S *= W_vv
            list_concat.append(QK_S + QV_S)#self.headattention_qkv(W_q, W_k, W_v, mask, flag, antimask))
        concat_head = tf.concat(list_concat, -1)
        W_o = tf.layers.dense(concat_head, d, name="qkv2head", use_bias=False)#self.weight_variable(shape=[int(concat_head.shape[2]), d])
        return W_o#self.mul(concat_head, W_o)

    def multiheadattention(self, H, mask, k, flag=False, antimask="", use_posi_att=False, posi_embedding=""):
        m = int(H.shape[1])
        d = int(H.shape[2])
        list_concat = []
        #W_o = self.weight_variable(shape=[d, d])
        for i in range(k):
            W_q = tf.layers.dense(H, d//k, name="headq" + str(i), use_bias=False)#self.weight_variable(shape=[d, d // k])
            W_k = tf.layers.dense(H, d//k, name="headk" + str(i), use_bias=False)#self.weight_variable(shape=[d, d // k])
            W_v = tf.layers.dense(H, d//k, name="headv" + str(i), use_bias=False)#self.weight_variable(shape=[d, d // k])
            if use_posi_att:
                #if not flag:
                #    list_concat.append(self.headattention_position(W_q, W_k, W_v, mask, tf.layers.dense(posi_embedding, d//k, name="headpos" + str(i))))    
                #else:
                #with tf.device("/device:GPU:1"):
                posi = tf.layers.dense(posi_embedding, d//k, name="headqp" + str(i))
                posi_k = tf.layers.dense(posi_embedding, d//k, name="headkp" + str(i))
                posi_v = tf.layers.dense(posi_embedding, d//k, name="headkv" + str(i))
                #with tf.device("/cpu:" + str(i + 1 + 10)):
                list_concat.append(self.headattention_position(W_q, W_k, W_v, antimask, posi, posi_k, posi_v, flag, mask))    
            else:
                list_concat.append(self.headattention(W_q, W_k, W_v, mask, flag, antimask))
        concat_head = tf.concat(list_concat, -1)
        W_o = tf.layers.dense(concat_head, d, name="head", use_bias=False)#self.weight_variable(shape=[int(concat_head.shape[2]), d])
        return W_o#self.mul(concat_head, W_o)

    def multiheadattention_QKV_Copy(self, p, Query, Keys, Values, mask):
        m = int(Values.shape[1])
        d = int(Values.shape[2])
        list_concat = []
        #W_o = self.weight_variable(shape=[d, d])
        heads = k = 1
        for i in range(heads):
            W_q = tf.layers.dense(Query, d//k, name="qkv_headq" + str(i), use_bias=False)#self.weight_variable(shape=[d, d // k])
            W_k = tf.layers.dense(Keys, d//k, name="qkv_headk" + str(i), use_bias=False)#self.weight_variable(shape=[d, d // k])
            #W_v = tf.layers.dense(Values, d)#self.weight_variable(shape=[d, d // k])
            W_q = tf.expand_dims(W_q, 2)
            W_k = tf.expand_dims(W_k, 1)
            return tf.reduce_sum(tf.layers.dense(tf.tanh(W_q + W_k), 1) , -1)
            list_concat.append(tf.reduce_sum(W_q * W_k, -1))
            #self.headattention_copy(W_q, W_k, None, mask)
        concat_head = tf.concat(list_concat, -1)
        #W_o = tf.layers.dense(concat_head, 1, name="copy", use_bias=False)
        #W_o = tf.layers.dense(concat_head, 1, name="copy")
        #W_o = tf.reduce_max(W_o, reduction_indices=[-1])#self.weight_variable(shape=[int(concat_head.shape[2]), d])
        return W_o#self.mul(concat_head, W_o)


    def multiheadattention_QKV(self, Query, Keys, Values, mask, flag=False, antimask="", use_posi_att=False, posi_embedding=""):
        m = int(Values.shape[1])
        d = int(Values.shape[2])
        list_concat = []
        #W_o = self.weight_variable(shape=[d, d])
        heads = 8
        for i in range(heads):
            W_q = tf.layers.dense(Query, d//heads, name="qkv_headq" + str(i), use_bias=False)#self.weight_variable(shape=[d, d // k])
            W_k = tf.layers.dense(Keys, d//heads, name="qkv_headk" + str(i), use_bias=False)#self.weight_variable(shape=[d, d // k])
            W_v = tf.layers.dense(Values, d//heads, name="qkv_headv" + str(i), use_bias=False)#self.weight_variable(shape=[d, d // k])
            list_concat.append(self.headattention(W_q, W_k, W_v, mask, flag, antimask))
        concat_head = tf.concat(list_concat, -1)
        W_o = tf.layers.dense(concat_head, d, name="qkv_head", use_bias=False)#self.weight_variable(shape=[int(concat_head.shape[2]), d])
        return W_o#self.mul(concat_head, W_o)


    def drop(self, input, drop_rate=0.4):
        return tf.nn.dropout(input, self.keep_prob)

    def layer_norm(self, vec, na=None, axis=2):
        return tf.contrib.layers.layer_norm(vec, scope=na, begin_norm_axis=axis, reuse=None)
    
        

    def sepconv(self, state, size, mask):
        state = self.drop(tf.layers.separable_conv1d(tf.expand_dims(mask, -1) * self.drop(tf.layers.separable_conv1d(state, size, 3, activation=self.gelu, padding="SAME", name="conv")), size, 3, padding="SAME", name="dense_2") + state)
        return state

    def get_timing_signal_1d(self, length,
                             channels,
                             min_timescale=1.0,
                             max_timescale=1.0e4,
                             start_index=0):
          position = tf.to_float(tf.range(length) + start_index)
          num_timescales = channels // 2
          log_timescale_increment = (
              math.log(float(max_timescale) / float(min_timescale)) /
              tf.maximum(tf.to_float(num_timescales) - 1, 1))
          inv_timescales = min_timescale * tf.exp(
              tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
          scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
          signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
          signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
          signal = tf.reshape(signal, [1, length, channels])
          return signal

    def nl_reader(self, state, embedding_size, step, halting_probability, remainders, n_updates, previous_state, em_char, mask, mask1):
        size = embedding_size
        i = step
        state_shape_static = state.get_shape()
        state += self.get_timing_signal_1d(tf.shape(state)[1], tf.shape(state)[2], start_index=0) + self.get_timing_signal_1d(tf.shape(state)[1], tf.shape(state)[2], start_index=step)#self.position_embedding(state, i)
        state = self.layer_norm(self.drop(self.multiheadattention_QKV(state, state, state, mask) + state), "norm1")
        with tf.variable_scope("Char_Att", reuse=None):
            state = self.layer_norm(self.drop(self.multiheadattention_QKV_2(state, state, em_char, mask, False, "") + state), "norm2")
            #self.probe = state
            #state += em_char
            #state = self.layer_norm(self.drop(self.multiheadattention(state, mask, 8) + state), "norm1")
        #with tf.variable_scope("Char_Att", reuse=None):
        #    state = self.layer_norm(self.drop(self.multiheadattention_QKV_2(state, em_char, em_char, mask, False, "") + state), "norm2")
        state *= tf.expand_dims(mask, -1)
        #print (state)
        with tf.variable_scope("Dense", reuse=None):
            state = self.sepconv(state, self.embedding_size, mask)
            #state = self.drop(tf.layers.separable_conv1d(tf.expand_dims(mask, -1) * self.drop(tf.layers.separable_conv1d(state, size, 3, activation=self.gelu, padding="SAME", name="conv")), size, 5, padding="SAME", name="dense_2") + state)
        state = self.layer_norm(state, "norm3")
        new_state = previous_state
        step += 1
        return (state, embedding_size, step, halting_probability, remainders, n_updates, new_state, em_char, mask, mask)

    def sepconv_A (self, state, A, kernel):
        #ori = state
        l = [state]
        now = state
        for i in range(kernel - 1):
            now = tf.transpose(tf.matmul(now, A, transpose_a=True), [0, 2, 1])
            l.append(now)
        state = tf.stack(l, 2)
        state = self.drop(tf.layers.separable_conv2d(state, self.embedding_size, [1, kernel], name="separ_1"))
        state = self.gelu(state)
        state = tf.reduce_max(state, 2)
        l = [state]
        now = state
        for i in range(kernel - 1):
            now = tf.transpose(tf.matmul(now, A, transpose_a=True), [0, 2, 1])
            l.append(now)
        state = tf.stack(l, 2)
        #state = tf.stack([state, tf.transpose(tf.matmul(state, A, transpose_a=True), [0, 2, 1])], 2)
        state = tf.layers.separable_conv2d(state, self.embedding_size, [1, kernel], name="separ_2")
        #state = self.gelu(state)
        state = tf.reduce_max(state, 2)
        print (state.get_shape())
        return state

    def ast_reader(self, state, embedding_size, step, halting_probability, remainders, n_updates, previous_state, Decoder1, nl_conv, state_ast, ):
        size = embedding_size
        i = step
        state_shape_static = state.get_shape()
        em_Rule_Type = Decoder1
        Decoder = state
       
        Decoder += self.em_depth + self.get_timing_signal_1d(tf.shape(state)[1], tf.shape(state)[2], start_index=0) + self.get_timing_signal_1d(tf.shape(state)[1], tf.shape(state)[2], start_index=step)#self.position_embedding(state, i)
        Decoder = self.layer_norm(self.drop(self.multiheadattention_QKV(Decoder, Decoder, Decoder, self.mask_rule, True, self.antimask) + Decoder), "norm1")
        with tf.variable_scope("TP_R", reuse=None):
            Decoder = self.layer_norm(self.drop(self.multiheadattention_QKV_2(Decoder, Decoder, em_Rule_Type, self.mask_rule, False, self.antimask) + Decoder), "norm3") 
        with tf.variable_scope("NL_ATT", reuse=None):
            Decoder =self.layer_norm(self.drop(Decoder + self.multiheadattention_QKV(Decoder, nl_conv, nl_conv, self.mask_nl)))
        Decoder = self.drop(self.sepconv_A(Decoder, self.tree_A, 3))
        state = Decoder = self.layer_norm(Decoder, "norm4") 
        new_state = previous_state
        step += 1
        return (state, embedding_size, step, halting_probability, remainders, n_updates, new_state, Decoder1, nl_conv, state_ast)

    def query_decoder(self, state, embedding_size, step, halting_probability, remainders, n_updates, previous_state, ast_p, nl_conv, tree_path):
        size = embedding_size
        i = step
        state_shape_static = state.get_shape()
        Decoder = state
        
        Decoder = self.layer_norm(self.drop(self.multiheadattention_QKV(Decoder, ast_p, ast_p, self.mask_rule, True, self.antimask) + Decoder), "norm1")
        with tf.variable_scope("TP_R", reuse=None):
            Decoder = self.layer_norm(self.drop(self.multiheadattention_QKV(Decoder, nl_conv, nl_conv, self.mask_nl, False, self.antimask) + Decoder), "norm3")
        Decoder = self.drop(tf.layers.dense(self.drop(self.gelu(tf.layers.dense(Decoder, self.embedding_size * 4, name="decode2"))), self.embedding_size, name="decode1") + Decoder)
        state = Decoder = self.layer_norm(Decoder, "norm4")
        step += 1
        new_state = previous_state
        return (state, embedding_size, step, halting_probability, remainders, n_updates, new_state, ast_p, nl_conv, tree_path)

    def transf(self, em_NL, mask, layers, position=False, size=None, em_char=""):
        if size is None:
            size = int(em_NL.shape[2])
        state = em_NL
        state_slice = slice(0, 2)
        update_shape = tf.shape(state)[state_slice]
        halting_probability = tf.zeros(update_shape, name="halting_probability")
        remainders = tf.zeros(update_shape, name="remainder")
        n_updates = tf.zeros(update_shape, name="n_updates")
        previous_state = tf.zeros_like(state, name="previous_state")
        step = tf.constant(0, dtype=tf.int32)

        for i in range(int(self.max_steps)):
            with tf.variable_scope("NL_CONV" + str(i), reuse=None):
                (state, size, step, halting_probability, remainders, n_updates, previous_state, em_char, mask, mask) = self.nl_reader(state, size, step, halting_probability, remainders, n_updates, previous_state, em_char, mask, mask)
        #self.probe = state
        return state

    def cal_pgen(self, x, y):
        height = int(x.shape[2])
        width = int(y.shape[2])
        w_matrix = self.weight_variable(shape=[height, width])
        y = tf.transpose(y, [0, 2, 1])
        tmp = tf.einsum("ijk,kl->ijl", x, w_matrix)
        same_f = tf.matmul(tmp, y)
        same_f = tf.reduce_max(same_f, reduction_indices=[2])
        same_f = tf.nn.sigmoid(same_f)
        return same_f

    def __init__(self, classnum, embedding_size, conv_layernum, conv_layersize, rnn_layernum,
                 batch_size, NL_vocabu_size, Tree_vocabu_size, NL_len, Tree_len, parent_len, learning_rate, keep_prob, Char_vocabu_size, rules_len):
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.vocabu_size = NL_vocabu_size
        self.NL_len = NL_len
        self.Tree_len = Tree_len
        self.conv_layernum = conv_layernum
        self.conv_layersize = conv_layersize
        self.learning_rate = learning_rate
        self.BatchNormalization = tf.layers.batch_normalization
        self.Relu = self.gelu
        self.Conv1d = tf.layers.conv1d
        self.rnn_layernum = rnn_layernum
        self.layernum = 3
        self.layerparentlist = 3
        self.class_num = classnum
        self.n_stages = 5
        self.steps = 0.
        act_epsilon = 0.01
        self.max_steps = 5
        self.global_step=tf.Variable(1, trainable=False, name="global_step") 
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)
        self.input_NL = tf.placeholder(tf.int32, shape=[None, NL_len])
        self.mask_nl = self.weights_nonzero(self.input_NL)
        self.input_NLChar = tf.placeholder(tf.int32, shape=[None, NL_len, 10])
        self.inputY_Num = tf.placeholder(tf.int32, shape=[None, rules_len])
        self.loss_mask = tf.placeholder(tf.float32, shape=[None, rules_len])
        loss_mask = self.loss_mask#self.weights_nonzero(self.inputY_Num)
    
        self.inputY = tf.one_hot(self.inputY_Num, self.class_num)
        self.inputparentlist = tf.placeholder(tf.int32, shape = [None, parent_len])
        self.inputrulelist = tf.placeholder(tf.int32, shape = [None, rules_len])
        self.state = tf.placeholder(tf.int32, shape = [None])
        self.tree_path_vec = tf.placeholder(tf.int32, shape=[None, rules_len, 10])
        self.mask_rule = self.weights_nonzero(self.inputrulelist)
        self.mask_rule_de = self.weights_zero(self.inputrulelist)
        self.inputunderfunclist = tf.placeholder(tf.int32, shape=[None,1])
        self.rewards = tf.placeholder(tf.float32, shape=[None])
        self.inputrulelistnode = tf.placeholder(tf.int32, shape = [None, rules_len])
        self.inputrulelistson = tf.placeholder(tf.int32, shape = [None, rules_len, 10])
        self.antimask = tf.placeholder(tf.float32, shape = [rules_len, rules_len])
        self.sitemask = tf.placeholder(tf.float32, shape = [rules_len, rules_len])
        self.treemask = tf.placeholder(tf.float32, shape = [None, rules_len, rules_len])
        self.father_mat = tf.placeholder(tf.float32, shape=[None, rules_len, rules_len])
        self.labels = tf.placeholder(tf.int32, shape=[None, rules_len])
        self.depth = self.labels
        label_smoothing = 0
        smooth_positives = 1.0 - label_smoothing
        smooth_negatives = label_smoothing / classnum
        self.inputY = self.inputY * smooth_positives + smooth_negatives
        
        self.embedding = tf.get_variable("embedding", [NL_vocabu_size , embedding_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))
        self.char_embedding = tf.get_variable("char_embedding", [Char_vocabu_size, embedding_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))
        self.Tree_embedding = tf.get_variable("Tree_embedding", [Tree_vocabu_size, embedding_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))
        self.Rule_embedding = tf.get_variable("Rule_embedding", [classnum + 10, embedding_size], dtype=tf.float32, initializer=tf.random_uniform_initializer(-math.sqrt(3), math.sqrt(3)))
        channels = embedding_size
        self.depth_embedding = tf.get_variable("Depth_embedding", [40, embedding_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=channels**-0.5)) * (channels**0.5)
        em_NL = tf.nn.embedding_lookup(self.embedding, self.input_NL)
        em_Char = tf.nn.embedding_lookup(self.char_embedding, self.input_NLChar)
        em_Rule_List = tf.nn.embedding_lookup(self.Rule_embedding, self.inputrulelist)
        self.em_depth = tf.nn.embedding_lookup(self.depth_embedding, self.depth)
        em_Rule_List = tf.nn.embedding_lookup(self.Rule_embedding, self.inputrulelist)
        em_Rule_Node = tf.nn.embedding_lookup(self.Tree_embedding, self.inputrulelistnode)
        em_Rule_Son = tf.nn.embedding_lookup(self.Tree_embedding, self.inputrulelistson)
        em_Tree_Path = tf.nn.embedding_lookup(self.Tree_embedding, self.tree_path_vec)
        self.tree_A = tf.transpose(self.treemask, [0, 2, 1])


        em_Tree_Conv = self.drop(tf.layers.conv2d(em_Tree_Path, embedding_size, [1, 10]))
        em_Tree_Conv = tf.reduce_max(em_Tree_Conv, reduction_indices=[-2])
        em_Tree_Conv = self.layer_norm(em_Tree_Conv)
        em_Tree_Path = em_Tree_Conv#self.layer_norm(self.drop(em_Tree_Path))
        with tf.variable_scope("char_embedding", reuse=None): 
            em_char_conv = self.drop(tf.layers.conv2d(em_Char, embedding_size, [1, 10], name="char_ebd"))
            em_char = tf.reduce_max(em_char_conv, reduction_indices=[-2])
            em_char = self.layer_norm(em_char)

        
        em_conv = self.drop(tf.layers.conv2d(em_Rule_Son, embedding_size, [1, 10]))
        em_conv = tf.reduce_max(em_conv, reduction_indices=[-2])
        em_conv = self.layer_norm(em_conv)
        em_Rule_Type = tf.layers.conv2d(tf.stack([em_Rule_Node, em_Rule_List, em_conv], -2), embedding_size, [1, 3])
        em_Rule_Type = tf.reduce_max(em_Rule_Type, reduction_indices=[-2])
        em_Rule_Type = self.layer_norm(self.drop(em_Rule_Type))

        # Encoder
        with tf.variable_scope("Q_conv", reuse=None):
            nl_conv = self.transf(em_NL, self.mask_nl, 3, True, embedding_size, em_char)
        # Decoder
        with tf.variable_scope("RL_conv", reuse=None):
            Decoder = em_Rule_List
            antimask = self.antimask
            just_time = False 
            # Tree Reader
            state = em_Rule_List
            state_slice = slice(0, 2)
            update_shape = tf.shape(state)[state_slice]
            halting_probability = tf.zeros(update_shape, name="halting_probability")
            remainders = tf.zeros(update_shape, name="remainder")
            n_updates = tf.zeros(update_shape, name="n_updates")
            previous_state = tf.zeros_like(state, name="previous_state")
            step = tf.constant(0, dtype=tf.int32)
            copy = tf.zeros([tf.shape(state)[0], tf.shape(state)[1], tf.shape(em_NL)[1]], name="copy_prev")
            for i in range(int(self.max_steps - 1)):
                with tf.variable_scope("AST_READER" + str(i), reuse=None):
                    (state, self.embedding_size, step, halting_probability, remainders, n_updates, previous_state, em_Rule_Type, nl_conv, em_Rule_List) = self.ast_reader(state, self.embedding_size, step, halting_probability, remainders, n_updates, previous_state, em_Rule_Type, nl_conv, em_Rule_List)
            Decoder = state
            with tf.variable_scope("TBCNN_TP"):
                f_state = em_Tree_Path#tf.matmul(state, self.father_A, transpose_a=True)
            state = Decoder
            with tf.variable_scope("TBCNN"):
                state = self.drop(self.sepconv_A(Decoder, self.tree_A, 3))
            state = self.layer_norm(state, "state_change_1")
            Decoder = f_state
            for i in range(int(self.max_steps - 1)):
                with tf.variable_scope("DECODER", reuse=None):
                    with tf.variable_scope("QUERY_DECODER" + str(i), reuse=None):
                        (f_state, self.embedding_size, step, halting_probability, remainders, n_updates, previous_state, state, nl_conv, em_Tree_Path) = self.query_decoder(f_state, self.embedding_size, step, halting_probability, remainders, n_updates, previous_state, state, nl_conv, em_Tree_Path)
            
            Decoder = f_state

        All_q_a = tf.layers.dense(Decoder, classnum - NL_len)
        self.y_result = tf.nn.softmax(All_q_a)
        copy = self.multiheadattention_QKV_Copy(All_q_a, Decoder, nl_conv, nl_conv, self.mask_nl)

        All_q_a = All_q_a
        W_o = copy
        ma = self.mask_nl
        W_o *= tf.expand_dims(ma, 1)
        W_o = tf.exp(W_o -  tf.reduce_max(W_o, reduction_indices=[-1], keepdims=True))
        ma = tf.expand_dims(self.mask_nl, 1)
        W_o *= ma
        W_o = W_o / tf.reduce_sum(W_o, reduction_indices=[-1], keepdims=True)

        copy_output = W_o#tf.nn.softmax(copy)
        P_gen = tf.layers.dense(Decoder, 1, activation=tf.nn.sigmoid)
        copy_output *= 1 - P_gen
        self.y_result *= P_gen
        self.y_result = tf.concat([self.y_result, copy_output], 2)
        self.max_res = tf.argmax(self.y_result, 2)
        self.correct_prediction = tf.cast(tf.equal(tf.argmax(self.y_result, 2), tf.argmax(self.inputY, 2)), tf.float32) * loss_mask
        self.accuracy = tf.reduce_mean(self.correct_prediction * tf.expand_dims( rules_len / tf.reduce_sum(loss_mask, reduction_indices=[1]), -1))
        self.cross_entropy = tf.reduce_sum(tf.reduce_sum(loss_mask *
            -tf.reduce_sum(self.inputY * tf.log(tf.clip_by_value(self.y_result, 1e-10, 1.0)), reduction_indices=[2]), reduction_indices=[1])) / tf.reduce_sum(loss_mask, reduction_indices=[0, 1])
        tf.add_to_collection("losses", self.cross_entropy)

        self.loss = self.cross_entropy 
        self.params = [param for param in tf.trainable_variables()]
        global_step = tf.cast(self.global_step, dtype=tf.float32)
        self.optim = AdafactorOptimizer().minimize(self.loss , global_step=self.global_step)
