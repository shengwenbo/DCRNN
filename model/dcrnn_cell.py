from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell

from lib import utils


class DCGRUCell(RNNCell):
    """Graph Convolution Gated Recurrent Unit cell.
    """

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units, adj_mx, max_diffusion_step, num_nodes, nheads=None, hid_units=None, split_parts=None, ffd_drop=None, attn_drop=None, num_proj=None,
                 activation=tf.nn.tanh, reuse=None, filter_type="laplacian", use_gc_for_ru=False):
        """

        :param num_units:
        :param adj_mx:
        :param max_diffusion_step:
        :param num_nodes:
        :param input_size:
        :param num_proj:
        :param activation:
        :param reuse:
        :param filter_type: "laplacian", "random_walk", "dual_random_walk".
        :param use_gc_for_ru: whether to use Graph convolution to calculate the reset and update gates.
        """
        super(DCGRUCell, self).__init__(_reuse=reuse)
        self._activation = activation
        self._num_nodes = num_nodes
        self._num_proj = num_proj
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._supports = []
        self._use_gc_for_ru = use_gc_for_ru
        supports = []
        if filter_type == "dense_laplacian":
            supports.append(utils.calculate_adj_list(adj_mx, max_nei=32, lambda_max=None))
            self.n_heads = nheads
            self.hid_units = hid_units
            self.split_parts = split_parts
            self.ffd_drop = ffd_drop
            self.attn_drop = attn_drop
        elif filter_type == "laplacian":
            supports.append(utils.calculate_scaled_laplacian(adj_mx, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
        elif filter_type == "dual_random_walk":
            supports.append(utils.calculate_random_walk_matrix(adj_mx).T)
            supports.append(utils.calculate_random_walk_matrix(adj_mx.T).T)
        else:
            supports.append(utils.calculate_scaled_laplacian(adj_mx))

        if filter_type != "dense_laplacian":
            for support in supports:
                self._supports.append(self._build_sparse_matrix(support))
        else:
            self._supports = supports

    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        output_size = self._num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self._num_nodes * self._num_proj
        return output_size

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with Graph Convolution.
        :param inputs: (B, num_nodes * input_dim)

        :return
        - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
        - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`
        """
        with tf.variable_scope(scope or "dcgru_cell"):
            with tf.variable_scope("gates"):  # Reset gate and update gate.
                output_size = 2 * self._num_units
                # We start with bias of 1.0 to not reset and not update.
                if self._use_gc_for_ru:
                    fn = self._gat
                else:
                    fn = self._fc
                value = tf.nn.sigmoid(fn(inputs, state, output_size, bias_start=1.0))
                value = tf.reshape(value, (-1, self._num_nodes, output_size))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)
                r = tf.reshape(r, (-1, self._num_nodes * self._num_units))
                u = tf.reshape(u, (-1, self._num_nodes * self._num_units))
            with tf.variable_scope("candidate"):
                c = self._gat(inputs, r * state, self._num_units)
                if self._activation is not None:
                    c = self._activation(c)
            output = new_state = u * state + (1 - u) * c
            if self._num_proj is not None:
                with tf.variable_scope("projection"):
                    w = tf.get_variable('w', shape=(self._num_units, self._num_proj))
                    batch_size = inputs.get_shape()[0].value
                    output = tf.reshape(new_state, shape=(-1, self._num_units))
                    output = tf.reshape(tf.matmul(output, w), shape=(batch_size, self.output_size))
        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        dtype = inputs.dtype
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = tf.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=-1)
        input_size = inputs_and_state.get_shape()[-1].value
        weights = tf.get_variable(
            'weights', [input_size, output_size], dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer())
        value = tf.nn.sigmoid(tf.matmul(inputs_and_state, weights))
        biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                 initializer=tf.constant_initializer(bias_start, dtype=dtype))
        value = tf.nn.bias_add(value, biases)
        return value

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        """Graph convolution between input and the graph matrix.

        :param args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        :param output_size:
        :param bias:
        :param bias_start:
        :param scope:
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = tf.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value
        dtype = inputs.dtype

        x = inputs_and_state
        x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, total_arg_size, batch_size)
        x0 = tf.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = tf.expand_dims(x0, axis=0)

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            if self._max_diffusion_step == 0:
                pass
            else:
                for support in self._supports:
                    x1 = tf.sparse_tensor_dense_matmul(support, x0)
                    x = self._concat(x, x1)

                    for k in range(2, self._max_diffusion_step + 1):
                        x2 = 2 * tf.sparse_tensor_dense_matmul(support, x1) - x0
                        x = self._concat(x, x2)
                        x1, x0 = x2, x1

            num_matrices = len(self._supports) * self._max_diffusion_step + 1  # Adds for x itself.
            x = tf.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
            x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_size, order)
            x = tf.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])

            weights = tf.get_variable(
                'weights', [input_size * num_matrices, output_size], dtype=dtype,
                initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._num_nodes, output_size)

            biases = tf.get_variable("biases", [output_size], dtype=dtype,
                                     initializer=tf.constant_initializer(bias_start, dtype=dtype))
            x = tf.nn.bias_add(x, biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return tf.reshape(x, [batch_size, self._num_nodes * output_size])

    def _gat(self, inputs, state, output_size, bias_start=0.0):
        """Graph attention between input and the graph matrix.

               :param args: a 2D Tensor or a list of 2D, batch x n, Tensors.
               :param output_size:
               :param bias:
               :param bias_start:
               :param scope:
               :return:
               """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.get_shape()[0].value
        inputs = tf.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = tf.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = tf.concat([inputs, state], axis=2)
        input_size = inputs_and_state.get_shape()[2].value
        dtype = inputs.dtype

        x = inputs_and_state
        x = tf.transpose(x, [1, 0 ,2]) # [batch_size, num_nodes, total_arg_size]
        # x0 = tf.transpose(x, perm=[1, 2, 0])  # (num_nodes, total_arg_size, batch_size)
        # x = tf.reshape(x0, shape=[self._num_nodes, input_size * batch_size])

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            attns = []
            for i in range(self.n_heads[0]):
                attns.append(self._attn_head(x, ids=self._supports[0],
                                              split_parts=self.split_parts[0],
                                              out_sz=self.hid_units[0], activation=tf.nn.leaky_relu,
                                              in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=False,
                                              name="attn_{}_{}".format("in", i)))
            h_1 = tf.concat(attns, axis=-1)
            for i in range(1, len(self.hid_units)):
                h_old = h_1
                attns = []
                for j in range(self.n_heads[i]):
                    attns.append(self._attn_head(h_1, ids=self._supports[0],
                                                  split_parts=self.split_parts[i],
                                                  out_sz=self.hid_units[i], activation=tf.nn.leaky_relu,
                                                  in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=False,
                                                  name="attn_{}_{}".format(i, j)))
                h_1 = tf.concat(attns, axis=-1)

            out = []
            for i in range(self.n_heads[-1]):
                out.append(self._attn_head(h_1, ids=self._supports[0],
                                             split_parts=self.split_parts[-1],
                                             out_sz=output_size, activation=lambda x: x,
                                             in_drop=self.ffd_drop, coef_drop=self.attn_drop, residual=False,
                                             name="attn_{}_{}".format("out", i)))

            logits = tf.add_n(out) / self.n_heads[-1] # (1, num_nodes, state_dim*batch_size)

        logits = tf.reshape(logits, [self._num_nodes, output_size, batch_size])
        logits = tf.transpose(logits, [2, 0, 1])

        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return tf.reshape(logits, [batch_size, self._num_nodes * output_size])

    def _attn_head(self, seq, ids, out_sz, activation, sparse=False, split_parts=2, attn="simple", sp_wei=None, in_drop=0.0, coef_drop=0.0, residual=False, name="attn"):
        if attn == "inner":
            attn = self._attn_inner
        elif attn == "simple":
            attn = self._attn_simple
        else:
            raise Exception

        with tf.name_scope('my_attn'):

            if in_drop != 0.0:
                seq = tf.layers.dropout(seq, 1.0 - in_drop)

            seq_fts_sp = []
            # seq [ns, bs, fd]
            num_nodes = seq.shape[0]
            batch_size = seq.shape[1]
            for _ in range(split_parts):
                sf = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)  # [ns, bs, d]
                sf = tf.nn.embedding_lookup(tf.reshape(sf, [num_nodes, -1]), ids) # [ns, n, bs * d]
                seq_fts_sp.append(sf)
            seq_fts_sp = tf.stack(seq_fts_sp, 2)  # [ns, n, sp, bs * d]
            seq_fts_sp = tf.reshape(seq_fts_sp, [num_nodes, -1, split_parts, batch_size, out_sz])
            seq_fts_sp = tf.transpose(seq_fts_sp, [3, 0, 1, 2, 4]) # [bs, ns, n, sp, d]
            seq_fts_sp = tf.reshape(seq_fts_sp, [batch_size * num_nodes, -1, split_parts, out_sz]) # [bs * ns, n, sp, d]
            sfs_bk = seq_fts_sp
            seq_fts = tf.reduce_mean(seq_fts_sp, 2)  # [bs * ns, n, bs * d]

            cnt_fts = seq_fts[:, 0:1, :]  # [bs * ns, 1, d]

            # simplest self-attention possible
            logits = attn(tf.expand_dims(cnt_fts, 1), seq_fts_sp, 4, name)  # [bs * ns, n, 1, sp]
            in_coefs = tf.nn.softmax(tf.nn.leaky_relu(logits), axis=-1)  # [bs * ns, n, 1, sp]

            logits = in_coefs * logits  # [bs * ns, n, 1, sp]
            coefs = tf.nn.softmax(tf.nn.leaky_relu(tf.reduce_sum(logits, axis=-1)))  # [bs * ns, n, 1]
            coefs = tf.expand_dims(coefs, -1)  # [bs * ns, n, 1, 1]
            coefs = coefs * in_coefs  # [bs * ns, n, 1, sp]

            if coef_drop != 0.0:
                coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
            if in_drop != 0.0:
                seq_fts_sp = tf.nn.dropout(seq_fts_sp, 1.0 - in_drop)

            vals = tf.matmul(coefs, seq_fts_sp)  # [bs * ns, n, 1, d]
            ret = tf.reduce_sum(vals, axis=1)  # [bs * ns, 1, d]
            ret = tf.reshape(ret, [batch_size, num_nodes, out_sz]) # [bs, ns, d]
            ret = tf.transpose(ret, [1, 0, 2]) # [ns, bs, d]

            return activation(ret)  # activation

    def _attn_inner(self, f1, f2, attn_size, name):

        f1 = tf.layers.dense(f1, attn_size, use_bias=False, name=name + "_attn_inner", reuse=tf.AUTO_REUSE)
        f2 = tf.layers.dense(f2, attn_size, use_bias=False, name=name + "_attn_inner", reuse=tf.AUTO_REUSE)
        logits = f1 * f2
        logits = tf.reduce_sum(logits, -1)
        logits = tf.expand_dims(logits, -2)

        return logits

    def _attn_simple(self, f1, f2, attn_size, name):

        f1 = tf.layers.dense(f1, 1, use_bias=False, name=name + "_attn_inner", reuse=tf.AUTO_REUSE)
        f2 = tf.layers.dense(f2, 1, use_bias=False, name=name + "_attn_inner", reuse=tf.AUTO_REUSE)
        logits = f1 + f2
        logits = tf.transpose(logits, [0, 1, 3, 2])

        return logits