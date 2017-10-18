"""Base class for models"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from src import losses as seq2seq_losses
from src.models.model_base import ModelBase
from tensorflow.contrib.crf import viterbi_decode
from src.models import model_helper


class BiRNNModel(ModelBase):
    """Base class for seq2seq models with embeddings
    """

    def __init__(self, params, mode, name="birnn_model"):
        super(BiRNNModel, self).__init__(params, mode, name)

        self.encoder_params = self.params["encoder.params"]
        # placeholders
        self.source_placeholder = tf.placeholder(tf.int32,
                                                 shape=[None, None],
                                                 name="source_placeholder")
        self.seq_len_palceholder = tf.placeholder(tf.int32,
                                                  shape=[None],
                                                  name="seq_len_palceholder")
        self.src_char_palceholder = tf.placeholder(tf.int32,
                                                   shape=[None, None, None],
                                                   name="src_char_palceholder")
        self.word_len_placeholder = tf.placeholder(tf.int32,
                                                   shape=[None, None],
                                                   name="word_len_placeholder")
        self.target_palceholder = tf.placeholder(tf.int32,
                                                 shape=[None, None],
                                                 name="target_palceholder")

        self.loss, self.logits, self.train_op, self.transition_param = \
            self.build_graph()

        # Saver
        self.saver = tf.train.Saver(tf.global_variables(),
                                    max_to_keep=self.params["keep_checkpoint_max"])

    @staticmethod
    def default_params():
        params = ModelBase.default_params()
        params.update({
            "use_char": False,
            "embedding.word_dim": 300,
            "embedding.char_dim": 100,
            "embedding.init_scale": 0.04,
            "optimizer.clip_embed_gradients": 0.1,
            "count_vocab_source": 0,
            "count_vocab_char": 0,
            "count_vocab_target": 0,
            "dropout_keep_prob": 0.5,
            "encoder.class": "seq2seq.encoders.BidirectionalRNNEncoder",
            "encoder.params": {},  # Arbitrary parameters for the encoder
            "keep_checkpoint_max": 5,
        })
        return params

    def _clip_gradients(self, grads_and_vars):
        """In addition to standard gradient clipping, also clips embedding
        gradients to a specified value."""
        grads_and_vars = super(BiRNNModel, self)._clip_gradients(grads_and_vars)

        clipped_gradients = []
        variables = []
        for gradient, variable in grads_and_vars:
            if "embedding" in variable.name:
                tmp = tf.clip_by_norm(
                    gradient.values, self.params["optimizer.clip_embed_gradients"])
                gradient = tf.IndexedSlices(tmp, gradient.indices, gradient.dense_shape)
            clipped_gradients.append(gradient)
            variables.append(variable)
        return list(zip(clipped_gradients, variables))

    @property
    def word_embedding(self):
        return self._word_embeddings

    def embedding(self):
        with tf.variable_scope("words"):
            self._word_embeddings = tf.get_variable(name="_word_embeddings",
                                                    shape=[self.params["count_vocab_source"],
                                                           self.params["embedding.word_dim"]],
                                                    trainable=True
                                                    )
            word_embeddings = tf.nn.embedding_lookup(self._word_embeddings, self.source_placeholder,
                                                     name="word_embeddings")
        with tf.variable_scope("chars"):
            if self.params["use_char"]:
                # get embeddings matrix
                _char_embeddings = tf.get_variable(name="_char_embeddings",
                                                   shape=[self.params["count_vocab_char"],
                                                          self.params["embedding.char_dim"]],
                                                   dtype=tf.float32
                                                   )
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings, self.src_char_palceholder,
                                                         name="char_embeddings")
                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings, shape=[-1, s[-2], self.params["embedding.char_dim"]])
                word_lengths = tf.reshape(self.word_len_placeholder, shape=[-1])

                output = model_helper.bidirectional_LSTM(
                    char_embeddings, self.params["embedding.char_dim"], word_lengths, output_sequence=False)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output, shape=[-1, s[1], 2 * self.params["embedding.char_dim"]])

                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            word_embeddings = tf.nn.dropout(word_embeddings, self.params["dropout_keep_prob"])

        return word_embeddings

    def encode(self, word_embeddings):
        with tf.variable_scope("words_encoder"):
            num_unit = self.encoder_params["rnn_cell"]["num_units"]

            output = model_helper.bidirectional_LSTM(
                word_embeddings, num_unit, self.seq_len_palceholder, output_sequence=True)

            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                output = tf.nn.dropout(output, self.params["dropout_keep_prob"])

        with tf.variable_scope("dense"):
            W = tf.get_variable("W", shape=[2 * num_unit, self.params["count_vocab_target"]],
                                dtype=tf.float32)

            b = tf.get_variable("b", shape=[self.params["count_vocab_target"]],
                                dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            ntime_steps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * num_unit])
            logits = tf.nn.xw_plus_b(output, W, b)
            logits = tf.reshape(logits, [-1, ntime_steps, self.params["count_vocab_target"]], name="logits")

        return logits

    def compute_loss(self, logits):
        """Computes the loss for this model.

        Returns a tuple `(losses, loss)`, where `losses` are the per-batch
        losses and loss is a single scalar tensor to minimize.
        """
        loss, transition_param = seq2seq_losses.crf_loss(
            logits=logits,
            targets=self.target_palceholder,
            sequence_length=self.seq_len_palceholder)

        return loss, transition_param

    def build_graph(self):
        # Pre-process features and labels
        with tf.variable_scope(self.name):
            word_embedded = self.embedding()
            logits = self.encode(word_embedded)

            loss, transition_param = self.compute_loss(logits)

            train_op = None
            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                train_op = self._build_train_op(loss)

        return loss, logits, train_op, transition_param

    def create_feed_dict(self, batch):
        """
        :param batch: list train/evaluate data
        :return: structured data to feed
        """
        sources, seq_lens, src_chars, word_lens, targets = batch
        feed_dict = {
            self.source_placeholder: sources,
            self.seq_len_palceholder: seq_lens,
            self.target_palceholder: targets
        }
        if self.params["use_char"]:
            feed_dict[self.src_char_palceholder] = src_chars
            feed_dict[self.word_len_placeholder] = word_lens

        return feed_dict

    def train(self, sess, batch):
        """
        :param sess: session to run the batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        feed_dict = self.create_feed_dict(batch)
        return sess.run([self.train_op,
                         self.loss,
                         self.global_step
                         ],
                        feed_dict=feed_dict)

    def eval(self, sess, batch):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        feed_dict = self.create_feed_dict(batch)
        return sess.run([self.loss], feed_dict=feed_dict)

    def infer(self, sess, batch):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        feed_dict = self.create_feed_dict(batch)
        return sess.run([self.logits,
                         self.seq_len_palceholder,
                         self.transition_param],
                        feed_dict=feed_dict)

    def decode(self, sess, batch):
        """Creates the dictionary of predictions that is returned by the model.
        """
        logits, sequence_lengths, transition_param = self.infer(sess, batch=batch)
        predicted_ids = []
        for logit, length in zip(logits, sequence_lengths):
            logit = logit[:length]
            predicted_id, _ = viterbi_decode(logit, transition_param)
            predicted_ids.append(predicted_id)

        return predicted_ids
