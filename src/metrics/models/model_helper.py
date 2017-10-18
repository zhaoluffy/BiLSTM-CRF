import tensorflow as tf
import collections
import time


class Model(
    collections.namedtuple("TrainModel", ("graph", "model"))):
    pass


def create_model(model_creator, params, mode):
    """Create train graph, model, and iterator."""

    graph = tf.Graph()

    with graph.as_default():
        model = model_creator(params, mode=mode)

    return Model(
        graph=graph,
        model=model)


def create_or_load_model(session, model, model_dir, name, loaded_vector=None):
    """Create translation model and initialize or load parameters in session."""
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        if loaded_vector is not None:
            session.run(model.word_embedding.assign(loaded_vector))
            print("Load pre-trained embedding.")
        print("  created %s model with fresh parameters, time %.2fs" %
              (name, time.time() - start_time))

    global_step = model.global_step.eval(session=session)
    return model, global_step


def load_model(model, ckpt, session, name):
    start_time = time.time()
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    print(
        "  loaded %s model parameters from %s, time %.2fs" %
        (name, ckpt, time.time() - start_time))
    return model


def create_learning_rate_decay_fn(decay_type,
                                  decay_steps,
                                  decay_rate,
                                  start_decay_at=0,
                                  stop_decay_at=1e9,
                                  min_learning_rate=None,
                                  staircase=False):
    """Creates a function that decays the learning rate.

    Args:
      decay_steps: How often to apply decay.
      decay_rate: A Python number. The decay rate.
      start_decay_at: Don't decay before this step
      stop_decay_at: Don't decay after this step
      min_learning_rate: Don't decay below this number
      decay_type: A decay function name defined in `tf.train`
      staircase: Whether to apply decay in a discrete staircase,
        as opposed to continuous, fashion.

    Returns:
      A function that takes (learning_rate, global_step) as inputs
      and returns the learning rate for the given step.
      Returns `None` if decay_type is empty or None.
    """
    if decay_type is None or decay_type == "":
        return None

    start_decay_at = tf.to_int32(start_decay_at)
    stop_decay_at = tf.to_int32(stop_decay_at)

    def decay_fn(learning_rate, global_step):
        """The computed learning rate decay function.
        """
        global_step = tf.to_int32(global_step)

        decay_type_fn = getattr(tf.train, decay_type)
        decayed_learning_rate = decay_type_fn(
            learning_rate=learning_rate,
            global_step=tf.minimum(global_step, stop_decay_at) - start_decay_at,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name="decayed_learning_rate")

        final_lr = tf.train.piecewise_constant(
            x=global_step,
            boundaries=[start_decay_at],
            values=[learning_rate, decayed_learning_rate])

        if min_learning_rate:
            final_lr = tf.maximum(final_lr, min_learning_rate)

        return final_lr

    return decay_fn


def bidirectional_LSTM(input, num_unit, sequence_length, output_sequence=True):

    with tf.variable_scope("bidirectional_LSTM"):
        cell_fw = tf.contrib.rnn.LSTMCell(num_unit, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.LSTMCell(num_unit, state_is_tuple=True)
        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, input, sequence_length=sequence_length, dtype=tf.float32)

        if output_sequence == True:
            outputs_forward, outputs_backward = outputs
            output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
        else:
            final_states_forward, final_states_backward = final_states
            output = tf.concat([final_states_forward[1], final_states_backward[1]], axis=1, name='output')

    return output
