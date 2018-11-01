# partially copied from tensorflow example project
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops  # pylint: disable=E0611
from tensorflow.python.client.session import Session  # pylint: disable=E0611
from tensorflow.python.training.saver import get_checkpoint_state  # pylint: disable=E0611


def get_logger():
    return logging.getLogger(__name__)


class CustomSessionManager(object):
    def __init__(self, session_init_fn, graph=None):
        self._session_init_fn = session_init_fn
        if graph is None:
            graph = ops.get_default_graph()
        self._graph = graph

    def prepare_session(self, master, checkpoint_dir=None, saver=None, config=None, **_):
        logger = get_logger()
        logger.info('prepare_session')
        session = Session(master, graph=self._graph, config=config)
        self._session_init_fn(session)
        if saver and checkpoint_dir:
            ckpt = get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:  # pylint: disable=no-member
                logger.info('restoring from %s',
                            ckpt.model_checkpoint_path)  # pylint: disable=no-member
                saver.restore(session, ckpt.model_checkpoint_path)  # pylint: disable=no-member
                saver.recover_last_checkpoints(
                    ckpt.all_model_checkpoint_paths)  # pylint: disable=no-member
            else:
                logger.info('no valid checkpoint in %s', checkpoint_dir)
        return session


class CustomSupervisor(tf.train.Supervisor):
    def __init__(self, model, graph, init_op=None, ready_op=None, save_model_secs=0, **kwargs):
        with graph.as_default():
            init_op = tf.global_variables_initializer()

        def custom_init(session):
            logging.info('initializing, session: %s', session)
            session.run(init_op)
            model.initialize(session)
            return True

        session_manager = CustomSessionManager(
            session_init_fn=custom_init,
            graph=graph
        )
        super(CustomSupervisor, self).__init__(
            session_manager=session_manager,
            graph=graph,
            init_op=init_op,
            ready_op=ready_op,
            save_model_secs=save_model_secs,
            **kwargs
        )


class SimpleStepScheduler(object):
    """
    Rather than using threads, with this scheduler the client has full control.
    For example it can be triggered any time intentionally or at the end.
    """

    def __init__(self, do_fn, min_interval, min_freq=0, step=0, last_run=None):
        self.do_fn = do_fn
        self.min_interval = min_interval
        self.min_freq = min_freq
        self.current_step = step
        self.last_run = last_run
        self.dirty = False

    def run_now(self, now):
        self.do_fn()
        self.last_run = now
        self.dirty = False

    def should_trigger(self, now):
        result = (
            (
                (self.min_freq > 0) and
                (self.current_step % self.min_freq == 0)
            ) or
            (
                (self.min_interval > 0) and
                (self.last_run is None or (now - self.last_run) >= self.min_interval)
            )
        )
        if result:
            get_logger().info(
                'should_trigger: current_step:%s, min_freq=%s, now=%s, '
                'last_run=%s, min_interval=%s, result=%s',
                self.current_step, self.min_freq, now,
                self.last_run, self.min_interval, result
            )
        return result

    def step(self, now):
        self.current_step += 1
        if self.should_trigger(now=now):
            self.run_now(now=now)
        else:
            self.dirty = True

    def flush(self, now):
        if self.dirty:
            self.run_now(now)


def loss(loss_value):
    """Calculates aggregated mean loss."""
    total_loss = tf.Variable(0.0, False)
    loss_count = tf.Variable(0, False)
    total_loss_update = tf.assign_add(total_loss, loss_value)
    loss_count_update = tf.assign_add(loss_count, 1)
    loss_op = total_loss / tf.cast(loss_count, tf.float32)
    return [total_loss_update, loss_count_update], loss_op


def get_graph_size():
    return sum([
        int(np.product(v.get_shape().as_list()) * v.dtype.size)
        for v in tf.global_variables()
    ])
