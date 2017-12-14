import logging
from collections import namedtuple
from mock import patch

import tensorflow as tf
import numpy as np
import pytest

from sciencebeam_gym.utils.collection import (
  extend_dict
)

import sciencebeam_gym.trainer.models.pix2pix.pix2pix_core as pix2pix_core

from sciencebeam_gym.trainer.models.pix2pix.pix2pix_core import (
  create_pix2pix_model,
  BaseLoss
)

DEFAULT_ARGS = dict(
  ngf=64,
  ndf=64,
  lr=0.0002,
  beta1=0.5,
  l1_weight=1.0,
  gan_weight=0.0,
  base_loss=BaseLoss.L1,
  use_separate_discriminator_channels=False,
  use_separate_discriminators=False
)

BATCH_SIZE = 10
WIDTH = 256
HEIGHT = 256
CHANNELS = 3

def get_logger():
  return logging.getLogger(__name__)

def setup_module():
  logging.basicConfig(level='DEBUG')

def create_args(*args, **kwargs):
  d = extend_dict(*list(args) + [kwargs])
  return namedtuple('args', d.keys())(**d)

def patch_spy_object(o, name):
  return patch.object(o, name, wraps=getattr(o, name))

@pytest.mark.slow
@pytest.mark.very_slow
class TestCreatePix2pixModel(object):
  def test_should_be_able_to_construct_graph_with_defaults_without_gan(self):
    with tf.Graph().as_default():
      inputs = tf.constant(np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=np.float32))
      targets = tf.constant(np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=np.float32))
      a = create_args(DEFAULT_ARGS, gan_weight=0.0)
      create_pix2pix_model(inputs, targets, a, is_training=True)

  def test_should_be_able_to_construct_graph_with_defaults_and_gan(self):
    with tf.Graph().as_default():
      inputs = tf.constant(np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=np.float32))
      targets = tf.constant(np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=np.float32))
      a = create_args(DEFAULT_ARGS, gan_weight=1.0)
      create_pix2pix_model(inputs, targets, a, is_training=True)

  def test_should_be_able_to_construct_graph_with_gan_and_sep_discrim_channels(self):
    with patch_spy_object(pix2pix_core, 'l1_loss') as l1_loss:
      with tf.Graph().as_default():
        inputs = tf.constant(np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=np.float32))
        targets = tf.constant(np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=np.float32))
        a = create_args(DEFAULT_ARGS, gan_weight=1.0, use_separate_discriminator_channels=True)
        create_pix2pix_model(inputs, targets, a, is_training=True)
        assert l1_loss.called

  def test_should_be_able_to_construct_graph_with_sep_discrim_channels_and_cross_entropy_loss(self):
    with patch_spy_object(pix2pix_core, 'cross_entropy_loss') as cross_entropy_loss:
      with tf.Graph().as_default():
        inputs = tf.constant(np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=np.float32))
        targets = tf.constant(np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=np.float32))
        a = create_args(
          DEFAULT_ARGS,
          gan_weight=1.0, use_separate_discriminator_channels=True, base_loss=BaseLoss.CROSS_ENTROPY
        )
        create_pix2pix_model(inputs, targets, a, is_training=True)
        assert cross_entropy_loss.called

  def test_should_be_able_to_construct_graph_with_weighted_cross_entropy_loss(self):
    with patch_spy_object(pix2pix_core, 'weighted_cross_entropy_loss') \
      as weighted_cross_entropy_loss:

      with tf.Graph().as_default():
        inputs = tf.constant(np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=np.float32))
        targets = tf.constant(np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNELS), dtype=np.float32))
        a = create_args(
          DEFAULT_ARGS,
          gan_weight=1.0, use_separate_discriminator_channels=True,
          base_loss=BaseLoss.WEIGHTED_CROSS_ENTROPY
        )
        create_pix2pix_model(inputs, targets, a, is_training=True, pos_weight=[1.0] * CHANNELS)
        assert weighted_cross_entropy_loss.called
