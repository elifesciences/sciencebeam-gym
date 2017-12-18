# Mostly copied from https://github.com/affinelayer/pix2pix-tensorflow (MIT)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import collections

import tensorflow as tf

from sciencebeam_gym.trainer.models.pix2pix.tf_utils import (
  blank_other_channels,
  get_channel_slice
)

from sciencebeam_gym.trainer.models.pix2pix.loss import (
  l1_loss,
  cross_entropy_loss,
  weighted_cross_entropy_loss
)

EPS = 1e-12

class BaseLoss(object):
  L1 = "L1"
  CROSS_ENTROPY = "CE"
  # Loss with pre-calculated weights (and cross entropy)
  # (weights calculated across entire dataset)
  WEIGHTED_CROSS_ENTROPY = "WCE"
  # Loss with per sample weights (and cross entropy)
  # (weights calculated as part of the graph)
  SAMPLE_WEIGHTED_CROSS_ENTROPY = "SWCE"
  # Combination of the above; i.e. sample weights multiplied by pre-calculated weights
  WEIGHTED_SAMPLE_WEIGHTED_CROSS_ENTROPY = "WSWCE"

ALL_BASE_LOSS = [
  BaseLoss.L1, BaseLoss.CROSS_ENTROPY, BaseLoss.WEIGHTED_CROSS_ENTROPY,
  BaseLoss.SAMPLE_WEIGHTED_CROSS_ENTROPY, BaseLoss.WEIGHTED_SAMPLE_WEIGHTED_CROSS_ENTROPY
]

ALL_CE_BASE_LOSS = {
  BaseLoss.CROSS_ENTROPY,
  BaseLoss.WEIGHTED_CROSS_ENTROPY,
  BaseLoss.SAMPLE_WEIGHTED_CROSS_ENTROPY,
  BaseLoss.WEIGHTED_SAMPLE_WEIGHTED_CROSS_ENTROPY
}

ALL_WCE_BASE_LOSS = ALL_CE_BASE_LOSS - {BaseLoss.CROSS_ENTROPY}

Pix2PixModel = collections.namedtuple(
  "Pix2PixModel", [
    "inputs",
    "targets",
    "outputs",
    "predict_real",
    "predict_fake",
    "discrim_loss",
    "discrim_grads_and_vars",
    "gen_loss_GAN",
    "gen_loss_L1",
    "gen_grads_and_vars",
    "global_step",
    "train"
  ]
)

def get_logger():
  return logging.getLogger(__name__)

def lrelu(x, a):
  with tf.name_scope("lrelu"):
    # adding these together creates the leak part and linear part
    # then cancels them out by subtracting/adding an absolute value term
    # leak: a*x/2 - a*abs(x)/2
    # linear: x/2 + abs(x)/2

    # this block looks like it has 2 inputs on the graph unless we do this
    x = tf.identity(x)
    return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(batch_input):
  with tf.variable_scope("batchnorm"):
    # this block looks like it has 3 inputs on the graph unless we do this
    batch_input = tf.identity(batch_input)

    input_shape = batch_input.get_shape()
    get_logger().debug('batchnorm, input_shape: %s', input_shape)
    channels = input_shape[-1]
    offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
    scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
    mean, variance = tf.nn.moments(batch_input, axes=[0, 1, 2], keep_dims=False)
    variance_epsilon = 1e-5
    normalized = tf.nn.batch_normalization(batch_input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
    return normalized

def conv(batch_input, out_channels, stride):
  with tf.variable_scope("conv"):
    input_shape = batch_input.get_shape()
    get_logger().debug('conv, input_shape: %s', input_shape)
    in_channels = input_shape[-1]
    conv_filter = tf.get_variable(
      "filter",
      [4, 4, in_channels, out_channels],
      dtype=tf.float32,
      initializer=tf.random_normal_initializer(0, 0.02)
    )
    # [batch, in_height, in_width, in_channels],
    # [filter_width, filter_height, in_channels, out_channels]
    #     => [batch, out_height, out_width, out_channels]
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.nn.conv2d(padded_input, conv_filter, [1, stride, stride, 1], padding="VALID")

def deconv(batch_input, out_channels):
  with tf.variable_scope("deconv"):
    input_shape = batch_input.get_shape()
    get_logger().debug('deconv, input_shape: %s', input_shape)

    batch, in_height, in_width, in_channels = input_shape.as_list()
    conv_filter = tf.get_variable(
      "filter",
      [4, 4, out_channels, in_channels],
      dtype=tf.float32,
      initializer=tf.random_normal_initializer(0, 0.02)
    )
    # [batch, in_height, in_width, in_channels],
    # [filter_width, filter_height, out_channels, in_channels]
    #     => [batch, out_height, out_width, out_channels]
    # tf.shape creates a tensor to allow calculation of shape at runtime
    dynamic_batch_size = tf.shape(batch_input)[0]
    output_shape = [batch, in_height * 2, in_width * 2, out_channels]
    return tf.reshape(
      tf.nn.conv2d_transpose(
        batch_input,
        conv_filter,
        tf.stack([dynamic_batch_size] + output_shape[1:]),
        [1, 2, 2, 1],
        padding="SAME"
      ),
      [batch or -1] + output_shape[1:]
    )

def create_encoder_layers(
  generator_inputs,
  layer_specs):
  """
  Creates encoders with every layer halfing width and height,
  using an output depth specified by layer_specs.

  Args:
    generator_inputs: A `Tensor`. The input to the generator.
    layer_specs: A list of number of output channels for each layer.
  Returns:
    A list of `Tensor`. The output tensor of each layer.
  """

  layers = []
  for out_channels in layer_specs:
    with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
      if not layers:
        # very first layer is connected to inputs and doesn't use batch norm
        output = conv(generator_inputs, out_channels, stride=2)
        layers.append(output)
      else:
        rectified = lrelu(layers[-1], 0.2)
        # [batch, in_height, in_width, in_channels] =>
        # [batch, in_height/2, in_width/2, out_channels]
        convolved = conv(rectified, out_channels, stride=2)
        output = batchnorm(convolved)
        layers.append(output)
  return layers

def conditional_dropout(cond, x, **kwargs):
  return tf.cond(cond, lambda: tf.nn.dropout(x, **kwargs), lambda: x)

def create_decoder_layers(
  decoder_inputs,
  layer_specs,
  skip_connection_layers,
  is_training):
  """
  Creates decoders with every layer doubling width and height,
  using an output depth and dropout specified by layer_specs.

  Args:
    generator_inputs: A `Tensor`. The input to the generator.
    layer_specs: A list of tuples for each layer. Each tuple containing
      number of output channels and dropout.
  Returns:
    A list of `Tensor`. The output tensor of each layer.
  """

  layers = []
  num_encoder_layers = len(skip_connection_layers)
  for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
    skip_layer = num_encoder_layers - decoder_layer - 1
    is_last_layer = decoder_layer == len(layer_specs) - 1
    with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
      if decoder_layer == 0:
        # first decoder layer doesn't have skip connections
        # since it is directly connected to the skip_layer
        layer_input = decoder_inputs
      else:
        layer_input = tf.concat([layers[-1], skip_connection_layers[skip_layer]], axis=3)

      rectified = tf.nn.relu(layer_input)
      # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
      output = deconv(rectified, out_channels)
      if not is_last_layer:
        # very last layer does not use batch norm
        output = batchnorm(output)

      if dropout > 0.0:
        # only apply dropout in training mode
        output = conditional_dropout(
          tf.convert_to_tensor(is_training),
          output,
          keep_prob=1 - dropout
        )

      layers.append(output)
  return layers

def create_encoder_decoder(
  generator_inputs,
  encoder_layer_specs,
  decoder_layer_specs,
  is_training):

  encoder_layers = create_encoder_layers(
    generator_inputs,
    encoder_layer_specs
  )
  decoder_layers = create_decoder_layers(
    encoder_layers[-1],
    decoder_layer_specs,
    encoder_layers,
    is_training
  )
  return decoder_layers[-1]

def create_generator(generator_inputs, generator_outputs_channels, a, is_training):
  encoder_layer_specs = [
    a.ngf,     # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
    a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
    a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
    a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
    a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
    a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
    a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
  ]
  decoder_layer_specs = [
    (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
    (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
    (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
    (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
    (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
    (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
    (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    # decoder_1 [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    (generator_outputs_channels, 0.0)
  ]
  return create_encoder_decoder(
    generator_inputs,
    encoder_layer_specs,
    decoder_layer_specs,
    is_training
  )

def create_discriminator(discrim_inputs, discrim_targets, a, out_channels=1):
  n_layers = 3
  layers = []

  # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
  layer_input = tf.concat([discrim_inputs, discrim_targets], axis=3)

  # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
  with tf.variable_scope("layer_1"):
    convolved = conv(layer_input, a.ndf, stride=2)
    rectified = lrelu(convolved, 0.2)
    layers.append(rectified)

  # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
  # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
  # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
  for i in range(n_layers):
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
      layer_out_channels = a.ndf * min(2**(i+1), 8)
      stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
      convolved = conv(layers[-1], layer_out_channels, stride=stride)
      normalized = batchnorm(convolved)
      rectified = lrelu(normalized, 0.2)
      layers.append(rectified)

  # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
  with tf.variable_scope("layer_%d" % (len(layers) + 1)):
    convolved = conv(rectified, out_channels=out_channels, stride=1)
    output = tf.sigmoid(convolved)
    layers.append(output)

  return layers[-1]

def with_variable_scope(name, fn, reuse=False, **kwargs):
  with tf.variable_scope(name, reuse=reuse):
    return fn(**kwargs)

def create_separate_discriminators(discrim_inputs, discrim_targets, a, out_channels=1):
  if out_channels == 1:
    return create_discriminator(discrim_inputs, discrim_targets, a, out_channels=1)
  n_targets_channels = discrim_targets.shape[-1]
  reuse = tf.get_variable_scope().reuse
  return tf.concat([
    with_variable_scope(
      'layer_{}'.format(channel_index),
      lambda i: create_discriminator(
        discrim_inputs,
        get_channel_slice(discrim_targets, i),
        a,
        out_channels=1
      ),
      reuse=reuse,
      i=channel_index
    )
    for channel_index in range(n_targets_channels)
  ], axis=0)

def create_separate_channel_discriminator_by_blanking_out_channels(inputs, targets, a):
  # We need to teach the discriminator to detect the real channels,
  # by just looking at the real channel.
  # For each channel:
  # - let the discriminator only see the current channel, blank out all other channels
  # - expect output to not be fake for the not blanked out channel
  n_targets_channels = int(targets.shape[-1])
  predict_real_channels = []
  predict_real_blanked_list = []
  for i in range(n_targets_channels):
    masked_targets = blank_other_channels(
      targets,
      i
    )
    with tf.variable_scope("discriminator", reuse=(i > 0)):
      # 2x [batch, height, width, channels] => [batch, 30, 30, n_targets_channels]
      predict_real_i = create_discriminator(
        inputs, masked_targets, a,
        out_channels=n_targets_channels
      )
      predict_real_channels.append(predict_real_i[:, :, :, i])
      for j in range(n_targets_channels):
        if j != i:
          predict_real_blanked_list.append(predict_real_i[:, :, :, j])
  predict_real = tf.stack(
    predict_real_channels,
    axis=-1,
    name='predict_real'
  )
  predict_real_blanked = tf.stack(
    predict_real_blanked_list,
    axis=-1,
    name='predict_real_blanked'
  )
  return predict_real, predict_real_blanked


def create_pix2pix_model(inputs, targets, a, is_training, pos_weight=None, n_output_channels=None):
  get_logger().info('gan_weight: %s, l1_weight: %s', a.gan_weight, a.l1_weight)
  gan_enabled = abs(a.gan_weight) > 0.000001

  is_predict = targets is None
  if n_output_channels is None:
    n_output_channels = int(targets.shape[-1])
  with tf.variable_scope("generator"):
    outputs = create_generator(inputs, n_output_channels, a, is_training)
    if a.base_loss in ALL_CE_BASE_LOSS:
      output_logits = outputs
      outputs = tf.nn.softmax(output_logits)
    else:
      outputs = tf.tanh(outputs)

  if is_predict:
    return Pix2PixModel(
      inputs=inputs,
      targets=None,
      predict_real=None,
      predict_fake=None,
      discrim_loss=None,
      discrim_grads_and_vars=None,
      gen_loss_GAN=None,
      gen_loss_L1=None,
      gen_grads_and_vars=None,
      outputs=outputs,
      global_step=None,
      train=None,
    )

  n_targets_channels = n_output_channels

  if gan_enabled:
    discrim_out_channels = (
      n_targets_channels
      if a.use_separate_discriminator_channels
      else 1
    )
    get_logger().info('discrim_out_channels: %s', discrim_out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
      if discrim_out_channels > 1:
        predict_real, predict_real_blanked = (
          create_separate_channel_discriminator_by_blanking_out_channels(
            inputs, targets, a
          )
        )
      else:
        with tf.variable_scope("discriminator"):
          if a.use_separate_discriminators:
            get_logger().info('using separate discriminators: %s', n_targets_channels)
            predict_real = create_separate_discriminators(
              inputs, targets, a, out_channels=n_targets_channels
            )
          else:
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets, a)

    with tf.name_scope("fake_discriminator"):
      with tf.variable_scope("discriminator", reuse=True):
        # 2x [batch, height, width, channels] => [batch, 30, 30, discrim_out_channels]
        # We don't need to split the channels, the discriminator should detect them all as fake
        if a.use_separate_discriminators:
          predict_fake = create_separate_discriminators(
            inputs, outputs, a, out_channels=n_targets_channels
          )
        else:
          predict_fake = create_discriminator(
            inputs, outputs, a,
            out_channels=discrim_out_channels
          )

    with tf.name_scope("discriminator_loss"):
      # minimizing -tf.log will try to get inputs to 1
      # predict_real => 1
      # predict_fake => 0
      discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
      if discrim_out_channels > 1:
        discrim_loss += tf.reduce_mean(-tf.log(1 - tf.reshape(predict_real_blanked, [-1]) + EPS))

    with tf.name_scope("discriminator_train"):
      discrim_tvars = [
        var for var in tf.trainable_variables() if var.name.startswith("discriminator")
      ]
      discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
      discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
      discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)
  else:
    with tf.name_scope("gan_disabled"):
      discrim_loss = tf.constant(0.0)
      predict_real = None
      predict_fake = None
      discrim_grads_and_vars = []

  with tf.name_scope("generator_loss"):
    get_logger().info('using loss: %s', a.base_loss)
    if a.base_loss == BaseLoss.L1:
      gen_base_loss = l1_loss(labels=targets, outputs=outputs)
    elif a.base_loss == BaseLoss.CROSS_ENTROPY:
      gen_base_loss = cross_entropy_loss(
        logits=output_logits,
        labels=targets
      )
    elif a.base_loss in ALL_WCE_BASE_LOSS:
      if pos_weight is None:
        raise ValueError('pos_weight missing')
      pos_weight = tf.convert_to_tensor(pos_weight)
      gen_base_loss = weighted_cross_entropy_loss(
        logits=output_logits,
        labels=targets,
        pos_weight=pos_weight
      )
    else:
      raise ValueError('unrecognised base loss: %s' % a.base_loss)
    gen_loss = gen_base_loss * a.l1_weight

    if gan_enabled:
      # predict_fake => 1
      gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
      gen_loss += gen_loss_GAN * a.gan_weight
    else:
      gen_loss_GAN = tf.constant(0.0)

  with tf.name_scope("generator_train"):
    generator_train_dependencies = (
      [discrim_train] if gan_enabled
      else []
    )
    with tf.control_dependencies(generator_train_dependencies):
      gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
      gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
      gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
      gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

  ema = tf.train.ExponentialMovingAverage(decay=0.99)
  update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_base_loss])

  global_step = tf.contrib.framework.get_or_create_global_step()
  incr_global_step = tf.assign(global_step, global_step+1)

  # TODO change gen_loss_L1 name
  return Pix2PixModel(
    inputs=inputs,
    targets=targets,
    predict_real=predict_real,
    predict_fake=predict_fake,
    discrim_loss=ema.average(discrim_loss),
    discrim_grads_and_vars=discrim_grads_and_vars,
    gen_loss_GAN=ema.average(gen_loss_GAN),
    gen_loss_L1=ema.average(gen_base_loss),
    gen_grads_and_vars=gen_grads_and_vars,
    outputs=outputs,
    global_step=global_step,
    train=tf.group(update_losses, incr_global_step, gen_train),
  )

def create_image_summaries(model):
  def convert(image):
    return tf.image.convert_image_dtype(
      image,
      dtype=tf.uint8,
      saturate=True
    )
  summaries = {}

  # reverse any processing on images so they can be written to disk or displayed to user
  with tf.name_scope("convert_inputs"):
    converted_inputs = convert(model.inputs)

  with tf.name_scope("convert_targets"):
    converted_targets = convert(model.targets)

  with tf.name_scope("convert_outputs"):
    converted_outputs = convert(model.outputs)

  with tf.name_scope("inputs_summary"):
    tf.summary.image("inputs", converted_inputs)

  with tf.name_scope("targets_summary"):
    tf.summary.image("targets", converted_targets)

  with tf.name_scope("outputs_summary"):
    tf.summary.image("outputs", converted_outputs)
    summaries['output_image'] = converted_outputs

  with tf.name_scope("predict_real_summary"):
    tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

  with tf.name_scope("predict_fake_summary"):
    tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))
  return summaries

def create_other_summaries(model):
  tf.summary.scalar("discriminator_loss", model.discrim_loss)
  tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
  tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name + "/values", var)

  for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
    tf.summary.histogram(var.op.name + "/gradients", grad)
