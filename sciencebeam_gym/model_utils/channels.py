import tensorflow as tf

from sciencebeam_gym.utils.tf import (
    variable_scoped
)


def color_equals_mask(image, color):
    return tf.reduce_all(
        tf.equal(image, color),
        axis=-1,
        name='is_color'
    )


def color_equals_mask_as_float(image, color):
    return tf.cast(color_equals_mask(image, color), tf.float32)


def calculate_color_masks(image, colors, use_unknown_class=False):
    color_masks = [
        variable_scoped(
            'channel_%d' % i,
            lambda color_param: color_equals_mask_as_float(image, color_param),
            color_param=color
        )
        for i, color in enumerate(colors)
    ]
    if use_unknown_class:
        with tf.variable_scope("unknown_class"):
            shape = tf.shape(color_masks[0])
            ones = tf.fill(shape, 1.0, name='ones')
            zeros = tf.fill(shape, 0.0, name='zeros')
            color_masks.append(
                tf.where(
                    tf.add_n(color_masks) < 0.5,
                    ones,
                    zeros
                )
            )
    return color_masks
