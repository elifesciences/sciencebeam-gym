import logging
from io import BytesIO

from PIL import Image
import numpy as np
import tensorflow as tf

from sciencebeam_gym.trainer.util import FileIO
from sciencebeam_gym.trainer.models.pix2pix.pix2pix_model import (
  batch_dimensions_to_most_likely_colors_list
)

def get_logger():
  return logging.getLogger(__name__)

def load_checkpoint_and_predict(model, checkpoint_path, predict_filename, output_image_filename):
  with FileIO(predict_filename, 'rb') as input_f:
    image_bytes = input_f.read()
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img = img.resize((model.image_height, model.image_width))
    img_data = np.asarray(img, dtype=np.uint8)
    img_data = np.reshape(img_data, tuple([1] + list(img_data.shape)))

  with tf.Graph().as_default():
    with tf.Session() as session:
      last_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
      get_logger().info(
        'last_checkpoint: %s (%s)', last_checkpoint, checkpoint_path
      )
      tensors = model.build_predict_graph()
      inputs_tensor = tensors.inputs['image']
      outputs_tensor = tensors.pred
      saver = tf.train.Saver()
      saver.restore(session, last_checkpoint)

      if model.use_separate_channels:
        outputs_tensor = batch_dimensions_to_most_likely_colors_list(
          outputs_tensor,
          model.dimension_colors_with_unknown
        )
      output_img_data = session.run(outputs_tensor, feed_dict={
        inputs_tensor: img_data
      })[0]
      output_image = Image.fromarray(output_img_data, 'RGB')
      out = BytesIO()
      output_image.save(out, 'png')
      output_image_bytes = out.getvalue()
      get_logger().info('writing to %s', output_image_filename)
      with FileIO(output_image_filename, 'wb') as output_f:
        output_f.write(output_image_bytes)
