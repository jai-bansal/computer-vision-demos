# This script evalutes an object detection model.

# Overall, to get this project working I used the following blog post:
# https://towardsdatascience.com/building-a-toy-detector-with-tensorflow-object-detection-api-63c0fdf2ac95 
# and LOTS of Googling errors.

# This script in particular can be a bitch.

# You will need the "cocoapi" repository. The link is:
# https://github.com/cocodataset/cocoapi
# IF YOU ARE ON WINDOWS, BE SURE TO USE:
# https://github.com/philferriere/cocoapi

# You will need to install Visual C++. Instructions are in the Github link 
# above (the Windows version link).

# Finally, you will need to run the command:
# python PythonAPI/setup.py build_ext install
# This installs the "pycocotools" package and enables the 
# "from object_detection.legacy import evaluator"
# command below.

###############################################################################

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Evaluation executable for detection models.

This executable is used to evaluate DetectionModels. There are two ways of
configuring the eval job.

1) A single pipeline_pb2.TrainEvalPipelineConfig file maybe specified instead.
In this mode, the --eval_training_data flag may be given to force the pipeline
to evaluate on training data instead.

Example usage:
    ./eval \
        --logtostderr \
        --checkpoint_dir=path/to/checkpoint_dir \
        --eval_dir=path/to/eval_dir \
        --pipeline_config_path=pipeline_config.pbtxt

2) Three configuration files may be provided: a model_pb2.DetectionModel
configuration file to define what type of DetectionModel is being evaulated, an
input_reader_pb2.InputReader file to specify what data the model is evaluating
and an eval_pb2.EvalConfig file to configure evaluation parameters.

Example usage:
    ./eval \
        --logtostderr \
        --checkpoint_dir=path/to/checkpoint_dir \
        --eval_dir=path/to/eval_dir \
        --eval_config_path=eval_config.pbtxt \
        --model_config_path=model_config.pbtxt \
        --input_config_path=eval_input_config.pbtxt
"""

import functools
import tensorflow as tf

from google.protobuf import text_format 
#from object_detection import evaluator # error
from object_detection.legacy import evaluator
from object_detection.builders import input_reader_builder
from object_detection.builders import model_builder
from object_detection.protos import eval_pb2
from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import pipeline_pb2
from object_detection.utils import label_map_util


tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_boolean('eval_training_data', False,
                     'If training data should be evaluated for this job.')
flags.DEFINE_string('checkpoint_dir', 'training_artifacts',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')
flags.DEFINE_string('eval_dir', 'eval_dir',
                    'Directory to write eval summaries to.')
flags.DEFINE_string('pipeline_config_path', 'ssd_mobilenet_v1_coco_2018_01_28.config',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')

flags.DEFINE_string('eval_config_path', '',
                    'Path to an eval_pb2.EvalConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')

FLAGS = flags.FLAGS


def get_configs_from_pipeline_file():
  """Reads evaluation configuration from a pipeline_pb2.TrainEvalPipelineConfig.

  Reads evaluation config from file specified by pipeline_config_path flag.

  Returns:
    model_config: a model_pb2.DetectionModel
    eval_config: a eval_pb2.EvalConfig
    input_config: a input_reader_pb2.InputReader
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

  model_config = pipeline_config.model
  if FLAGS.eval_training_data:
    eval_config = pipeline_config.train_config
  else:
    eval_config = pipeline_config.eval_config
  input_config = pipeline_config.eval_input_reader

  return model_config, eval_config, input_config


def get_configs_from_multiple_files():
  """Reads evaluation configuration from multiple config files.

  Reads the evaluation config from the following files:
    model_config: Read from --model_config_path
    eval_config: Read from --eval_config_path
    input_config: Read from --input_config_path

  Returns:
    model_config: a model_pb2.DetectionModel
    eval_config: a eval_pb2.EvalConfig
    input_config: a input_reader_pb2.InputReader
  """
  eval_config = eval_pb2.EvalConfig()
  with tf.gfile.GFile(FLAGS.eval_config_path, 'r') as f:
    text_format.Merge(f.read(), eval_config)

  model_config = model_pb2.DetectionModel()
  with tf.gfile.GFile(FLAGS.model_config_path, 'r') as f:
    text_format.Merge(f.read(), model_config)

  input_config = input_reader_pb2.InputReader()
  with tf.gfile.GFile(FLAGS.input_config_path, 'r') as f:
    text_format.Merge(f.read(), input_config)

  return model_config, eval_config, input_config


def main(unused_argv):
  assert FLAGS.checkpoint_dir, '`checkpoint_dir` is missing.'
  assert FLAGS.eval_dir, '`eval_dir` is missing.'
  if FLAGS.pipeline_config_path:
    model_config, eval_config, input_config = get_configs_from_pipeline_file()
  else:
    model_config, eval_config, input_config = get_configs_from_multiple_files()

  model_fn = functools.partial(
      model_builder.build,
      model_config=model_config,
      is_training=False)

  create_input_dict_fn = functools.partial(
      input_reader_builder.build,
      input_config)

  label_map = label_map_util.load_labelmap(input_config.label_map_path)
  max_num_classes = max([item.id for item in label_map.item])
  categories = label_map_util.convert_label_map_to_categories(
      label_map, max_num_classes)

  evaluator.evaluate(create_input_dict_fn, model_fn, eval_config, categories,
                     FLAGS.checkpoint_dir, FLAGS.eval_dir)

if __name__ == '__main__':
  tf.app.run()
