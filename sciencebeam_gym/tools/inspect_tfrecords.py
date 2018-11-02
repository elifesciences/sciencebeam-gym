from __future__ import print_function
import os
import logging
import argparse

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from sciencebeam_gym.utils.tf import (
    FileIO
)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--records_paths',
        required=True,
        type=str,
        action='append',
        help='The paths to the tf-records files to inspect.'
    )
    parser.add_argument(
        '--inspect_key',
        required=False,
        type=str,
        help='The name of the key to further inspect.'
    )
    parser.add_argument(
        '--extract_dir',
        required=False,
        default=".",
        type=str,
        help='The directory to extract to.'
    )
    parser.add_argument(
        '--extract_image',
        required=False,
        action='append',
        type=str,
        help='The name of the key to extract as an image.'
    )
    return parser


def parse_args(argv=None):
    parser = get_args_parser()
    parsed_args, _ = parser.parse_known_args(argv)
    return parsed_args


def get_matching_files_for_paths(paths):
    files = []
    for path in paths:
        files.extend(file_io.get_matching_files(path))
    # logging.info('files: %s (%s)', files, paths)
    return files


def main():
    args = parse_args()

    files = get_matching_files_for_paths(args.records_paths)

    if args.extract_image:
        file_io.recursive_create_dir(args.extract_dir)

    total_count = 0
    for f in files:
        options = None
        if f.endswith('.gz'):
            options = tf.python_io.TFRecordOptions(
                compression_type=tf.python_io.TFRecordCompressionType.GZIP)
        print("file:", f)
        count = 0
        for i, example in enumerate(tf.python_io.tf_record_iterator(f, options=options)):
            result = tf.train.Example.FromString(example)  # pylint: disable=no-member
            if i == 0:
                print("  features:", result.features.feature.keys())
                if args.inspect_key:
                    print("  first value of {}:\n    {}".format(
                        args.inspect_key,
                        result.features.feature.get(args.inspect_key).bytes_list.value[0]
                    ))
                if args.extract_image:
                    for extract_image_key in args.extract_image:
                        image_bytes = result.features.feature.get(
                            extract_image_key).bytes_list.value[0]
                        print("  image size %d bytes (%s)" % (len(image_bytes), type(image_bytes)))
                        image_filename = os.path.join(
                            args.extract_dir,
                            '{}-{}-{}.png'.format(os.path.basename(f), extract_image_key, i)
                        )
                        print("  extracting image to {}".format(image_filename))
                        with FileIO(image_filename, 'wb') as image_f:
                            image_f.write(image_bytes)
            count += 1
        print("  found {} records".format(count))
        total_count += count
    print("found total of {} records in {} files".format(total_count, len(files)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()
