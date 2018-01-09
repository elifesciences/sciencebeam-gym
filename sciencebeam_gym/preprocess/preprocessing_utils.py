from __future__ import absolute_import

import os
import logging
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from itertools import groupby
from functools import reduce

from six import iteritems

from lxml import etree

from apache_beam.io.filesystems import FileSystems

from sciencebeam_gym.utils.xml import (
  xml_from_string_with_recover
)

from sciencebeam_gym.utils.stopwatch import (
  StopWatchRecorder
)

from sciencebeam_gym.utils.collection import (
  groupby_to_dict,
  sort_and_groupby_to_dict
)

from sciencebeam_gym.beam_utils.io import (
  dirname,
  find_matching_filenames,
  mkdirs_if_not_exists
)

from sciencebeam_gym.preprocess.lxml_to_svg import (
  iter_svg_pages_for_lxml
)

from sciencebeam_gym.structured_document.svg import (
  SvgStructuredDocument
)

from sciencebeam_gym.preprocess.annotator import (
  Annotator,
  DEFAULT_ANNOTATORS
)

from sciencebeam_gym.alignment.align import (
  native_enabled as align_native_enabled
)

from sciencebeam_gym.preprocess.matching_annotator import (
  MatchingAnnotator,
  xml_root_to_target_annotations
)

from sciencebeam_gym.preprocess.visualize_svg_annotation import (
  visualize_svg_annotations
)

from sciencebeam_gym.preprocess.blockify_annotations import (
  annotation_document_page_to_annotation_blocks,
  merge_blocks,
  expand_blocks,
  annotated_blocks_to_image
)

from sciencebeam_gym.pdf import (
  PdfToLxmlWrapper,
  PdfToPng
)


def get_logger():
  return logging.getLogger(__name__)

def group_files_by_parent_directory(filenames):
  return groupby_to_dict(sorted(filenames), lambda x: os.path.dirname(x))

def get_ext(filename):
  name, ext = os.path.splitext(filename)
  if ext == '.gz':
    ext = get_ext(name) + ext
  return ext

def strip_ext(filename):
  # strip of gz, assuming there will be another extension before .gz
  if filename.endswith('.gz'):
    filename = filename[:-3]
  return os.path.splitext(filename)[0]

def group_files_by_name_excl_ext(filenames):
  return sort_and_groupby_to_dict(filenames, strip_ext)

def zip_by_keys(*dict_list):
  keys = reduce(lambda agg, v: agg | set(v.keys()), dict_list, set())
  return (
    [d.get(k) for d in dict_list]
    for k in sorted(keys)
  )

def group_file_pairs_by_parent_directory_or_name(files_by_type):
  grouped_files_by_pattern = [
    group_files_by_parent_directory(files) for files in files_by_type
  ]
  for files_in_group_by_pattern in zip_by_keys(*grouped_files_by_pattern):
    if all(len(files or []) == 1 for files in files_in_group_by_pattern):
      yield tuple([files[0] for files in files_in_group_by_pattern])
    else:
      grouped_by_name = [
        group_files_by_name_excl_ext(files or [])
        for files in files_in_group_by_pattern
      ]
      for files_by_name in zip_by_keys(*grouped_by_name):
        if all(len(files or []) == 1 for files in files_by_name):
          yield tuple([files[0] for files in files_by_name])
        else:
          get_logger().info(
            'no exclusively matching files found: %s',
            [files for files in files_by_name]
          )

def find_file_pairs_grouped_by_parent_directory_or_name(patterns, limit=None):
  matching_files_by_pattern = [
    list(find_matching_filenames(pattern)) for pattern in patterns
  ]
  get_logger().info(
    'found number of files %s',
    ', '.join(
      '%s: %d' % (pattern, len(files))
      for pattern, files in zip(patterns, matching_files_by_pattern)
    )
  )
  patterns_without_files = [
    pattern
    for pattern, files in zip(patterns, matching_files_by_pattern)
    if len(files) == 0
  ]
  if patterns_without_files:
    raise RuntimeError('no files found for: %s' % patterns_without_files)
  return group_file_pairs_by_parent_directory_or_name(
    matching_files_by_pattern
  )

def convert_pdf_bytes_to_lxml(pdf_content, path=None, page_range=None):
  stop_watch_recorder = StopWatchRecorder()

  args = '-blocks -noImageInline -noImage -fullFontName'.split()
  if page_range:
    args += ['-f', str(page_range[0]), '-l', str(page_range[1])]

  stop_watch_recorder.start('convert to lxml')
  lxml_content = PdfToLxmlWrapper().process_input(
    pdf_content,
    args
  )
  stop_watch_recorder.stop()

  get_logger().info(
    'converted to lxml: path=%s, pdf size=%s, lxml size=%s, timings=[%s]',
    path, format(len(pdf_content), ','), format(len(lxml_content), ','),
    stop_watch_recorder
  )

  return lxml_content

def convert_and_annotate_lxml_content(lxml_content, xml_content, xml_mapping, name=None):
  stop_watch_recorder = StopWatchRecorder()

  stop_watch_recorder.start('parse lxml')
  lxml_root = etree.fromstring(lxml_content)

  # use a more lenient way to parse xml as xml errors are not uncomment
  stop_watch_recorder.start('parse xml')
  xml_root = xml_from_string_with_recover(xml_content)

  stop_watch_recorder.start('extract target annotations')
  target_annotations = xml_root_to_target_annotations(
    xml_root,
    xml_mapping
  )
  stop_watch_recorder.stop()

  annotators = DEFAULT_ANNOTATORS + [MatchingAnnotator(
    target_annotations
  )]
  annotator = Annotator(annotators)

  stop_watch_recorder.start('convert to svg')
  svg_roots = list(iter_svg_pages_for_lxml(lxml_root))

  stop_watch_recorder.start('annotate svg')
  annotator.annotate(SvgStructuredDocument(svg_roots))

  stop_watch_recorder.start('add visualisation')
  svg_roots = [
    visualize_svg_annotations(svg_root)
    for svg_root in svg_roots
  ]
  stop_watch_recorder.stop()

  get_logger().info(
    'processed: name=%s, lxml size=%s, xml size=%s, timings=[%s] (native align impl=%s)',
    name, format(len(lxml_content), ','), format(len(xml_content), ','),
    stop_watch_recorder, align_native_enabled
  )

  return svg_roots

def relative_path(base_path, path):
  if not base_path.endswith('/'):
    base_path += '/'
  return path[len(base_path):] if path.startswith(base_path) else path

def is_relative_path(path):
  return not path.startswith('/') and '://' not in path

def join_if_relative_path(base_path, path):
  return FileSystems.join(base_path, path) if is_relative_path(path) else path

def change_ext(path, old_ext, new_ext):
  if old_ext is None:
    old_ext = os.path.splitext(path)[1]
    if old_ext == '.gz':
      path = path[:-len(old_ext)]
      old_ext = os.path.splitext(path)[1]
  if old_ext and path.endswith(old_ext):
    return path[:-len(old_ext)] + new_ext
  else:
    return path + new_ext

def save_pages(output_filename, ext, bytes_by_page):
  mkdirs_if_not_exists(dirname(output_filename))
  with FileSystems.create(output_filename) as f:
    with ZipFile(f, 'w', compression=ZIP_DEFLATED) as zf:
      for i, data in enumerate(bytes_by_page):
        page_filename = 'page-%s%s' % (1 + i, ext)
        get_logger().debug('page_filename: %s', page_filename)
        zf.writestr(page_filename, data)
    return output_filename

def save_svg_roots(output_filename, svg_pages):
  return save_pages(output_filename, '.svg', (
    etree.tostring(svg_page)
    for svg_page in svg_pages
  ))

def pdf_bytes_to_png_pages(pdf_bytes, dpi, image_size, page_range=None):
  pdf_to_png = PdfToPng(dpi=dpi, image_size=image_size, page_range=page_range)
  return (
    fp.read()
    for fp in pdf_to_png.iter_pdf_bytes_to_png_fp(pdf_bytes)
  )

def svg_page_to_blockified_png_bytes(svg_page, color_map, image_size=None):
  structured_document = SvgStructuredDocument(svg_page)
  blocks = expand_blocks(
    merge_blocks(
      annotation_document_page_to_annotation_blocks(
        structured_document,
        structured_document.get_pages()[0]
      )
    )
  )
  viewbox = svg_page.attrib.get('viewBox')
  if not viewbox:
    raise RuntimeError(
      'viewbox missing on svg, available attributes: %s' % svg_page.attrib.keys()
    )
  _, _, width, height = viewbox.split()
  image = annotated_blocks_to_image(
    blocks, color_map,
    width=float(width), height=float(height), background='white',
    scale_to_size=image_size
  )
  out = BytesIO()
  image.save(out, 'png')
  return out.getvalue()

def filter_list_props_by_indices(d, indices, list_props):
  return {
    k: (
      [x for i, x in enumerate(v) if i in indices]
      if k in list_props
      else v
    )
    for k, v in iteritems(d)
  }

def get_page_indices_with_min_annotation_percentage(
  annotation_evaluation, min_annotation_percentage):

  return [
    i
    for i, page_evaluation in enumerate(annotation_evaluation)
    if page_evaluation['percentage'].get(None) <= (1 - min_annotation_percentage)
  ]

def parse_page_range(s):
  s = s.strip()
  if not s:
    return None
  a = tuple([int(x) for x in s.split('-')])
  if len(a) == 1:
    return (a[0], a[0])
  elif len(a) == 2:
    return a
  else:
    raise TypeError('invalid page range: %s' % s)
