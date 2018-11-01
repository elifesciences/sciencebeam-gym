from __future__ import absolute_import

import argparse
import os
import logging
from itertools import islice

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

from sciencebeam_utils.beam_utils.utils import (
    TransformAndCount,
    TransformAndLog,
    MapOrLog,
    PreventFusion
)

from sciencebeam_utils.beam_utils.csv import (
    WriteDictCsv,
    ReadDictCsv
)

from sciencebeam_utils.beam_utils.io import (
    read_all_from_path,
    basename,
    save_file_content
)

from sciencebeam_utils.beam_utils.main import (
    add_cloud_args,
    process_cloud_args
)

from sciencebeam_utils.utils.collection import (
    extend_dict,
    remove_keys_from_dict
)

from sciencebeam_utils.utils.file_path import (
    change_ext,
    relative_path,
    join_if_relative_path
)

from sciencebeam_utils.utils.file_pairs import (
    find_file_pairs_grouped_by_parent_directory_or_name,
)

from sciencebeam_gym.structured_document.svg import (
    SvgStructuredDocument
)

from sciencebeam_gym.preprocess.annotation.target_annotation import (
    parse_xml_mapping
)

from sciencebeam_gym.preprocess.color_map import (
    parse_color_map_from_file
)

from sciencebeam_gym.preprocess.annotation.annotation_evaluation import (
    evaluate_document_by_page,
    DEFAULT_EVALUATION_COLUMNS,
    to_csv_dict_rows as to_annotation_evaluation_csv_dict_rows
)

from sciencebeam_gym.preprocess.preprocessing_utils import (
    convert_pdf_bytes_to_lxml,
    convert_and_annotate_lxml_content,
    pdf_bytes_to_png_pages,
    svg_page_to_blockified_png_bytes,
    save_pages,
    save_svg_roots,
    filter_list_props_by_indices,
    get_page_indices_with_min_annotation_percentage,
    parse_page_range
)

from sciencebeam_gym.preprocess.preprocessing_transforms import (
    WritePropsToTFRecord
)


def get_logger():
    return logging.getLogger(__name__)


class MetricCounters(object):
    FILE_PAIR = 'file_pair_count'
    PAGE = 'page_count'
    FILTERED_PAGE = 'filtered_page_count'
    CONVERT_PDF_TO_LXML_ERROR = 'ConvertPdfToLxml_error_count'
    CONVERT_PDF_TO_PNG_ERROR = 'ConvertPdfToPng_error_count'
    CONVERT_LXML_TO_SVG_ANNOT_ERROR = 'ConvertPdfToSvgAnnot_error_count'


def configure_pipeline(p, opt):
    image_size = (
        (opt.image_width, opt.image_height)
        if opt.image_width and opt.image_height
        else None
    )
    page_range = opt.pages
    first_page = page_range[0] if page_range else 1
    xml_mapping = parse_xml_mapping(opt.xml_mapping_path)
    if opt.lxml_path:
        lxml_xml_file_pairs = (
            p |
            beam.Create([[
                join_if_relative_path(opt.base_data_path, s)
                for s in [opt.lxml_path, opt.xml_path]
            ]]) |
            "FindFilePairs" >> TransformAndLog(
                beam.FlatMap(
                    lambda patterns: islice(
                        find_file_pairs_grouped_by_parent_directory_or_name(patterns),
                        opt.limit
                    )
                ),
                log_prefix='file pairs: ',
                log_level='debug'
            ) |
            PreventFusion() |
            "ReadFileContent" >> beam.Map(lambda filenames: {
                'source_filename': filenames[0],
                'xml_filename': filenames[1],
                'lxml_content': read_all_from_path(filenames[0]),
                'xml_content': read_all_from_path(filenames[1])
            })
        )
    elif opt.pdf_path or opt.pdf_xml_file_list:
        if opt.pdf_xml_file_list:
            pdf_xml_url_pairs = (
                p |
                "ReadFilePairUrls" >> ReadDictCsv(opt.pdf_xml_file_list, limit=opt.limit) |
                "TranslateFilePairUrls" >> beam.Map(lambda row: (row['source_url'], row['xml_url']))
            )
        else:
            pdf_xml_url_pairs = (
                p |
                beam.Create([[
                    join_if_relative_path(opt.base_data_path, s)
                    for s in [opt.pdf_path, opt.xml_path]
                ]]) |
                "FindFilePairs" >> TransformAndLog(
                    beam.FlatMap(
                        lambda patterns: islice(
                            find_file_pairs_grouped_by_parent_directory_or_name(patterns),
                            opt.limit
                        )
                    ),
                    log_prefix='file pairs: ',
                    log_level='debug'
                )
            )
        pdf_xml_file_pairs = (
            pdf_xml_url_pairs |
            PreventFusion() |
            "ReadFileContent" >> TransformAndCount(
                beam.Map(lambda filenames: {
                    'source_filename': filenames[0],
                    'xml_filename': filenames[1],
                    'pdf_content': read_all_from_path(filenames[0]),
                    'xml_content': read_all_from_path(filenames[1])
                }),
                MetricCounters.FILE_PAIR
            )
        )

        lxml_xml_file_pairs = (
            pdf_xml_file_pairs |
            "ConvertPdfToLxml" >> MapOrLog(lambda v: remove_keys_from_dict(
                extend_dict(v, {
                    'lxml_content': convert_pdf_bytes_to_lxml(
                        v['pdf_content'], path=v['source_filename'],
                        page_range=page_range
                    )
                }),
                # we don't need the pdf_content unless we are writing tf_records
                None if opt.save_tfrecords else {'pdf_content'}
            ), log_fn=lambda e, v: (
                get_logger().warning(
                    'caught exception (ignoring item): %s, pdf: %s, xml: %s',
                    e, v['source_filename'], v['xml_filename'], exc_info=e
                )
            ), error_count=MetricCounters.CONVERT_PDF_TO_LXML_ERROR)
        )
    else:
        raise RuntimeError('either lxml-path or pdf-path required')

    if opt.save_png or opt.save_tfrecords:
        with_pdf_png_pages = (
            (lxml_xml_file_pairs if opt.save_tfrecords else pdf_xml_file_pairs) |
            "ConvertPdfToPng" >> MapOrLog(lambda v: remove_keys_from_dict(
                extend_dict(v, {
                    'pdf_png_pages': list(pdf_bytes_to_png_pages(
                        v['pdf_content'],
                        dpi=opt.png_dpi,
                        image_size=image_size,
                        page_range=page_range
                    ))
                }),
                {'pdf_content'}  # we no longer need the pdf_content
            ), error_count=MetricCounters.CONVERT_PDF_TO_PNG_ERROR)
        )

        if opt.save_png:
            _ = (
                with_pdf_png_pages |
                "SavePdfToPng" >> TransformAndLog(
                    beam.Map(lambda v: save_pages(
                        FileSystems.join(
                            opt.output_path,
                            change_ext(
                                relative_path(opt.base_data_path, v['source_filename']),
                                None, '.png.zip'
                            )
                        ),
                        '.png',
                        v['pdf_png_pages']
                    )),
                    log_fn=lambda x: get_logger().info('saved result: %s', x)
                )
            )

    if opt.save_lxml:
        _ = (
            lxml_xml_file_pairs |
            "SaveLxml" >> TransformAndLog(
                beam.Map(lambda v: save_file_content(
                    FileSystems.join(
                        opt.output_path,
                        change_ext(
                            relative_path(opt.base_data_path, v['source_filename']),
                            None, '.lxml.gz'
                        )
                    ),
                    v['lxml_content']
                )),
                log_fn=lambda x: get_logger().info('saved lxml: %s', x)
            )
        )

    annotation_results = (
        (with_pdf_png_pages if opt.save_tfrecords else lxml_xml_file_pairs) |
        "ConvertLxmlToSvgAndAnnotate" >> TransformAndCount(
            MapOrLog(lambda v: remove_keys_from_dict(
                extend_dict(v, {
                    'svg_pages': list(convert_and_annotate_lxml_content(
                        v['lxml_content'], v['xml_content'], xml_mapping,
                        name=v['source_filename']
                    ))
                }),
                # Won't need the XML anymore
                {'lxml_content', 'xml_content'}
            ), log_fn=lambda e, v: (
                get_logger().warning(
                    'caught exception (ignoring item): %s, source: %s, xml: %s',
                    e, v['source_filename'], v['xml_filename'], exc_info=e
                )
            ), error_count=MetricCounters.CONVERT_LXML_TO_SVG_ANNOT_ERROR),
            MetricCounters.PAGE,
            lambda v: len(v['svg_pages'])
        )
    )

    if opt.save_svg:
        _ = (
            annotation_results |
            "SaveSvgPages" >> TransformAndLog(
                beam.Map(lambda v: save_svg_roots(
                    FileSystems.join(
                        opt.output_path,
                        change_ext(
                            relative_path(opt.base_data_path, v['source_filename']),
                            None, '.svg.zip'
                        )
                    ),
                    v['svg_pages']
                )),
                log_fn=lambda x: get_logger().info('saved result: %s', x)
            )
        )

    if opt.annotation_evaluation_csv or opt.min_annotation_percentage:
        annotation_evaluation_results = (
            annotation_results |
            "EvaluateAnnotations" >> TransformAndLog(
                beam.Map(lambda v: remove_keys_from_dict(
                    extend_dict(v, {
                        'annotation_evaluation': evaluate_document_by_page(
                            SvgStructuredDocument(v['svg_pages'])
                        )
                    }),
                    None if opt.min_annotation_percentage else {'svg_pages'}
                )),
                log_fn=lambda x: get_logger().info(
                    'annotation evaluation result: %s: %s',
                    x['source_filename'], x['annotation_evaluation']
                )
            )
        )

    if opt.save_block_png or opt.save_tfrecords:
        color_map = parse_color_map_from_file(opt.color_map)
        with_block_png_pages = (
            (
                annotation_evaluation_results
                if opt.min_annotation_percentage
                else annotation_results
            ) |
            "GenerateBlockPng" >> beam.Map(lambda v: remove_keys_from_dict(
                extend_dict(v, {
                    'block_png_pages': [
                        svg_page_to_blockified_png_bytes(svg_page, color_map, image_size=image_size)
                        for svg_page in v['svg_pages']
                    ]
                }),
                {'svg_pages'}
            ))
        )

        if opt.save_block_png:
            _ = (
                with_block_png_pages |
                "SaveBlockPng" >> TransformAndLog(
                    beam.Map(lambda v: save_pages(
                        FileSystems.join(
                            opt.output_path,
                            change_ext(
                                relative_path(opt.base_data_path, v['source_filename']),
                                None, '.block-png.zip'
                            )
                        ),
                        '.png',
                        v['block_png_pages']
                    )),
                    log_fn=lambda x: get_logger().info('saved result: %s', x)
                )
            )

        if opt.save_tfrecords:
            if opt.min_annotation_percentage:
                filtered_pages = (
                    with_block_png_pages |
                    "FilterPages" >> TransformAndCount(
                        beam.Map(
                            lambda v: filter_list_props_by_indices(
                                v,
                                get_page_indices_with_min_annotation_percentage(
                                    v['annotation_evaluation'],
                                    opt.min_annotation_percentage
                                ),
                                {'pdf_png_pages', 'block_png_pages'}
                            )
                        ),
                        MetricCounters.FILTERED_PAGE,
                        lambda v: len(v['block_png_pages'])
                    )
                )
            else:
                filtered_pages = with_block_png_pages
            _ = (
                filtered_pages |
                "WriteTFRecords" >> WritePropsToTFRecord(
                    FileSystems.join(opt.output_path, 'data'),
                    lambda v: (
                        {
                            'input_uri': v['source_filename'] + '#page%d' % (first_page + i),
                            'input_image': pdf_png_page,
                            'annotation_uri': (
                                v['source_filename'] + '.annot' + '#page%d' % (first_page + i)
                            ),
                            'annotation_image': block_png_page,
                            'page_no': first_page + i
                        }
                        for i, pdf_png_page, block_png_page in zip(
                            range(len(v['pdf_png_pages'])), v['pdf_png_pages'], v['block_png_pages']
                        )
                    )
                )
            )

    if opt.annotation_evaluation_csv:
        annotation_evaluation_csv_name, annotation_evaluation_ext = (
            os.path.splitext(opt.annotation_evaluation_csv)
        )
        _ = (  # flake8: noqa
            annotation_evaluation_results |
            "FlattenAnotationEvaluationResults" >> beam.FlatMap(
                lambda v: to_annotation_evaluation_csv_dict_rows(
                    v['annotation_evaluation'],
                    document=basename(v['source_filename'])
                )
            ) |
            "WriteAnnotationEvaluationToCsv" >> WriteDictCsv(
                join_if_relative_path(opt.output_path, annotation_evaluation_csv_name),
                file_name_suffix=annotation_evaluation_ext,
                columns=DEFAULT_EVALUATION_COLUMNS
            )
        )


def add_main_args(parser):
    parser.add_argument(
        '--data-path', type=str, required=True,
        help='base data path'
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--lxml-path', type=str, required=False,
        help='path to lxml file(s)'
    )
    source_group.add_argument(
        '--pdf-path', type=str, required=False,
        help='path to pdf file(s) (alternative to lxml)'
    )
    source_group.add_argument(
        '--pdf-xml-file-list', type=str, required=False,
        help='path to pdf-xml csv/tsv file list'
    )
    parser.add_argument(
        '--limit', type=int, required=False,
        help='limit the number of file pairs to process'
    )

    parser.add_argument(
        '--save-lxml', default=False, action='store_true',
        help='save generated lxml (if using pdf as an input)'
    )

    parser.add_argument(
        '--save-svg', default=False, action='store_true',
        help='save svg pages with annotation tags'
    )

    parser.add_argument(
        '--save-png', default=False, action='store_true',
        help='save png pages of the original pdf'
    )
    parser.add_argument(
        '--png-dpi', type=int, default=90,
        help='dpi of rendered pdf pages'
    )

    parser.add_argument(
        '--image-width', type=int, required=False,
        help='image width of resulting PNGs'
    )
    parser.add_argument(
        '--image-height', type=int, required=False,
        help='image height of resulting PNGs'
    )

    parser.add_argument(
        '--save-block-png', default=False, action='store_true',
        help='save blockified version of the svg as a png'
    )
    parser.add_argument(
        '--color-map', default='color_map.conf',
        help='color map to use (see save-block-png)'
    )

    parser.add_argument(
        '--xml-path', type=str, required=False,
        help='path to xml file(s)'
    )
    parser.add_argument(
        '--xml-mapping-path', type=str, default='annot-xml-front.conf',
        help='path to xml mapping file'
    )

    parser.add_argument(
        '--pages', type=parse_page_range, default=None,
        help='only processes the selected pages'
    )

    parser.add_argument(
        '--save-tfrecords', default=False, action='store_true',
        help='Save TFRecords with PDF PNG and Annotation PNG'
        ' (--image-width and --image-height recommended)'
    )

    parser.add_argument(
        '--min-annotation-percentage', type=float, required=False,
        help='Minimum percentage of annotations per page'
        ' (pages below that threshold will get dropped)'
    )

    parser.add_argument(
        '--annotation-evaluation-csv', type=str, required=False,
        help='Annotation evaluation CSV output file'
    )
    parser.add_argument(
        '--output-path', required=False,
        help='Output directory to write results to.'
    )


def process_main_args(parser, args):
    args.base_data_path = args.data_path.replace('/*/', '/')

    if not args.output_path:
        args.output_path = os.path.join(
            os.path.dirname(args.base_data_path),
            os.path.basename(args.base_data_path + '-results')
        )

    if not args.xml_path and not args.pdf_xml_file_list:
        parser.error('--xml-path required unless --pdf-xml-file-list is specified')

    pdf_path_or_pdf_xml_file_list = args.pdf_path or args.pdf_xml_file_list

    if args.save_lxml and not pdf_path_or_pdf_xml_file_list:
        parser.error('--save-lxml only valid with --pdf-path or --pdf-xml-file-list')

    if args.save_png and not pdf_path_or_pdf_xml_file_list:
        parser.error('--save-png only valid with --pdf-path or --pdf-xml-file-list')

    if args.save_tfrecords and not pdf_path_or_pdf_xml_file_list:
        parser.error('--save-tfrecords only valid with --pdf-path or --pdf-xml-file-list')

    if sum(1 if x else 0 for x in (args.image_width, args.image_height)) == 1:
        parser.error('--image-width and --image-height need to be specified together')

    if not (args.save_lxml or args.save_svg or args.save_png or args.save_tfrecords):
        parser.error(
            'at least one of the output options required:'
            ' --save-lxml --save-svg --save-png or --save-tfrecords'
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    add_main_args(parser)
    add_cloud_args(parser)

    # parsed_args, other_args = parser.parse_known_args(argv)
    parsed_args = parser.parse_args(argv)

    process_main_args(parser, parsed_args)
    process_cloud_args(
        parsed_args, parsed_args.output_path,
        name='sciencbeam-gym-preprocessing'
    )

    get_logger().info('parsed_args: %s', parsed_args)

    return parsed_args


def run(argv=None):
    """Main entry point; defines and runs the tfidf pipeline."""
    known_args = parse_args(argv)

    # We use the save_main_session option because one or more DoFn's in this
    # workflow rely on global context (e.g., a module imported at module level).
    pipeline_options = PipelineOptions.from_dictionary(vars(known_args))
    pipeline_options.view_as(SetupOptions).save_main_session = True

    with beam.Pipeline(known_args.runner, options=pipeline_options) as p:
        configure_pipeline(p, known_args)

        # Execute the pipeline and wait until it is completed.


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    run()
