import argparse
import concurrent.futures
import logging
from abc import abstractmethod
from typing import Callable, Generic, List, TypeVar

import apache_beam as beam
from apache_beam.options.pipeline_options import (
    DirectOptions,
    PipelineOptions,
    SetupOptions,
    WorkerOptions
)


from sciencebeam_utils.beam_utils.utils import (
    Count,
    MapOrLog,
    PreventFusion,
    TransformAndCount
)

from sciencebeam_utils.utils.progress_logger import logging_tqdm
from sciencebeam_utils.tools.check_file_list import map_file_list_to_file_exists

from sciencebeam_utils.beam_utils.main import (
    add_cloud_args,
    process_cloud_args
)


LOGGER = logging.getLogger(__name__)


T_Item = TypeVar('T_Item')


class MetricCounters(object):
    ITEM_COUNT = 'item_count'
    ERROR_COUNT = 'error_count'


def get_item_list_without_output_file(
    item_list: List[T_Item],
    get_output_file_for_item: Callable[[T_Item], str]
) -> List[T_Item]:
    output_file_exists_list = map_file_list_to_file_exists([
        get_output_file_for_item(item)
        for item in item_list
    ])
    LOGGER.debug('output_file_exists_list: %s', output_file_exists_list)
    return [
        item
        for item, output_file_exists in zip(item_list, output_file_exists_list)
        if not output_file_exists
    ]


def add_pipeline_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume processing (skip files that already have an output file)'
    )
    parser.add_argument(
        '--multi-processing',
        action='store_true',
        default=False,
        help='enable multi processing rather than multi threading'
    )
    parser.add_argument(
        '--skip-errors',
        action='store_true',
        help='Skip errors processing documents (no output will be generated for those documents)'
    )
    parser.add_argument(
        '--use-beam',
        action='store_true',
        help='Use Apache Beam pipeline even when running locally'
    )
    add_cloud_args(parser)
    direct_runner_parser = parser.add_argument_group('direct runner', conflict_handler='resolve')
    DirectOptions._add_argparse_args(direct_runner_parser)  # pylint: disable=protected-access
    worker_parser = parser.add_argument_group('worker', conflict_handler='resolve')
    WorkerOptions._add_argparse_args(worker_parser)  # pylint: disable=protected-access


def process_pipeline_args(
    args: argparse.Namespace,
    output_path: str
):
    process_cloud_args(args, output_path=output_path)


class AbstractPipelineFactory(Generic[T_Item]):
    def __init__(
        self,
        resume: bool,
        skip_errors: bool
    ):
        self.resume = resume
        self.skip_errors = skip_errors

    @staticmethod
    def get_init_kwargs_for_parsed_args(args: argparse.Namespace) -> dict:
        return {
            'resume': args.resume,
            'skip_errors': args.skip_errors
        }

    @abstractmethod
    def process_item(self, item: T_Item):
        pass

    @abstractmethod
    def get_item_list(self) -> List[T_Item]:
        pass

    @abstractmethod
    def get_output_file_for_item(self, item: T_Item) -> str:
        pass

    def process_item_or_skip_errors(self, item: T_Item):
        try:
            self.process_item(item)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error('failed to process item: %r due to %r', item, exc, exc_info=True)
            if not self.skip_errors:
                raise

    def get_remaining_item_list(self) -> List[T_Item]:
        item_list = self.get_item_list()
        LOGGER.debug('item_list: %s', item_list)

        if not item_list:
            LOGGER.warning('no files found')
            return item_list

        LOGGER.info('total number of files: %d', len(item_list))
        if self.resume:
            item_list = get_item_list_without_output_file(
                item_list,
                get_output_file_for_item=self.get_output_file_for_item
            )
            LOGGER.info('remaining number of files: %d', len(item_list))
        return item_list

    def configure_beam_pipeline(
        self,
        p: beam.Pipeline,
        item_list: List[T_Item]
    ):
        _pipeline = (
            p
            | beam.Create(item_list)
            | PreventFusion()
        )
        if self.skip_errors:
            LOGGER.debug('configuring pipeline, skipping errors')
            _pipeline |= (
                "Process Item" >> MapOrLog(
                    self.process_item,
                    error_count=MetricCounters.ERROR_COUNT
                )
                | "Count" >> Count(MetricCounters.ITEM_COUNT, counter_value_fn=None)
            )
        else:
            LOGGER.debug('configuring pipeline, not skipping errors')
            _pipeline |= (
                "Process Item" >> TransformAndCount(
                    beam.Map(self.process_item),
                    MetricCounters.ITEM_COUNT
                )
            )

    def run_beam_pipeline(
        self,
        args: argparse.Namespace,
        item_list: List[T_Item],
        save_main_session: bool = False
    ):
        if args.num_workers == 0:
            args.num_workers = None
        pipeline_options = PipelineOptions.from_dictionary({
            key: value
            for key, value in vars(args).items()
            if value is not None
        })
        pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
        LOGGER.info('save_main_session: %r', save_main_session)
        LOGGER.info('pipeline_options: %r', vars(pipeline_options))

        with beam.Pipeline(args.runner, options=pipeline_options) as p:
            self.configure_beam_pipeline(p, item_list=item_list)

            # Execute the pipeline and wait until it is completed.

    def run_local_pipeline(self, args: argparse.Namespace, item_list: List[T_Item]):
        num_workers = min(args.num_workers, len(item_list))
        multi_processing = args.multi_processing
        LOGGER.info('using %d workers (multi_processing: %s)', num_workers, multi_processing)
        PoolExecutor = (
            concurrent.futures.ProcessPoolExecutor if multi_processing
            else concurrent.futures.ThreadPoolExecutor
        )
        with PoolExecutor(max_workers=num_workers) as executor:
            with logging_tqdm(total=len(item_list), logger=LOGGER) as pbar:
                future_to_url = {
                    executor.submit(self.process_item_or_skip_errors, item): item
                    for item in item_list
                }
                LOGGER.debug('future_to_url: %s', future_to_url)
                for future in concurrent.futures.as_completed(future_to_url):
                    pbar.update(1)
                    future.result()

    def run(self, args: argparse.Namespace, save_main_session: bool = False):
        item_list = self.get_remaining_item_list()
        if not item_list:
            LOGGER.warning('no files to process')
            return

        if not args.use_beam and not args.cloud and args.num_workers >= 1:
            self.run_local_pipeline(args, item_list=item_list)
            return

        self.run_beam_pipeline(
            args,
            save_main_session=save_main_session,
            item_list=item_list
        )
