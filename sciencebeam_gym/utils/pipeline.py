import argparse
import concurrent.futures
import logging
from abc import abstractmethod
from typing import Callable, Generic, List, TypeVar

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions


from sciencebeam_utils.beam_utils.utils import (
    PreventFusion
)

from sciencebeam_utils.utils.progress_logger import logging_tqdm
from sciencebeam_utils.tools.check_file_list import map_file_list_to_file_exists

from sciencebeam_utils.beam_utils.main import (
    add_cloud_args,
    process_cloud_args
)


LOGGER = logging.getLogger(__name__)


T_Item = TypeVar('T_Item')


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
    add_cloud_args(parser)


def process_pipeline_args(
    args: argparse.Namespace,
    output_path: str
):
    process_cloud_args(args, output_path=output_path)


class AbstractPipelineFactory(Generic[T_Item]):
    def __init__(
        self,
        resume: bool = False
    ):
        self.resume = resume

    @abstractmethod
    def process_item(self, item: T_Item):
        pass

    @abstractmethod
    def get_item_list(self) -> List[T_Item]:
        pass

    @abstractmethod
    def get_output_file_for_item(self, item: T_Item) -> str:
        pass

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
        _ = (
            p
            | beam.Create(item_list)
            | PreventFusion()
            | "Process Item" >> beam.Map(self.process_item)
        )

    def run_beam_pipeline(
        self,
        args: argparse.Namespace,
        item_list: List[T_Item],
        save_main_session: bool = False
    ):
        pipeline_options = PipelineOptions.from_dictionary(vars(args))
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
                    executor.submit(self.process_item, item): item
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

        if not args.cloud and args.num_workers >= 1:
            self.run_local_pipeline(args, item_list=item_list)
            return

        self.run_beam_pipeline(
            args,
            save_main_session=save_main_session,
            item_list=item_list
        )
