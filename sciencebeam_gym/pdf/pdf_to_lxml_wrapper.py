import logging
import subprocess
from subprocess import PIPE
import os
from zipfile import ZipFile
from shutil import rmtree
from urllib.request import urlretrieve
from tempfile import NamedTemporaryFile

from sciencebeam_utils.utils.io import makedirs
from sciencebeam_utils.utils.zip import extract_all_with_executable_permission


def get_logger():
    return logging.getLogger(__name__)


def iter_read_lines(reader):
    while True:
        line = reader.readline()
        if not line:
            break
        yield line


def stream_lines_to_logger(lines, logger, prefix=''):
    for line in lines:
        line = line.strip()
        if line:
            logger.info('%s%s', prefix, line)


def download_if_not_exist(url, target_file):
    if not os.path.isfile(target_file):
        get_logger().info('downloading %s to %s', url, target_file)

        makedirs(os.path.dirname(target_file), exists_ok=True)

        temp_filename = target_file + '.part'
        if os.path.isfile(temp_filename):
            os.remove(temp_filename)
        urlretrieve(url, temp_filename)
        os.rename(temp_filename, target_file)
    return target_file


def unzip_if_not_exist(zip_filename, target_directory, ignore_subdirectory=None):
    if ignore_subdirectory is None:
        ignore_subdirectory = os.path.basename(zip_filename)
    if not os.path.isdir(target_directory):
        get_logger().info('unzipping %s to %s', zip_filename, target_directory)
        temp_target_directory = target_directory + '.part'
        if os.path.isdir(temp_target_directory):
            rmtree(temp_target_directory)

        with ZipFile(zip_filename, 'r') as zf:
            extract_all_with_executable_permission(zf, temp_target_directory)
            # ignore first level in the directory structure, if applicable
            sub_dir = (
                os.path.join(temp_target_directory, ignore_subdirectory)
                if ignore_subdirectory
                else target_directory
            )
            if os.path.isdir(sub_dir):
                os.rename(sub_dir, target_directory)
                rmtree(temp_target_directory)
            else:
                os.rename(temp_target_directory, target_directory)
    return target_directory


class PdfToLxmlWrapper(object):
    def __init__(self):
        temp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.temp'))
        self.target_directory = os.path.join(temp_dir, 'pdf2xml')
        self.zip_filename = os.path.join(temp_dir, 'pdf2xml.zip')
        self.zip_url = (
            'https://storage.googleapis.com/elife-ml/artefacts/pdf2xml-linux-64.zip'
        )

    def download_pdf2xml_zip_if_not_exist(self):
        download_if_not_exist(
            self.zip_url,
            self.zip_filename
        )

    def unzip_pdf2xml_zip_if_target_directory_does_not_exist(self):
        if not os.path.isdir(self.target_directory):
            self.download_pdf2xml_zip_if_not_exist()
            unzip_if_not_exist(self.zip_filename, self.target_directory)

    def get_pdf2xml_executable_path(self):
        self.unzip_pdf2xml_zip_if_target_directory_does_not_exist()
        # use pdftoxml_server as it already handles timeouts
        return os.path.join(
            self.target_directory,
            'lin-64/pdftoxml'
        )

    def process_input(self, source_data, args):
        with NamedTemporaryFile() as f:
            f.write(source_data)
            f.flush()
            os.fsync(f)
            return self.process_file(f.name, args)

    def process_file(self, source_filename, args):
        pdf2xml = self.get_pdf2xml_executable_path()
        get_logger().info('processing %s using %s', source_filename, pdf2xml)
        cmd = [pdf2xml] + args + [source_filename, '-']
        timeout_bin = '/usr/bin/timeout'
        if not os.path.exists(timeout_bin):
            timeout_bin = 'timeout'
        with subprocess.Popen(
            [timeout_bin, '20s'] + cmd,
            stdout=PIPE,
            stderr=PIPE,
            stdin=None
        ) as p:
            out, err = p.communicate()
            return_code = p.returncode
        if return_code != 0:
            get_logger().warning(
                'process failed with %d, stderr=%s, stdout=%.200s...',
                return_code, err, out
            )
            raise RuntimeError('process failed with %d, stderr: %s' % (return_code, err))
        if len(out) == 0:
            get_logger().warning(
                'process returned empty response (code %d), stderr=%s, stdout=%.500s...',
                return_code, err, out
            )
            raise RuntimeError(
                'process returned empty response (code %d), stderr: %s' % (return_code, err)
            )
        get_logger().info(
            'received response for %s (%s bytes)',
            source_filename, format(len(out), ',')
        )
        return out


if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    sample_pdf_url = 'https://rawgit.com/elifesciences/XML-mapping/master/elife-00666.pdf'
    sample_pdf_filename = '.temp/elife-00666.pdf'
    download_if_not_exist(sample_pdf_url, sample_pdf_filename)
    with open(sample_pdf_filename, 'rb') as sample_f:
        sample_pdf_contents = sample_f.read()
    get_logger().info('pdf size: %s bytes', format(len(sample_pdf_contents), ','))
    process_out = PdfToLxmlWrapper().process_input(
        sample_pdf_contents,
        '-blocks -noImageInline -noImage -fullFontName'.split()
    )
    get_logger().info('out: %.1000s...', process_out)
