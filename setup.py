from __future__ import print_function

import os
import subprocess
import shlex
from getpass import getuser

from distutils.command.build import build  # type: ignore

from setuptools import (
    find_packages,
    setup,
    Command
)

import numpy as np


CUSTOM_COMMANDS = [
    shlex.split(command_line) for command_line in [
        'apt-get update',
        'apt-get --assume-yes install libxml2',
        'apt-get --assume-yes install poppler-utils',
        'apt-get --assume-yes install libgl1'
    ]
]

with open(os.path.join('requirements.txt'), 'r', encoding='utf-8') as f:
    REQUIRED_PACKAGES = f.readlines()

packages = find_packages()

# This class handles the pip install mechanism.


class CustomBuild(build):
    """A build command class that will be invoked during package install.
    The package built using the current setup.py will be staged and later
    installed in the worker using `pip install package'. This class will be
    instantiated during install for this specific scenario and will trigger
    running the custom commands specified.
    """
    sub_commands = build.sub_commands + [('CustomCommands', None)]


class CustomCommands(Command):
    """A setuptools Command class able to run arbitrary commands."""

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def _run_custom_command(self, command_list):
        if getuser() != 'root' or os.environ.get('SCIENCEBEAM_GYM_NO_APT'):
            print('Skipping setup command (not root): %s' % command_list)
            return
        print('Running command: %s' % command_list)
        p = subprocess.Popen(
            command_list,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        # Can use communicate(input='y\n'.encode()) if the command run requires
        # some confirmation.
        stdout_data, _ = p.communicate()
        print('Command output: %s' % stdout_data)
        if p.returncode != 0:
            raise RuntimeError(
                'Command %s failed: exit code: %s (output: %s)' %
                (command_list, p.returncode, stdout_data)
            )

    def run(self):
        for command in CUSTOM_COMMANDS:
            self._run_custom_command(command)


setup(
    name='sciencebeam_gym',
    version='0.0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=packages,
    include_package_data=True,
    description='ScienceBeam Gym',
    include_dirs=[np.get_include()],
    cmdclass={
        'build': CustomBuild,
        'CustomCommands': CustomCommands
    }
)
