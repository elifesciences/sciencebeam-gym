import matplotlib as mpl
# this is important to run on the cloud - we won't have python-tk installed
mpl.use("Agg")

# pylint: disable=unused-import, wrong-import-position
from matplotlib import pyplot  # flake8: noqa
