import logging
import warnings
import re

from .main import model


# This is for hiding flytekit errors logs to declutter the unionml demo
warnings.simplefilter("ignore")

class Filter(logging.Filter):
    def filter(self, record):
        return not re.match("^Error from command.+", record.getMessage())


flytekit_logger = logging.getLogger("flytekit")
flytekit_logger.addFilter(Filter())
