import errno
import logging
import os
import dill as pickle

from DKG import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(module)s.%(funcName)s] %(message)s')
_logger = logging.getLogger(__name__)


def load_from_cache(pickle_file, cache_dir=None):
    if cache_dir is None:
        cache_dir = settings.CACHE_ROOT

    pickle_full_file = os.path.join(cache_dir, pickle_file)
    try:
        with open(pickle_full_file, 'rb') as handle:
            data = pickle.load(handle)
            _logger.info("Loaded pickled data from {}.".format(pickle_full_file))
            return data
    except Exception as e:
        _logger.info("Failed to load pickled data from {}.".format(pickle_full_file))
        print(e)
        raise


def write_to_cache(data, pickle_file, cache_dir=None):
    if cache_dir is None:
        cache_dir = settings.CACHE_ROOT

    pickle_full_file = os.path.join(cache_dir, pickle_file)
    _logger.info("Pickling data to {} ...".format(pickle_full_file))

    directory = os.path.dirname(pickle_full_file)
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    try:
        with open(pickle_full_file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _logger.info("Done")

    except Exception:
        _logger.info("Failed.")
        raise
