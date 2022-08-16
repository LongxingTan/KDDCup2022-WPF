import os
import gc
import zipfile
import random
import numpy as np
import shutil
import tensorflow as tf

import time
import logging
from contextlib import contextmanager
from typing import Union, Optional


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


def compress_submitted_zip(res_dir='../../submit', output_dir='../../weights/result'):
    # z = zipfile.ZipFile(output_dir + '.zip', 'w')
    # for d in os.listdir(res_dir):
    #     z.write(res_dir + os.sep + d)
    # z.close()
    shutil.make_archive(output_dir, 'zip', res_dir)
    print('compressed to {}'.format(output_dir))
    return


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    msg = f"[{name}] start"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)


if __name__ == '__main__':
    compress_submitted_zip()
