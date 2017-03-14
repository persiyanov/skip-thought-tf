import os

import math
import time
import dill
import logging
import tensorflow as tf

from skipthought import SkipthoughtModel
from skipthought.data_utils import TextData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(**kwargs):
    logger.info("Your params:")
    logger.info(kwargs)

    # check compatibility if training is continued from previously saved model
    if kwargs['init_from'] is not None:
        logger.info("Check if I can restore model from {0}".format(kwargs['init_from']))
        # check if all necessary files exist
        assert os.path.isdir(kwargs['init_from']), "%s must be a a path" % kwargs['init_from']
        assert os.path.isfile(os.path.join(kwargs['init_from'], "config.pkl")), "config.pkl file does not exist in path %s" % kwargs['init_from']
        assert os.path.isfile(os.path.join(kwargs['init_from'], "textdata.pkl")), "textdata.pkl file does not exist in path %s" % kwargs['init_from']
        ckpt = tf.train.get_checkpoint_state(kwargs['init_from'])
        assert ckpt, "No checkpoint found"
        assert ckpt.model_checkpoint_path, "No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(kwargs['init_from'], 'config.pkl'), 'rb') as f:
            saved_model_args = dill.load(f)
            saved_model_args.update(kwargs)
            kwargs = saved_model_args
            logger.info("Args load done.")
            logger.info(kwargs)
        logger.info("Load TextData")
        textdata = TextData(kwargs['data_path'], max_len=kwargs['max_len'], max_vocab_size=kwargs['max_vocab_size'])
    else:
        logger.error('Argument init_from is needed')


if __name__ == "__main__":
    main(init_from='./save', data_path='./data/input.txt')
