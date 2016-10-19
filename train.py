import os

import math
import tensorflow as tf
import time
import click
import dill
import logging

from skipthought import SkipthoughtModel
from skipthought.data_utils import TextData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--data_path', default='data/input.txt', type=str, help="Path to file with text data.")
@click.option('--save_dir', default='save/', type=str, help='Directory to store checkpointed models')
@click.option('--num_epochs', default=10, type=int, help="Number of epochs.")
@click.option('--num_layers', default=1, type=int, help="Number of layers in encoder.")
@click.option('--batch_size', default=128, type=int, help="The size of batch.")
@click.option('--max_len', default=100, type=int, help='Maximum sequence length in encoder and decoder.'
                                                       'Lines with higher length will be cutted.')
@click.option('--num_hidden', default=512, type=int, help="Hidden size of the cell.")
@click.option('--cell_type', default='gru',
              type=click.Choice(SkipthoughtModel.SUPPORTED_CELLTYPES),
              help='Type of the RNN cell.')
@click.option('--embedding_size', default=300, type=int, help="The size of word embeddings.")
@click.option('--max_vocab_size', default=100000, type=int, help="Size of vocabulary. Most frequent words are used.")
@click.option('--num_samples', default=512, type=int, help="Number of samples in sampled softmax.")
@click.option('--learning_rate', default=0.01, type=float, help="Initial learning rate.")
@click.option('--decay_rate', default=0.99, type=float, help="Exponential decay rate.")
@click.option('--grad_clip', default=5.0, type=float, help="Value for gradient clipping.")
@click.option('--save_every', default=1000, type=int, help="Number of batch steps before creating a model checkpoint")
@click.option('--verbose', default=100, type=int, help="How often to print batch loss and other info.")
@click.option('--init_from', default=None, help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'chars_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
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
            need_be_same = ["cell_type", "num_hidden", "num_layers", "num_samples", "max_vocab_size"]
            for checkme in need_be_same:
                assert saved_model_args[checkme] == kwargs[checkme], "Command line argument and saved model disagree on '%s' " % checkme
        logger.info("Args checker. Load TextData")
        # open saved TextData
        textdata = TextData.load(os.path.join(kwargs['init_from'], 'textdata.pkl'))
    else:
        textdata = TextData(kwargs['data_path'], max_len=kwargs['max_len'], max_vocab_size=kwargs['max_vocab_size'])

    logger.info("Save config and textdata.")
    with open(os.path.join(kwargs['save_dir'], 'config.pkl'), 'wb') as f:
        dill.dump(kwargs, f)
    TextData.save(textdata, os.path.join(kwargs['save_dir'], 'textdata.pkl'))

    # Make triples.
    logger.info("Making triples")
    triples = textdata.make_triples(textdata.dataset)
    logger.info("Number of triples: {0}".format(len(triples[0])))
    decay_steps = len(triples[0])
    vocab_size = len(textdata.vocab)
    logger.info("actual vocab_size={0}".format(vocab_size))

    model = SkipthoughtModel(kwargs['cell_type'], kwargs['num_hidden'], kwargs['num_layers'],
                             kwargs['embedding_size'], vocab_size, kwargs['learning_rate'],
                             kwargs['decay_rate'], decay_steps, kwargs['grad_clip'],
                             kwargs['num_samples'], kwargs['max_len'])

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=20)

        if kwargs['init_from'] is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Restored from {0}".format(ckpt.model_checkpoint_path))

        num_batches = len(triples[0])//kwargs['batch_size']
        loss_history = []
        for e in range(kwargs['num_epochs']):
            it = textdata.triples_data_iterator(triples[0], triples[1], triples[2],
                                                textdata.max_len, kwargs['batch_size'], shuffle=True)
            for b, batch in enumerate(it):
                train_op, loss, feed_dict = model.train_step(*batch)

                start_time = time.time()
                batch_loss, _ = sess.run([loss, train_op], feed_dict=feed_dict)
                batch_perplexity = math.exp(float(batch_loss)) if batch_loss < 300 else float("inf")
                end_time = time.time()

                loss_history.append(batch_loss)
                if b % kwargs['verbose'] == 0:
                            print(
                                "{}/{} (epoch {}), train_loss = {:.3f}, perplexity = {:.3f}, time/batch = {:.3f}" \
                                .format(e * num_batches + b,
                                        kwargs['num_epochs'] * num_batches,
                                        e, batch_loss, batch_perplexity, end_time - start_time))
                if (e * num_batches + b) % kwargs['save_every'] == 0 \
                        or (e == kwargs['num_epochs']-1 and b == num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(kwargs['save_dir'], 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * num_batches + b)
                    with open(os.path.join(kwargs['save_dir'], 'loss_history.pkl'), 'wb') as f:
                        dill.dump(loss_history, f)
                    print("model & loss_history saved to {}".format(checkpoint_path))

if __name__ == "__main__":
    main()
