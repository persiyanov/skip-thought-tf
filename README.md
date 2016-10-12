# skip-thought-tf
An implementation of skip-thought vectors in Tensorflow

# Usage
```python3
from skipthought import SkipthoughtModel
from skipthought.data_utils import TextData
from skipthought.utils import seq2seq_triples_data_iterator

model = SkipthoughtModel(...)

td = TextData("path/to/data")
lines = td.dataset

prev, curr, next = td.make_triples(lines)
it = td.triples_data_iterator(prev, curr, next, td.max_len, batch_size)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    
    for enc_inp, prev_inp, prev_targ, next_inp, next_targ in it:
        ....
    
```