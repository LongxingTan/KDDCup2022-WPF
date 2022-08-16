""" 
Data pipeline for model
""" 

import tensorflow as tf
import random
random.seed(315)


class DataReader(object):
    def __init__(
        self, 
        data, 
        train_sequence_length=10*24*6, 
        predict_sequence_length=48 * 6, 
        idx=None,  # start idx for 
        target_aggs=1,
        target_column_idx=[-2, -1],  # target and target
        feature_column_idx_short=[0],
        feature_column_idx_long=[-2, -1],
        ): 
        """ 
        data: 2D array, for each idx, choose its history and target
        """    
        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length
        self.target_aggs = target_aggs
        self.target_column_idx = target_column_idx
        self.feature_column_idx_short = feature_column_idx_short
        self.feature_column_idx_long = feature_column_idx_long

        if idx is None:
            drop_idx = data.groupby(['TurbID']).tail(predict_sequence_length - 1).index.tolist()
            drop_idx += data.groupby(['TurbID']).head(train_sequence_length).index.tolist()
            all_idx = data.index.to_list()
            self.idx = [i for i in all_idx if i not in drop_idx]
        else:
            self.idx = idx
        self.data = data.values
    
    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        train_begin_idx = idx - self.train_sequence_length
        test_end_idx = idx + self.predict_sequence_length
        turbine_id = self.data[idx, 0]
        raw= self.data[train_begin_idx: idx, self.feature_column_idx_short]  # train_seq * 10
        raw_long = self.data[train_begin_idx: test_end_idx, self.feature_column_idx_long]
        
        teacher = self.data[idx: test_end_idx, self.target_column_idx[0]: self.target_column_idx[0]+1]
        target = self.data[idx: test_end_idx, self.target_column_idx]
        
        return {'inputs': (turbine_id, raw, raw_long), 'teacher': teacher}, target

    def iter(self):
        for i in self.idx:
            yield self[i]


class DataLoader(object):
    def __init__(self, data_reader, short_feature_size=10, long_feature_size=2, target_column_size=2):
        self.data_reader = data_reader
        self.train_sequence_length = data_reader.train_sequence_length
        self.predict_sequence_length = data_reader.predict_sequence_length
        self.target_aggs = data_reader.target_aggs
        self.short_feature_size = short_feature_size
        self.long_feature_size = long_feature_size
        self.target_column_size = target_column_size

    def __call__(self, batch_size, shuffle=False, drop_remainder=False): 
        dataset = tf.data.Dataset.from_generator(
            self.data_reader.iter,
            # output_types=({'inputs':(tf.int32, tf.float32, tf.float32), 'teacher': tf.float32}, tf.float32),
            output_signature=({'inputs': 
            (tf.TensorSpec(shape=(), dtype=tf.int32), 
            tf.TensorSpec(shape=(self.train_sequence_length, self.short_feature_size), dtype=tf.float32),
            tf.TensorSpec(shape=(self.train_sequence_length+self.predict_sequence_length, self.long_feature_size), dtype=tf.float32)), 
            'teacher': tf.TensorSpec(shape=(self.predict_sequence_length//self.target_aggs, 1), dtype=tf.float32)}, 
            tf.TensorSpec(shape=(self.predict_sequence_length//self.target_aggs, self.target_column_size ), dtype=tf.float32)
        ))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
