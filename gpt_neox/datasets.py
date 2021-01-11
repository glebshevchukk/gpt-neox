import torch
from torch.utils.data import Dataset
from .data_utils import get_tokenizer, natural_sort, skip, FixedSizeOrderedDict
import random
import glob
import tensorflow as tf
import re
import logging
from itertools import cycle
import simdjson as json
import linecache
import numpy as np

PAD_TOKEN=50257

"""
Dataset that gets sequences from a set of sharded jsonl files
"""
class JsonShardedDataset(Dataset):

    def __init__(self, shards_filename:str, tokenizer, seq_length:int,
                 initial_seed:int):
        # Input parameters.
        self.shards_filename = shards_filename
        self.seq_length = seq_length
        self.initial_seed = initial_seed

        #TODO: Check if using linecache actually helps
        self.shard_summary = json.loads(linecache.getline(self.shards_filename,1))
        self.total_files = len(self.shard_summary['file_names'])

        self.data_length = self.shard_summary['total_shards']
        self.tokenizer = tokenizer

    def __len__(self):
        return self.data_length
    
    def tokenize(self,data):
        return self.tokenizer(data, max_length=self.seq_length, return_tensors='pt',\
                padding='max_length', truncation=True)['input_ids']

    def __getitem__(self, idx):
        try:
            shard = json.loads(linecache.getline(self.shards_filename,idx+2))
        except:
            print("Could not load: ", self.shards_filename, idx+2)
            return None
        try:
            file_idx,shard_line = shard
            filename = self.shard_summary['file_names'][file_idx-1]   
        except:
            print("Could not get the right file name", self.shard_summary['file_names'], file_idx-1)
            return None
        try:
            #if we've already tokenized the input, all we have to do is pad it 
            if not self.tokenizer:
                line = linecache.getline(filename,shard_line)
                line = list(json.loads(line))

                if len(line) == 0:
                    print("No words")
                    raise Exception("An example has no words in it.")

                if len(line) < self.seq_length:
                    line.extend([PAD_TOKEN for _ in range(self.seq_length-len(line))])
                line = torch.IntTensor(line)
         
                return line, line[1:]
            #otherwise, we have to tokenize on the fly
            else:
                loaded_line = linecache.getline(filename,shard_line)
  
                line = list(json.loads(loaded_line))
                if len(line) == 0:
                    raise Exception("An example has no words in it.")
                data =  " ".join(line)
          
                tokenized = self.tokenize(data)

                if len(tokenized) == 0:
                    raise Exception("A tokenized example has no words in it.")
                
                return tokenized[0], tokenized[0,1:]
        except:
            print("Error: filename: ",filename,", shard line: ",shard_line)
           
            return None



class GPT2Dataset(Dataset):

    def __init__(self, glob_pattern, seq_len, seed=1, shuffle_input_filenames=True, pretokenized=True,
                 filetype="tfrecords", mode="chunks", train=True, tokenizer=None, **kwargs):

        super().__init__()
        self.files = glob.glob(glob_pattern)  # glob pattern pointing to files
        self.seed = seed  # random seed for shuffling

        # shuffle or sort files
        if shuffle_input_filenames:
            random.seed(self.seed)
            random.shuffle(self.files)
        else:
            self.files = natural_sort(self.files)
        self.filetype = filetype  # filetype ["tfrecords"]
        implemented_filetypes = ["tfrecords"]
        if self.filetype not in implemented_filetypes:
            raise NotImplementedError

        self.processed_files = FixedSizeOrderedDict(max=1)  # storage for lazily loading data

        # parses the length of the files, either by encoding in the filenames or by iterating over them
        self._get_lens()

        self.seq_len = seq_len  # set sequence length
        self.mode = mode  # set mode ["chunks"]
        implemented_modes = ["chunks"]
        if self.mode not in implemented_modes:
            raise NotImplementedError

        self.pretokenized = pretokenized
        if not self.pretokenized:
            raise NotImplementedError  # TODO: tokenize text data on the fly

        self.train = train

    def _get_number_of_documents(self, filename):
        # extracts number of files from a filename formatted "<name>_<num_documents>.{filetype}."
        # if no pattern is matched, returns None
        match = re.search("_(\d{1,})." + self.filetype + "$", filename)
        return int(match.group(1)) if match is not None else match

    def _get_number_of_documents_by_iteration(self, filename):
        # extracts number of files from a tfrecord document in the event it doesn't have metadata in the filename
        # this could be very slow.
        logging.warning(
            "Found no metadata found in filename - iterating through first tfrecord to find global length")
        count = 0
        if self.filetype == "tfrecords":
            for _ in tf.io.tf_record_iterator(filename):
                count += 1
        return count

    def _get_lens(self):
        lens = []
        for f in self.files:
            n_documents = self._get_number_of_documents(f)
            if n_documents is None:
                n_documents = self._get_number_of_documents_by_iteration(f)
            lens.append(n_documents)
        self.lens = lens
        self._len = sum(self.lens)
    
    def _parse_function(self, example_proto):
        features = {
            "text": tf.io.VarLenFeature(tf.int64)
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        return tf.sparse.to_dense(parsed_features["text"], parsed_features["text"].dense_shape[0])

    def _process_tfrecord(self, tfrecords_file, resume_idx=None):
        dataset = tf.data.TFRecordDataset([tfrecords_file])
        dataset = dataset.map(self._parse_function, num_parallel_calls=1)
        for example in dataset.as_numpy_iterator():
            yield torch.tensor(example, dtype=torch.long)

    def _maybe_process_tfrecord(self, file_idx):
        if self.processed_files.get(file_idx) is None:
            self.processed_files[file_idx] = list(self._process_tfrecord(self.files[file_idx]))
        return self.processed_files[file_idx]

    def _seek(self, idx):
        cumsum = 0
        for count, (f, length) in cycle(enumerate(zip(self.files, self.lens))):
            prev_cumsum = cumsum
            cumsum += length
            if cumsum == idx:
                remainder = 0
                skip_idx = count + 1
                return skip_idx, remainder
            elif cumsum > idx:
                remainder = idx - prev_cumsum
                skip_idx = count
                return skip_idx, remainder

    def __getitem__(self, idx):
        # seek to correct chunk
        seek_idx, remainder = self._seek(idx)
        f = self.files[seek_idx]
        if self.filetype == "tfrecords":
            chunk = self._maybe_process_tfrecord(
                seek_idx)  # parses tfrecord file to a list *once* then stores in memory
        else:
            raise NotImplementedError
        return chunk[remainder]  # get item from current chunk

    def __len__(self):
        return self._len


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.seq_len
