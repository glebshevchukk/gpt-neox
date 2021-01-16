
import torch
import linecache
import jsonlines
import math
from multiprocessing import Process,Pool
import pathlib
from functools import partial
import logging
import os
from gpt_neox.data_utils import file_lines,strictly_increasing 
#this is here to suppress warnings about us passing in sequences that are longer than default
logging.getLogger("transformers").setLevel(logging.ERROR)

"""
Shards data files into a certain size,
then creates another "metadata" file that stores a single entry for every entry in the dataset
so we can easily index into data for training
"""
def shardify(data_paths:list,output_path:str, max_items_per_file:int=10000, output_dir:str="",tokenizer=None,num_workers=32):
    summary_dict = {'max_items_per_file':max_items_per_file,'file_names':[]}
    
    total_items = 0
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    #splitting workers per path intead of giving each worker a different path
    last_max = 0
    max_chunk = 0
    #this keeps track of how many items there are in each file
    items_per_file=[]
    #this keeps track of where each sentence starts in relation to the total item index
    inner_indexes = []
    #this keeps track of which file chunk the inner index refers to
    inner_chunk_starts = []
    for path in data_paths:
        total_lines = file_lines(path)
        num_lines = total_lines//num_workers

        pool = Pool(processes = num_workers)
        s_process = partial(shardify_process,path=path,num_lines=num_lines,output_dir=output_dir,\
            tokenizer=tokenizer,max_items_per_file=max_items_per_file)
        returns = pool.map(s_process, range(num_workers))
        for file_names,worker_items,worker_indexes,worker_chunk_starts,worker_total in returns:
            
            summary_dict['file_names'].extend(file_names)

            #add all current ones up
            for i in range(len(worker_items)-1):
                worker_items[i+1]+=worker_items[i]

            worker_items = [item + last_max for item in worker_items]
            #this last item is only used to know where the next sequence should start since this is the "end" of the current one
            last_max = worker_items[-1]
            items_per_file.extend(worker_items[:-1])
            inner_indexes.extend(worker_indexes)

            worker_chunk_starts = [chunk + max_chunk for chunk in worker_chunk_starts]
            
            #have to add 1 because we know implicitly that chunk 0 refers to next chunk
            max_chunk = max(worker_chunk_starts)+1
            inner_chunk_starts.extend(worker_chunk_starts)
            total_items += worker_total
    

    for i in range(len(inner_indexes)):
        inner_indexes[i]+=items_per_file[inner_chunk_starts[i]]

    summary_dict['indexes'] = items_per_file
    summary_dict['inner_indexes'] = inner_indexes
    summary_dict['total_items'] = total_items

    #finish by writing the summary dict
    with jsonlines.open(output_dir+"/"+output_path, mode='w') as writer:
        writer.write(summary_dict)
    
    return summary_dict

#Runs on a single worker to chunk and optionally tokenize a jsonl file
def shardify_process(worker_id,path,num_lines,output_dir,tokenizer,max_items_per_file):
    
    start_line = worker_id*num_lines
    end_line = (worker_id+1)*num_lines

    p = pathlib.Path(path) 
    dataset_name = p.stem
    extension = p.suffix

    file_names = []

    text_to_write = []
    single_file_chunk = 0
    items_per_file = [0]
    inner_indexes = []
    inner_chunk_starts = []
    total_written =0

    current_indices,current_chunk_starts=[],[]
    chunk_writer = None
    chunk_file = None
    with jsonlines.open(path) as reader:
    
        for line_idx,line_loaded in enumerate(reader):
            #only have the worker process the lines of the jsonl that it has been assigned to
            if line_idx<start_line or line_idx>end_line:
                continue
            
            text = line_loaded['text']
            if len(text) == 0:
                continue

            #This is here to make sure we know that sentences are different
            text ="<|endoftext|> "+ text

            if tokenizer:
                all_items = tokenizer(text,return_tensors='pt')['input_ids']
                all_items = all_items.numpy().tolist()[0]
                all_items = list(map(lambda x:int(x), all_items))
                
            else:
                all_items = text.split(" ")

            #this inner index corresponds to index where this specific sentence begins
            inner_index = len(text_to_write)
            current_indices.append(inner_index)
            #this inner chunk start tells us which chunk this sentence will be written to
            current_chunk_starts.append(single_file_chunk)
            text_to_write.extend(all_items)

            #once we've filled the buffer, write it down to the current file, then switch to a new file
            if len(text_to_write) >= max_items_per_file:
                if chunk_writer:
                    chunk_file.close()
                    chunk_writer.close()
                chunk_path=output_dir+"/"+dataset_name+"_"+str(worker_id)+"_"+str(single_file_chunk)+extension
                chunk_file = open(chunk_path,"w")
                chunk_writer = jsonlines.Writer(chunk_file)
                file_names.append(chunk_path)

                inner_indexes.extend(current_indices)
                inner_chunk_starts.extend(current_chunk_starts)
                current_indices,current_chunk_starts=[],[]
                chunk_writer.write(text_to_write)
                
                single_file_chunk += 1

                items_per_file.append(len(text_to_write))
                total_written += len(text_to_write)
              
                text_to_write = []
                

        #handle the case when we haven't written a full amount, but there are still items left over
        if len(text_to_write) > 0:
            if chunk_writer:
                chunk_file.close()
                chunk_writer.close()
            chunk_path=output_dir+"/"+dataset_name+"_"+str(worker_id)+"_"+str(single_file_chunk)+extension
            chunk_file = open(chunk_path,"w")
            chunk_writer = jsonlines.Writer(chunk_file)
            file_names.append(chunk_path)
            chunk_writer.write(text_to_write)

            inner_indexes.extend(current_indices)
            inner_chunk_starts.extend(current_chunk_starts)
        
            items_per_file.append(len(text_to_write))
            total_written +=len(text_to_write)
        if chunk_writer:
            chunk_file.close()
            chunk_writer.close()

    return file_names,items_per_file, inner_indexes,inner_chunk_starts,total_written