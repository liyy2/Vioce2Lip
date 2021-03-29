import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from tqdm import tqdm, trange

def attempt_load_day(video_by_day, day_idx):
    d = video_by_day[day_idx]
    v_sources = d['sources']
    v_target = d['target']
    
    a_target = v_target.replace('mp4', 'wav').replace('video', 'audio')
    v = np.load(v_target)
    a = np.load(a_target)
    
    v_paths = list(map(lambda x: x [0], v_sources))
    num_slices = list(map(lambda x: x [1], v_sources))
    total_slices = sum(num_slices)
    
    AUDIO_FEATS_LEN = 10
    # (780, 20, 2), (20, 7800), 780
    if v.shape[0] == total_slices - 1:
        v = np.concatenate((v, np.expand_dims(v[-1, :, :], axis=0)), axis=0)
    if v.shape[0] != total_slices or a.shape[1] != total_slices * AUDIO_FEATS_LEN:
        print('WARN: wrong # of slices.')
        print(v.shape, a.shape, total_slices)
        return
    
    elapsed_slices = 0
    results = []
    for idx, current_len in enumerate(num_slices):
        v_p = v_paths[idx]
        vf_from, vf_to = elapsed_slices, elapsed_slices+current_len
        af_from, af_to = elapsed_slices*10, (elapsed_slices+current_len)*10
        results.append({
            'source': v_p, 
            'len': current_len,
            'v': v[vf_from:vf_to,:,:],
            'a': a[:,af_from:af_to]
        })
        elapsed_slices += current_len
    
    return results

def load_day_to_batch(video_by_day, day_idx, device):
    day_slices = attempt_load_day(video_by_day, day_idx)

    src_len = list(map(lambda sliced: sliced['len'], day_slices))
    inputs = torch.zeros(max(src_len), len(day_slices), 20, 10)
    label = torch.zeros(max(src_len), len(day_slices), 20, 2)
    total_slices = sum(src_len)

    for idx, sliced in enumerate(day_slices):
        f_len, v, a = sliced['len'], sliced['v'], sliced['a']
        v = torch.from_numpy(v).float()
        a = torch.from_numpy(a).float().view(20, 10, -1)

        inputs[0:f_len, idx] = a.permute(2, 0, 1) / 100.0
        label[0:f_len, idx] = v
        
    return total_slices, src_len, inputs.to(device), label