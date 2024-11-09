import os
import torch

def init_model(model_name, local_rank, device):
    if local_rank == 0:
        print(f"[INFO] Loading model {model_name}...")
    if 'gpt2' in model_name:
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    model.half()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    return model

from torch.nn import CrossEntropyLoss

def get_loss(logit, label, ignore_index=50256):
    shift_logits = logit[..., :-1, :].contiguous()
    shift_labels = label[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    ).view(shift_logits.size(0), shift_logits.size(1))

    non_ignore_mask = (shift_labels != ignore_index)
    loss = (loss * non_ignore_mask).sum(dim=-1) / non_ignore_mask.sum(dim=-1)

    return loss

import itertools

def flatten_list(nested_list):
    return list(itertools.chain.from_iterable(nested_list))

def get_files(work_dir, ext=".json.gz.result"):
    for root, _, files in os.walk(work_dir):
        for file in files:
            if file.endswith(ext):
                yield os.path.join(root, file)