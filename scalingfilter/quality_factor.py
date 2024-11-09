import argparse
import glob
import gzip
import os
import traceback
from pathlib import Path

import jsonlines
import torch
import torch.distributed as dist
from more_itertools import divide
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer
from utils import get_loss, init_model, flatten_list, get_files


class TextDataset(Dataset):
    def __init__(self, file, tokenizer, input_dir):
        self.file = file
        self.input_dir = input_dir
        if file.endswith("jsonl"):
            with open(file) as f:
                self.texts = [js["text"] for js in jsonlines.Reader(f)]
        else:
            with gzip.open(file) as f:
                self.texts = [js["raw_content"] for js in jsonlines.Reader(f)]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.tokenizer(
            self.texts[idx],
            padding="max_length",
            max_length=args.max_len,
            truncation=True,
            return_tensors="pt",
        )


def get_args_parser():
    parser = argparse.ArgumentParser(
        "ScalingFilter: assign quality factors for text samples", add_help=False
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        default=0,
        type=int,
        help="local rank for distributed training",
    )
    parser.add_argument("--max_len", default=1024, type=int)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--save_interval", default=1000, type=int)

    parser.add_argument(
        "--input_dir", default=None, type=str, help="the dir with JSONL files in it."
    )
    parser.add_argument("--force_overwrite", action="store_true", default=False)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--file_ext", default="jsonl", type=str)

    parser.add_argument("--nparts", default=1, type=int)
    parser.add_argument(
        "--rank_part_start",
        default=0,
        type=int,
        help="the idx of the first part of current node",
    )
    parser.add_argument(
        "--rank_nparts",
        default=1,
        type=int,
        help="how many parts does current node have",
    )

    parser.add_argument("--small_model", default="gpt2", type=str)
    parser.add_argument("--large_model", default="gpt2-large", type=str)

    parser.add_argument("--debug", action="store_true")
    return parser


def calculate_quality_factor(args, inputs, outputs_small, outputs_large, padding):
    """
    Calculate which samples to keep or drop based on the model outputs.
    Select appropriate statistical data based on cls_design value.
    """
    large_loss = get_loss(outputs_large.logits, inputs["input_ids"], padding)
    small_loss = get_loss(outputs_small.logits, inputs["input_ids"], padding)

    large_ppl = torch.exp(large_loss)
    small_ppl = torch.exp(small_loss)

    quality_factor = small_ppl / large_ppl

    return {
        "large_loss": large_loss,
        "small_loss": small_loss,
        "large_ppl": large_ppl,
        "small_ppl": small_ppl,
        "quality_factor": quality_factor,
    }


def process_quality_factor(
    args, model_small, model_large, dataloader, rank, world, device
):
    output = []
    out_prefix = os.path.join(
        args.output_dir,
        os.path.relpath(dataloader.dataset.file, dataloader.dataset.input_dir).replace(
            f".{args.file_ext}", ""
        ),
    )

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    padding = int(
        getattr(
            dataloader.dataset.tokenizer,
            "pad_token_id",
            dataloader.dataset.tokenizer.eos_token_id,
        )
    )

    for step, batch in enumerate(tqdm(dataloader)):
        texts, inputs = batch
        for k in inputs.keys():
            inputs[k] = inputs[k].squeeze().to(device)
        with torch.no_grad():
            outputs_large = model_large(**inputs, labels=inputs["input_ids"])
            outputs_small = model_small(**inputs, labels=inputs["input_ids"])

            results = calculate_quality_factor(
                args, inputs, outputs_small, outputs_large, padding
            )

        for idx, text in enumerate(texts):
            data = {
                "text": text,
                "large_loss": results["large_loss"][idx].item(),
                "small_loss": results["small_loss"][idx].item(),
                "large_ppl": results["large_ppl"][idx].item(),
                "small_ppl": results["small_ppl"][idx].item(),
                "quality_factor": results["quality_factor"][idx].item(),
            }

            output.append(data)

        if (step + 1) % args.save_interval == 0:
            with jsonlines.open(
                out_prefix + f".{rank+1}_of_{world}.jsonl", "w"
            ) as f:
                for v in output:
                    f.write(v)
            if rank == 0:
                print(
                    f"[DEBUG] Rank #0 saved {len(output)} samples to {out_prefix + f'.{rank+1}_of_{world}.jsonl'}"
                )
        torch.distributed.barrier()

    with jsonlines.open(out_prefix + f".{rank+1}_of_{world}.jsonl", "w") as f:
        for v in output:
            f.write(v)
        if rank == 0:
            print(
                f"[DEBUG] Rank #0 saved {len(output)} samples to {out_prefix + f'.{rank+1}_of_{world}.jsonl'}"
            )
    torch.distributed.barrier()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ["LOCAL_RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # model
    small_model = init_model(args.small_model, local_rank, device)
    large_model = init_model(args.large_model, local_rank, device)

    # tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.small_model)
    tokenizer.pad_token = tokenizer.eos_token
    try:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    except:
        print(f"[WARN] Failed to set pad_token_id to eos_token_id, never mind.")

    input_dir = Path(args.input_dir)
    file_list = sorted(list(get_files(input_dir, ext=f".{args.file_ext}")))
    if local_rank == 0:
        print(f"[DEBUG] My Work Dir: {input_dir}")
        print(
            f"[DEBUG] Detected {len(file_list)} files, first 5 files: {file_list[:5]}"
        )
    file_list = list(
        divide(args.nparts, file_list)[
            args.rank_part_start : args.rank_part_start + args.rank_nparts
        ]
    )
    file_list = flatten_list(file_list)
    if local_rank == 0:
        print(
            f"[DEBUG] My Part before dedup: {len(file_list)}, first 5 files: {file_list[:5]}"
        )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # not overwrite
    real_file_list = []
    for file in file_list:
        out_prefix = os.path.join(
            input_dir / ".." / args.output_dir,
            os.path.relpath(file, input_dir).replace(f".{args.file_ext}", ""),
        )
        one_of_output_filename = out_prefix + f".1_of_{ws}.jsonl"
        if local_rank == 0:
            print(f"Checking {one_of_output_filename}...")
        existing_files = glob.glob(out_prefix + "*.jsonl")
        if len(existing_files) > 0:
            if local_rank == 0:
                print(f"Skipping {file} because {existing_files[0]} exists..")
        else:
            real_file_list.append(file)

    if local_rank == 0:
        print(
            f"[DEBUG] My Part: {len(real_file_list)}, first 5 files: {real_file_list[:5]}"
        )

    for file in real_file_list:
        try:
            dataset = TextDataset(file, tokenizer, input_dir)
            if local_rank == 0:
                print(f"[DEBUG] {file} has {len(dataset)} samples.")
            data_loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                sampler=torch.utils.data.distributed.DistributedSampler(dataset),
            )
            process_quality_factor(
                args,
                small_model,
                large_model,
                data_loader,
                local_rank,
                ws,
                device,
            )
        except Exception as e:
            print(f"[ERR] Error {e} occured when processing {file}..")
            traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "ScalingFilter: assign quality factors for text samples",
        parents=[get_args_parser()],
    )
    args = parser.parse_args()

    main(args)
