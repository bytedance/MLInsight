import argparse
import time
from functools import partial
import os
import logging
import json
from typing import List

import datasets
import deepspeed
import loguru
import torch
import torch.distributed as dist
import transformers
from transformers import CONFIG_MAPPING, MODEL_MAPPING, AutoConfig, OPTForCausalLM
from transformers.utils.versions import require_version
from transformers.deepspeed import HfDeepSpeedConfig
from accelerate import Accelerator, DistributedType, DeepSpeedPlugin


logger = loguru.logger

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
DS_CONFIG_DIR = 'ds_configs'
CONFIG_DIR = 'configs'

def log_dist(message: str,
             ranks: List[int] = [],
             level: int = logging.INFO) -> None:
    """Log messages for specified ranks only"""
    my_rank = int(os.environ.get("RANK", "0"))
    if my_rank in ranks:
        if level == logging.INFO:
            logger.info(f'[Rank {my_rank}] {message}')
        if level == logging.ERROR:
            logger.error(f'[Rank {my_rank}] {message}')
        if level == logging.DEBUG:
            logger.debug(f'[Rank {my_rank}] {message}')

def get_data(batch_size, seq_len, vocab_size):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=torch.cuda.current_device())
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


def get_tflops(model_numel, batch_size, seq_len, step_time):
    print(f"model_numel: {model_numel}, batch_size: {batch_size}, step_time: {step_time}")
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)


def get_checkpoint_path(model_name):
    ckpt_prefix = '/mnt/bn/lq-volume-3t-training/ckpt'
    ckpt_name = model_name + "_" + '1gpu_ds.ckpt'
    return os.path.join(ckpt_prefix, ckpt_name)

def parse_args():
    parser = argparse.ArgumentParser(prog = 'finetune_ds_opt')
    parser.add_argument(
        "--pretrain",
        action='store_true',
        default=False,
        help="Pretrain model otherwise finetune model.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default='125m',
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--ds_config_name",
        type=str,
        default='ds_config',
        help="DS config name or path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per dp group) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2,
        help="Total number of training steps to perform.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Which GPU to run on (-1 for CPU). Defaults to -1."
    )
    args = parser.parse_args()
    return args


def main():
    print(f"training program's pid {os.getpid()}")
    args = parse_args()
    deepspeed.init_distributed()
    config_path = os.path.join(CONFIG_DIR, args.config_name + '.json')
    config = AutoConfig.from_pretrained(config_path)
    log_dist(f"Model config loaded from local json {config_path}", ranks=[0])
    ds_config_file = os.path.join(DS_CONFIG_DIR, args.ds_config_name + '.json')
    with open(ds_config_file, 'r') as f:
        ds_config = json.load(f)
    ds_config['train_batch_size'] =  args.batch_size * int(os.environ['WORLD_SIZE'])
    ds_config['train_micro_batch_size_per_gpu'] = args.batch_size
    log_dist(f"DS config loaded from local json {ds_config_file}", ranks=[0])

    # build model
    start = time.time()
    log_dist("Finetune a pre-trained model", ranks=[0])

    if args.pretrain:
        logger.info("Pretrain a new model from scratch", ranks=[0])
        # with deepspeed.zero.Init(remote_device="cpu", dtype=torch.half, enabled=True):
        #model = OPTForCausalLM(config).half()
        ds_config_hf = HfDeepSpeedConfig(ds_config)
        model = OPTForCausalLM.from_pretrained(args.model_name_or_path,
                                                config=config,
                                                local_files_only=False)
        numel = sum([p.numel() for p in model.parameters()])
        logger.info(f"Load new model from {config_path} in {time.time() - start}s", ranks=[0])
        #ckpt_path = get_checkpoint_path(args.config_name)
        #torch.save(model.state_dict(), ckpt_path)
        #logger.info(f"HF checkpoint saved to {ckpt_path} in: {time.time() - start}s")
    else:
        # Finetune
        # Don't remove this otherwise zero 3 won't work
        ds_config_hf = HfDeepSpeedConfig(ds_config)
        model = OPTForCausalLM.from_pretrained(args.model_name_or_path,
                                                config=config,
                                                local_files_only=False)
        log_dist(f"Load pretrained model {args.model_name_or_path} in: {time.time() - start}s", ranks=[0])
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257
    numel = sum([p.numel() for p in model.parameters()])
    get_tflops_func = partial(get_tflops, numel, args.batch_size, SEQ_LEN)
    model.gradient_checkpointing_enable()
    model, optimizer, _, _  = deepspeed.initialize(model=model,
                                                   config=ds_config)
    is_main_process = dist.get_rank() == 0

    if is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        torch.mannul_seed(args.seed)
        logger.info(f"Rank {dist.get_rank()}: random seed is set to {args.seed}")




    log_dist("Start to train", ranks=[0])
    model.train()
    tst_time = time.time()
    for step in range(args.max_train_steps):
        st_time = time.time()
        input_ids, attn_mask = get_data(args.batch_size, SEQ_LEN, VOCAB_SIZE)
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids, use_cache=False)
        loss = outputs['loss']
        optimizer.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        step_time = time.time() - st_time
        step_tflops = get_tflops_func(step_time)
        log_dist("step {} finished, time {}, Tflops {}".format(step, step_time, step_tflops), ranks=[0])
    tl_time = time.time() - tst_time
    GPU_hours = tl_time / 3600 * int(os.environ['WORLD_SIZE'])
    logger.info(f"Total time is {tl_time} with step {args.max_train_steps}, GPU hours = {GPU_hours}", ranks=[0])
    dist.barrier()
    logger.info("Training finished", ranks=[0])


if __name__ == "__main__":
    main()

