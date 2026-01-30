import argparse
import pprint
import torch
import random
import numpy as np
import os
from datetime import datetime
import logging
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

supported_models = ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf', 'facebook/opt-125m']
supported_datasets = ['wikitext2', 'ptb', 'c4']

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def llama_down_proj_groupsize(model, groupsize):
    assert groupsize > 1, 'groupsize should be greater than 1!'
    if model.config.intermediate_size % groupsize == 0:
        return groupsize
    group_num = int(model.config.hidden_size/groupsize)
    down_proj_groupsize = model.config.intermediate_size//group_num
    return down_proj_groupsize

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

def config_logging(log_file, level=logging.INFO):
    class LogFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                self._style._fmt = "%(message)s"
            else:
                self._style._fmt = "%(levelname)s: %(message)s"
            return super().format(record)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LogFormatter())
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(LogFormatter())
    logging.basicConfig(level=level, handlers=[console_handler, file_handler])

def parser_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_dataset', type=str, default='wikitext2', choices=supported_datasets)
    parser.add_argument('--hf_token', type=str, default=None)
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--rotate', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--rotate_mode', type=str, default='hadamard', choices=['hadamard', 'random'])
    parser.add_argument('--rotation_seed', type=int, default=-1)
    parser.add_argument('--fp32_had', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--a_bits', type=int, default=16)
    parser.add_argument('--a_groupsize', type=int, default=-1)
    parser.add_argument('--a_asym', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--a_clip_ratio', type=float, default=1.0)
    parser.add_argument('--w_bits', type=int, default=16)
    parser.add_argument('--w_groupsize', type=int, default=-1)
    parser.add_argument('--w_asym', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--w_rtn', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--w_clip', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--cal_dataset', type=str, default='wikitext2', choices=supported_datasets)
    parser.add_argument('--percdamp', type=float, default=.01)
    parser.add_argument('--act_order', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--int8_down_proj', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--v_bits', type=int, default=16)
    parser.add_argument('--v_groupsize', type=int, default=-1)
    parser.add_argument('--v_asym', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--v_clip_ratio', type=float, default=1.0)
    parser.add_argument('--k_bits', type=int, default=16)
    parser.add_argument('--k_groupsize', type=int, default=-1)
    parser.add_argument('--k_asym', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--k_pre_rope', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--k_clip_ratio', type=float, default=1.0)
    parser.add_argument('--load_qmodel_path', type=str, default=None)
    parser.add_argument('--save_qmodel_path', type=str, default=None)
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb_id', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument("--mc_num", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--gen_length", type=int, default=1024)
    parser.add_argument("--block_length", type=int, default=1024)
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--capture_layer_io', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--layer_idx', type=int, default=10)
    parser.add_argument("--lm_eval", action="store_true")
    parser.add_argument('--tasks', type=str, default="winogrande")
    parser.add_argument('--lm_eval_batch_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--distribute", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)

    args = parser.parse_args()

    if args.save_name is None:
        args.save_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    setattr(args, 'save_path',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', args.model, args.save_name))
    os.makedirs(args.save_path, exist_ok=True)

    config_logging(os.path.join(args.save_path, f'{args.save_name}.log'))

    if args.a_groupsize != -1 and args.w_groupsize != -1:
        assert args.a_groupsize == args.w_groupsize, 'a_groupsize should be the same as w_groupsize!'

    assert args.k_pre_rope == False, 'Pre-RoPE quantization is not supported yet!'

    if args.wandb:
        assert args.wandb_id is not None and args.wandb_project is not None, 'WandB ID/project is not provided!'

    logging.info('Arguments: ')
    logging.info(pprint.pformat(vars(args)))
    logging.info('--' * 30)
    return args

def cleanup_memory(verbos=True):
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def distribute_model(model):
    # Simplified distribution
    cleanup_memory()