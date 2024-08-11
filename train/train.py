# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import jsonlines
import math
import pathlib
from typing import Dict, Optional, Sequence
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from tqdm import tqdm
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index



def set_seed(seed_num=42):
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)


set_seed()


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    data_random: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    loss_input: bool = field(
        default=False,
        metadata={
            "help": "sft, whether count loss on input"
        }
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    loss_input: bool
) -> Dict: 
    
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    
    for i, source in tqdm(enumerate(sources)):
        skip=False
        # skip 'system'
        if source[0]['from'] == 'system':
            source = source[1:]
            
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if role != conv.roles[j % 2]:
                print(f"skip {i}. role sequence error. {sentence['value']}")
                skip=True
                break
            # assert role == conv.roles[j % 2], f"{i} {sentence['value']}"
            conv.append_message(role, sentence["value"])
        if skip:
            continue
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    if not loss_input:
    # Mask targets. Only compute loss on the assistant outputs.
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID  #for bos
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2  # for bos & space after :

                # Ignore the user instructions
                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                cur_len += turn_len  # 自动产生的bos正好和忽略的eos抵消了

            target[cur_len:] = IGNORE_TOKEN_ID

            if False:  # Inspect and check the correctness of masking
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                rank0_print(tokenizer.decode(z))

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    rank0_print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, loss_input: bool):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, loss_input)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, loss_input: bool):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.loss_input = loss_input
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.loss_input)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret

def load_raw_data(data_path):
    if data_path.endswith(".json"):
            raw_data = json.load(open(data_path, "r"))
    elif data_path.endswith(".jsonl"):
        with jsonlines.open(data_path, mode="r") as reader:
            raw_data = [item for item in reader]
    return raw_data


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, loss_input: bool, train_ratio=0.98
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (     # LazySupervisedDataset是用的时候处理并缓存, SupervisedDataset是全部处理完后再用
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    if data_args.eval_data_path:
        train_raw_data = load_raw_data(data_args.data_path)
        eval_raw_data = load_raw_data(data_args.eval_data_path)
        if data_args.data_random:
            np.random.seed(42)
            train_raw_data = [train_raw_data[i] for i in np.random.permutation(len(train_raw_data))]
            eval_raw_data = [train_raw_data[i] for i in np.random.permutation(len(eval_raw_data))]
    else:
        raw_data = load_raw_data(data_args.data_path)
        # Split train/test
        np.random.seed(42)
        perm = np.random.permutation(len(raw_data))
        split = int(len(perm) * train_ratio)
        train_indices = perm[:split]
        if train_ratio < 1:
            eval_indices = perm[split:]
        else:
            # if train_ratio==1, we use 5% of data as eval data, make sure trainer will not throw error when eval data is empty
            eval_indices = perm[-int(len(perm) * 0.001) :]
        train_raw_data = [raw_data[i] for i in train_indices]
        eval_raw_data = [raw_data[i] for i in eval_indices]

    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")
    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer, loss_input=loss_input)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer, loss_input=loss_input)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args)
    print(data_args)
    print(training_args)
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        add_eos_token=False,    # eos也是false, 因为template中有</s>所以不用管了
        add_bos_token=False,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, train_ratio=1, data_args=data_args, loss_input=training_args.loss_input)

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        # use_flash_attention_2=True
        # Note 4.34.0新特性, 集成了flash attention 2
    )

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    model.config.use_cache = True
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
