import argparse
import os
import torch
import numpy as np
import pandas as pd
from categories import cmmlu_categories, cmmlu_subcategories, cmmlu_name_en2zh, ceval_categories, ceval_subcategories, ceval_name_en2zh, mmlu_categories, mmlu_subcategories, mmlu_name_en2zh
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import read_from_jsonl, write_to_jsonl
import time
from tqdm import tqdm
import ray


categories, subcategories, name_en2zh = None, None, None
choices = [chr(i) for i in range(ord('A'), ord('K'))]

vicuna_template=(
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "USER: {} "
        "ASSISTANT: "
    )


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(data, include_answer=True, lang='en'):
    prompt = data['question']
    for j in range(len(data['choices'])):
        prompt += "\n{}. {}".format(choices[j], data['choices'][j])
    if lang == 'en':
        prompt += "\nAnswer:"
    elif lang == 'zh':
        prompt += "\n答案："
    if include_answer:
        prompt += "{}. {}\n\n".format( choices[data['answer']], data['choices'][data['answer']])
    return prompt

def gen_prompt(devs, subject, k=-1, lang='en'):
    if lang == 'en':
        subject = format_subject(subject)
        prompt = f"The following are multiple choice questions about {subject}. Please choose the correct answer. \n\n"
    elif lang == 'zh':
        subject = name_en2zh[subject]
        prompt = f"以下是中国关于{subject}的单项选择题，请选出其中的正确答案。\n\n"
    if k == -1:
        return prompt
    for i in range(k):
        prompt += format_example(devs[i], lang=lang)
    return prompt

@ray.remote(num_gpus=1)
@torch.inference_mode()
def eval(args, subject, model_path, data_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, 
                                              add_bos_token=False, model_max_length=4096,
                                              padding_side="right", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, 
                                                 device_map="auto", offload_folder="offload",
                                                 trust_remote_code=True, low_cpu_mem_usage=True)
    
    devs = read_from_jsonl(os.path.join(data_dir, "dev", subject + ".jsonl"))
    import random
    random.seed(42)
    indexes = list(range(len(devs)))
    random.shuffle(indexes)
    devs = [devs[index] for index in indexes]
    datas = read_from_jsonl(os.path.join(data_dir, args.split, subject + f".jsonl"))
    
    cors = []
    preds = []
    all_probs = []
    golden_probs = []
    pred_probs = []

    for i in range(len(datas)):
        k = args.shot
        prompt_end = format_example(datas[i], include_answer=False, lang=args.lang_prompt)

        if args.hint == 1:
            train_prompt = gen_prompt(devs, subject, k, lang=args.lang_prompt)
            prompt = train_prompt + prompt_end
        else:
            prompt = prompt_end

        if args.vicuna == 1:
            prompt = vicuna_template.format(prompt)
        
        label = datas[i]["answer"]
        
        if args.eval_type == 'logits':
            pred, probs = eval_by_logits(model, tokenizer, prompt, datas[i])
        else:
            pred, probs = generate(model, tokenizer, prompt)
        
        cor = pred == label
        
        if args.eval_type == 'logits':
            preds.append(pred.item())
            cors.append(cor.item())
            all_probs.append(probs.tolist())
            golden_probs.append(probs.tolist()[label])
            pred_probs.append(probs.tolist()[pred])
        else:
            preds.append(pred)
            cors.append(cor)
            all_probs.append(probs)
            golden_probs.append(-1)
            pred_probs.append(-1)
    
    acc = np.mean(cors)

    print("Average accuracy {:.4f} - {}".format(acc, subject))
    return subject, datas, cors, preds, all_probs, golden_probs, pred_probs, acc


def eval_by_logits(model, tokenizer, text, data):
    print(text)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    logits = model(
        input_ids=input_ids,
    ).logits[:, -1].flatten()
    pred, probs = parse_logits(logits, tokenizer, num=len(data['choices']))
    print(pred)
    return pred, probs


def parse_logits(logits, tokenizer, num=4):
    """
    num: num of choices
    return
    pred: index
    logits: probabilities
    """
    probs = (
            torch.nn.functional.softmax(
                torch.tensor([
                    logits[tokenizer(item).input_ids[-1]] for item in [chr(i) for i in range(ord('A'), ord('A')+num)]
                    ]),
                dim=0,
            )
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )
    pred = np.argmax(probs)
    return pred, probs


def generate(model, tokenizer, text, temperature=0, top_p=0.8, max_new_tokens=64):
    print(text)
    inputs = tokenizer(text, return_tensors="pt")
    for k in inputs:
        inputs[k] = inputs[k].cuda()
    outputs = model.generate(**inputs, do_sample=False, temperature=temperature, top_p=top_p, max_length=max_new_tokens + inputs['input_ids'].size(-1))
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(response)
    if len(response) > 0 and response[0] in choices:
        return ord(response[0])-ord('A'), response
    else: 
        return -1, response


def get_subjects(data_dir, split):
    subjects = sorted(
        [
            f.split(".jsonl")[0]
            for f in os.listdir(os.path.join(data_dir, split))
            if ".jsonl" in f
        ]
    )
    return subjects


def save_subjects(datas, cors, preds, acc, all_probs, golden_probs, pred_probs, save_dir, subject):
    for i in range(len(datas)):
        datas[i]['correct'] = cors[i]
        datas[i]['pred'] = preds[i]
        datas[i]['probs'] = all_probs[i]
        datas[i]['golden_prob'] = golden_probs[i]
        datas[i]['pred_prob'] = pred_probs[i]
    write_to_jsonl(datas, f'{save_dir}/{subject}.jsonl')
    
    
def save_summary(save_dir, subcat_cors, cat_cors, all_cors):
    content = ''
    # save summary
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.4f} - {}".format(subcat_acc, subcat))
        content += str(subcat_acc)+'-'+subcat+'\n'
    
    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("-----------------sub----------------")
        print("Average accuracy {:.4f} - {}".format(cat_acc, cat))
        content += str(cat_acc)+'-'+cat+'\n'
    
    weighted_acc = np.mean(np.concatenate(all_cors))
    print('------------all-------------')
    print("Average accuracy: {:.4f}".format(weighted_acc))
    
    save_filepath = os.path.join(save_dir, "result.txt")
    content += str(weighted_acc) +'\n'
    
    with open(save_filepath,'w') as file:
        file.write(content)


def main(args):
    subjects = get_subjects(args.data_dir, args.split)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f'{args.output_dir}/split', exist_ok=True)

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    ans_handles = []
    for subject in tqdm(subjects):
        result_file = f'{args.output_dir}/split/{subject}.jsonl'
        if os.path.exists(result_file):
            datas = read_from_jsonl(result_file)
            cors = [ item['correct'] for item in datas ]
            subcats = subcategories[subject]    # 当前学科对应的二级学科
            for subcat in subcats:
                subcat_cors[subcat].append(cors)
                for key in categories.keys():
                    if subcat in categories[key]:   # 对应的一级学科
                        cat_cors[key].append(cors)
            all_cors.append(cors)
        else:
            ans_handles.append(
                eval.remote(
                    args, subject, args.model_path, args.data_dir
                )
            )
    
    for ans_handle in ans_handles:
        subject, datas, cors, preds, all_probs, golden_probs, pred_probs, acc = ray.get(ans_handle)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)
        save_subjects(datas, cors, preds, acc, all_probs, golden_probs, pred_probs, f'{args.output_dir}/split', subject)
        
    save_summary(args.output_dir, subcat_cors, cat_cors, all_cors)


def set_static_variables(task):
    global categories, subcategories, name_en2zh
    if 'ceval' == task:
        categories = ceval_categories
        subcategories = ceval_subcategories
        name_en2zh = ceval_name_en2zh
    elif 'cmmlu' == task:
        categories = cmmlu_categories
        subcategories = cmmlu_subcategories
        name_en2zh = cmmlu_name_en2zh
    elif 'mmlu' == task:
        categories = mmlu_categories
        subcategories = mmlu_subcategories
        name_en2zh = mmlu_name_en2zh
    else:
        raise ValueError("Unknown dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model-name", type=str, default='llama-2-7b')
    parser.add_argument("--model-path", type=str, default='models/mydownload/llama2/llama-2-7b-hf')
    
    parser.add_argument("--hint", type=int, default=1)
    parser.add_argument("--vicuna", type=int, default=0)
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--split", type=str, default="train", choices=['train', 'test']) # infer train or test
    parser.add_argument("--lang-prompt", type=str, default='en', choices=['en', 'zh'])
    
    parser.add_argument("--task", type=str, default='ceval', choices=['ceval', 'cmmlu', "mmlu"])
    parser.add_argument("--data-dir", type=str, default="datas/download/mmlu")
    parser.add_argument("--output-dir", type=str, default="results/20231123domain-base")
    
    parser.add_argument("--eval-type", type=str, default="logits", choices=[
        "generation", "logits"
    ])
    parser.add_argument("--num-gpus", type=int, default=1)

    args = parser.parse_args()
    set_static_variables(args.task)
    args.output_dir="{}/{}/{}-shot{}-{}-vicuna{}".format(
        args.output_dir,
        args.model_name, 
        args.eval_type,
        args.shot,
        args.split,
        args.vicuna
    )
    print(args)
    main(args)
