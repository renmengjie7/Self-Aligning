
from utils import ensure_file, read_from_jsonl, write_to_jsonl, save_txt
import argparse
from tqdm import tqdm
import os
import ray
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


choices = [chr(i) for i in range(ord('A'), ord('Z'))]

domains={
    "medmcqa-exp": "medicine", "medmcqa": "medicine", "medmcqa-all": "medicine",
    "history": "history", "engineering": "engineering", "jurisprudence": "jurisprudence",
}


vicuna_template=(
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "USER: {} "
        "ASSISTANT: "
    )

def format_example(data, include_answer=True, lang='en'):
    prompt = data['question']
    for j in range(len(data['choices'])):
        prompt += "\n{}. {}".format(choices[j], data['choices'][j])
    if lang == 'en':
        prompt += "\nAnswer:"
    elif lang == 'zh':
        prompt += "\n答案："
    if include_answer:
        prompt += "{}. {}\n\n".format(choices[data['answer']], data['choices'][data['answer']])
    return prompt

def gen_prompt(devs, domain, k=-1, lang='en'):
    if lang == 'en':
        if domain == 'truthfulqa':
            prompt = f"The following are multiple choice questions about true or false statements. Please choose the correct answer. \n\n"
        else:
            prompt = f"The following are multiple choice questions about {domain}. Please choose the correct answer. \n\n"
    elif lang == 'zh':
        prompt = f"以下是中国关于{domain}的单项选择题，请选出其中的正确答案。\n\n"
    if k == -1:
        return prompt
    for i in range(k):
        # TODO 或许需要增加随机性?
        prompt += format_example(devs[i], lang=lang)
    return prompt

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


@ray.remote(num_gpus=1)
@torch.inference_mode()
def run_single_task(datas, args, domain, devs):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, 
                                              add_bos_token=True, model_max_length=4096,
                                              padding_side="right", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, 
                                                 device_map="auto", offload_folder="offload",
                                                 low_cpu_mem_usage=True, trust_remote_code=True)
    
    cors = []   # correct or not
    preds = []  # predct label
    all_probs = []  # all labels's probabilities
    golden_probs = []   # golden label's probability
    pred_probs = []    # predict label's probability
    lang = args.lang_prompt
    for i in tqdm(range(len(datas))):
        # get prompt and make sure it fits
        k = args.shot
        prompt_end = format_example(datas[i], include_answer=False, lang=lang)
        if args.hint == 1:
            train_prompt = gen_prompt(devs, domain, k, lang)
            prompt = train_prompt + prompt_end
        else:
            prompt = prompt_end
        if args.vicuna == 1:
            prompt = vicuna_template.format(prompt)
        
        label = datas[i]["answer"]  # index

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
    
    return cors, preds, all_probs, golden_probs, pred_probs


def eval_by_logits(model, tokenizer, text, data):
    print(text)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    logits = model(
        input_ids=input_ids,
    ).logits[:, -1].flatten()
    pred, probs = parse_logits(logits, tokenizer, num=len(data['choices']))
    print(pred)
    return pred, probs


def generate(model, tokenizer, text, temperature=0, top_p=0.8, max_new_tokens=64):
    # have not tested
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


@torch.no_grad()
def eval(args, domain, devs, datas):
    
    chunk_size = len(datas) // args.num_gpus
    ans_handles = []
    
    for i in range(0, len(datas), chunk_size):
        ans_handles.append(
            run_single_task.remote(
                datas[i : i + chunk_size], args, domain, devs
            )
        )

    cors_all = []
    preds_all = []
    all_probs_all = []
    all_golden_probs = []
    all_pred_probs = []
    
    for ans_handle in tqdm(ans_handles):
        cors, preds, all_probs, golden_probs, pred_probs = ray.get(ans_handle)
        cors_all.extend(cors)
        preds_all.extend(preds)
        all_probs_all.extend(all_probs)
        all_golden_probs.extend(golden_probs)
        all_pred_probs.extend(pred_probs)
    
    acc = np.mean(cors_all).item()
    print("Average accuracy {:.5f} - {}".format(acc, domain))
    return cors_all, preds_all, acc, all_probs_all, all_golden_probs, all_pred_probs
   

def save_results(args, datas, cors, preds, acc, all_probs, golden_probs, pred_probs):
    for i in range(len(datas)):
        datas[i]['correct'] = cors[i]
        datas[i]['pred'] = preds[i]
        datas[i]['probs'] = all_probs[i]
        datas[i]['golden_prob'] = golden_probs[i]
        datas[i]['pred_prob'] = pred_probs[i]
    write_to_jsonl(datas, f'{args.output_file}.jsonl')
    save_txt("{:.5f}".format(acc), f'{args.output_file}.txt')
    
    
def main(args):
    ensure_file(args.output_file)
    path=f'{args.data_dir}/{args.task}'
    devs = read_from_jsonl(f'{path}/dev.jsonl') if os.path.exists(f'{path}/dev.jsonl') else []
    import random
    random.seed(42)
    indexes = list(range(len(devs)))
    random.shuffle(indexes)
    devs = [devs[index] for index in indexes]
    if not os.path.exists(f'{path}/{args.split}.jsonl'):
        return
    datas = read_from_jsonl(f'{path}/{args.split}.jsonl')
    cors, preds, acc, all_probs, golden_probs, pred_probs = eval(args, args.domain, devs, datas)
    save_results(args, datas, cors, preds, acc, all_probs, golden_probs, pred_probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model-name", type=str, default='llama-2-7b')
    parser.add_argument("--model-path", type=str, default='models/mydownload/llama2/llama-2-7b-hf')

    parser.add_argument("--vicuna", type=int, default=0)
    parser.add_argument("--hint", type=int, default=1)
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--split", type=str, default="train", choices=['train', 'test']) # infer train or test

    parser.add_argument("--lang-prompt", type=str, default='en', choices=['en', 'zh'])
    parser.add_argument("--data-dir", type=str, default="data/source")
    parser.add_argument("--task", type=str, default="cmb")     # the path "data-dir/task" need exist
    
    parser.add_argument("--output-dir", type=str, default="results/20231123domain-base")
    
    parser.add_argument("--num-gpus", type=int, default=1)
    # only support logits now
    parser.add_argument("--eval-type", type=str, default="logits", choices=['logits', 'generation'])

    args = parser.parse_args()
    args.output_file = "{}/{}/{}-{}-shot{}-{}-vicuna{}".format(
        args.output_dir,
        args.model_name, 
        args.task,
        args.eval_type,
        args.shot,
        args.split,
        args.vicuna
    )
    if os.path.exists(f'{args.output_file}.txt') and os.path.exists(f'{args.output_file}.jsonl'):
        print('over')
    else:
        args.domain=domains[args.task]
        print(args)
        main(args)
