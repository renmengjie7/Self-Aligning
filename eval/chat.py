from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils import write_to_jsonl, read_from_jsonl
from tqdm import tqdm
import os
import fire

choices = ['A', 'B', 'C', 'D', 'E']


def generate(model, tokenizer, text, temperature=0.7, top_p=0.8, max_new_tokens=256):
    print('>>>')
    print(text)
    inputs = tokenizer(text, return_tensors="pt")
    for k in inputs:
        inputs[k] = inputs[k].cuda()
    outputs = model.generate(**inputs, do_sample=True, temperature=temperature, top_p=top_p, max_length=max_new_tokens + inputs['input_ids'].size(-1))
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print('>>>')
    print(response)
    return response


def build_input2(dev, train):
    prompt="""# Instruction

Below is a list of conversations between a human and an AI assistant (you). 
Users place their queries under "# Query:", and your responses are under  "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsibly, and thus you can reject to answer some controversial topics.

# Query:
```Can you tell me some common types of renewable energy sources?```
 
# Answer:
```
Absolutely, below are some of the most common types of renewable energy sources:

1. Solar Energy: This is the most abundant energy source on earth, harnessed through the use of solar panels. These panels convert sunlight into electricity without any moving parts, noise, pollution, or damage to the environment.
2. Wind Energy: Wind turbines convert the kinetic energy in the wind into mechanical power. This mechanical power can be used for specific tasks (such as pumping water) or converted into electricity to power homes, businesses, and schools.
3. Hydropower: Generated by using electricity generators to capture the energy from falling or fast-moving water. This renewable source can come from various ways, including a large dam on a river, tidal or wave energy from the ocean, or using small scale turbines in streams.
4. Geothermal Energy: This type of energy is generated from the heat deep within the Earth. This heat can be used directly for heating buildings or to generate electricity. It is continuously produced inside the Earth and is nearly as reliable as the tides.
5. Biomass Energy: Biomass is organic material that comes from plants and animals, and it contains stored energy from the sun. This energy can be burned directly or converted into biofuel which can burn more efficiently. 

Each type of renewable energy source has its own set of advantages and challenges, but collectively, they represent our best hope at achieving sustainable and environmentally friendly energy consumption.
```

# Query:
```Below is a multiple-choice question and the answer. Please give the explanation.
Question: {train_question}
Choices: {train_choice}
Answer: {train_answer}
```

# Answer:
"""
    dev_choice = ''
    for j in range(len(dev['choices'])):
        dev_choice += "\n{}. {}".format(choices[j], dev['choices'][j])
    train_choice = ''
    for j in range(len(train['choices'])):
        train_choice += "\n{}. {}".format(choices[j], train['choices'][j])
    prompt = prompt.format(dev_question=dev['question'], dev_choice=dev_choice, dev_answer=choices[dev['answer']], dev_exp=dev['explanation'], train_question=train['question'], train_choice=train_choice, train_answer=choices[train['pred']])
    return prompt


def parse_exp(text):
    """
    使用URIAL的, 抽取出结果
    """
    import re
    # 定义正则表达式
    pattern = r'```(.*?)```'

    # 使用re.findall来查找所有匹配的内容，同时使用re.DOTALL标志
    matches = re.findall(pattern, text, re.DOTALL)

    # 打印匹配的内容
    if len(matches) >= 1:
        return matches[0].strip()
    else:
        # 定义正则表达式，匹配字符串开头的```
        pattern = r'^```'

        # 使用re.sub来替换字符串开头的```为空字符串
        # 如果存在的话，它将被替换，否则字符串保持不变
        result = re.sub(pattern, '', text)
        return result.strip()

    
def test(model, tokenizer, domain, model_name):
    train_root = f'xxx'
    dev_root = f'xxx'
    devs = read_from_jsonl(f'{dev_root}/dev.jsonl')
    trains = read_from_jsonl(f'{train_root}/consistent-predict-exp-sample.jsonl')
    trains = read_from_jsonl(f'{train_root}/inconsistent-predict.jsonl')

    print(len(trains))
    
    maps = {'medmcqa': 0, 'engineering': 8, 'history': 2, 'jurisprudence': 5}
    dev = devs[maps[domain]]    # ICL's demo
    
    result_path=f'{train_root}/consistent-predict-exp-sample-model-exp.jsonl'
    result_path=f'{train_root}/inconsistent-predict-model-exp.jsonl'
    print(result_path)
    results = read_from_jsonl(result_path) if os.path.exists(result_path) else []
    for train in tqdm(trains):
        if train['id'] not in [item['id'] for item in results]:
            if train["correct"] is False:
                prompt = build_input2(dev, train)
                response = generate(model, tokenizer, prompt)
                exp = parse_exp(response)
                results.append({**train, 'model_exp_gen': response, 'model_exp': exp})    # 其实还需要个处理
            else:
                results.append(train)
            if len(results)%10==0:
                write_to_jsonl(results, result_path)
                print(f'save! --- {len(results)} to {result_path}')
    write_to_jsonl(results, result_path)


def run(model_name, domain):
    model_root = 'models/mydownload'
    model_paths={
        "llama-2-7b": 'llama2/llama-2-7b-hf',
        "llama-2-13b": 'llama2/llama-2-13b-hf',
        "llama-2-70b": 'llama2/llama-2-70b-hf',
        "mistral-7b": 'Mistral-7B-v0d1'
    }
    model_path=model_paths[model_name]
    model_path=f'{model_root}/{model_path}'

    model=None
    tokenizer=None
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True, offload_folder='offload')
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, add_bos_token=False, model_max_length=4096, padding_side="right", trust_remote_code=True)

    if model_name == 'mistral-7b' or model_name == 'llama-2-13b':
        # Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
        print(tokenizer.pad_token_id)
        tokenizer.pad_token_id=tokenizer.eos_token_id = 2
        print(tokenizer.pad_token_id)
        model.config.pad_token_id = model.config.eos_token_id

    test(model, tokenizer, domain, model_name)


if __name__ == '__main__':
    fire.Fire(run)
