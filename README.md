# Learning or Self-aligning? Rethinking IFT

This is the office repository for ACL 2024 paper ["Learning or Self-aligning? Rethinking Instruction Fine-tuning"](https://arxiv.org/)

## 📣 Abstract
---
Instruction Fine-tuning (IFT) is a crucial phase in building large language models (LLMs). Previous works mainly focus on the IFT's role in the transfer of behavioral norms and the learning of additional world knowledge. However, the understanding of the underlying mechanisms of IFT remains significantly limited. In this paper, we design a knowledge intervention framework to decouple the potential underlying factors of IFT, thereby enabling individual analysis of different factors. Surprisingly, our experiments reveal that attempting to learn additional world knowledge through IFT often struggles to yield positive impacts and can even lead to markedly negative effects. Further, we discover that maintaining internal knowledge consistency before and after IFT is a critical factor for achieving successful IFT. Our findings reveal the underlying mechanisms of IFT and provide robust support for some very recent and potential future works.

## 🌟 Conclusion
---
**Conclusion 1.**  For IFT, there is little, if not even causing additional damage, benefits from the learning of world knowledge incongruent with parameter knowledge.

**Conclusion 2.**  The essence of an effective IFT lies in maintaining the consistency of model parameter knowledge before and after IFT.

## 🔥 Usage
---
1. use ICL to probe parameter knowledge of base LLMs
2. construct instruction data based on model parameter knowledge and golden world knowledge
3. use different groups of data to IFT models
4. eval the fine-tuned models

Modify the corresponding parameter Settings in the script and run it.
### eval
---
[eval/run_benchmark.sh](eval/run_benchmark.sh): mmlu eval
[eval/run_domain.sh](eval/run_domain.sh): domain eval
[eval/chat.py](eval/chat.py): use ICL to prompt base LLM to generate explanations for its prediction

### train
---
[train/run.sh](train/run.sh): train models

### analyse
---
[eval/analyse/metric.py](eval/analyse/metric.py): calculate ranking correlation and distribution's KL divergence of LLMs' prediction before and after IFT
[eval/analyse/partial_correlation.py](eval/analyse/partial_correlation.py): calculate the partial correlation of the fine-tuned model's performance and the pearson correlation coefficients of LLMs' prediction ranking on choices

## 🙏 Thanks
---
We refer to the code of the following public repositories, and we especially thank them for their open source.
[FastChat](https://github.com/lm-sys/FastChat)
[Baichuan-7B](https://github.com/baichuan-inc/Baichuan-7B/tree/main/evaluation)


## 📜 Citation
---
```
@misc{ren2024learning,
      title={Learning or Self-aligning? Rethinking Instruction Fine-tuning}, 
      author={Mengjie Ren and Boxi Cao and Hongyu Lin and Liu Cao and Xianpei Han and Ke Zeng and Guanglu Wan and Xunliang Cai and Le Sun},
      year={2024},
      eprint={2402.18243},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
