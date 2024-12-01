# RMCBench
Benchmarking Large Language Models’ Resistance to Malicious Code Generation Prompts

```
██████╗ ███╗   ███╗ ██████╗██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
██╔══██╗████╗ ████║██╔════╝██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
██████╔╝██╔████╔██║██║     ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
██╔══██╗██║╚██╔╝██║██║     ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
██║  ██║██║ ╚═╝ ██║╚██████╗██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝                
```



## Why do we need to do this study?

The large language models be used to generate malicious code!!!

This is a hidden danger to the security of LLMs content.

<img src="README.assets/good-and-bad-4-1.png" alt="good-and-bad-4-1" style="zoom: 25%;" />

Through RMCBench, you can conduct a comprehensive evaluation of LLMs' ability to resist malicious code generation.



## Result Leaderboard

GOOD：LLMs refuse to generate malicious code

| LLM                             | GOOD(%)   | BAD(%)    | UNCLEAR(%) |
| ------------------------------- | --------- | --------- | ---------- |
| llama-2-13b-chat-hf             | **48.84** | 49.26     | 1.90       |
| deepseek-coder-7b-instruct-v1.5 | 44.19     | 55.81     | 0.00       |
| Meta-Llama-3-8B-Instruct        | 43.55     | 56.24     | 0.21       |
| mpt-7b-chat                     | 39.96     | 57.08     | 2.96       |
| llama-2-7b-chat-hf              | 38.27     | 59.20     | 2.54       |
| gpt-4                           | 35.73     | 64.27     | 0.00       |
| CodeLlama-13b-Instruct-hf       | 30.66     | 68.92     | 0.42       |
| gpt-3.5-turbo                   | 18.39     | 81.18     | 0.42       |
| zephyr-7b-beta                  | 8.46      | **90.70** | 0.85       |
| vicuna-7b-v1.3                  | 4.86      | 84.14     | **10.99**  |
| tulu-2-13b                      | 2.96      | 90.27     | 6.77       |
| **Average**                     | 28.71     | 68.83     | 2.46       |



## Characteristics

### multi-scenarios

- text-to-code
- code-to-code

### multi-tasks

- text-to-code generation (Level 1 - 3)
- code completion
- code translation

## data details
The following is an explanation of all the fields in the dataset.
### These fields are present in all samples:
- **model_name:** The full name of the LLM being tested.
- **pid:** The ID of the prompt.
- **category:** The scenario of malicious code generation (text-to-code, code-to-code).
- **task:** The specific task of malicious code generation (text-to-code generation, code translation, code completion).
- **prompt:** The prompt that instructs the LLMs to generate malicious code.
- **malicious functionality:** The specific malicious intent/functionality of the malicious code.
- **malicious categories:** The category of malicious code corresponding to the malicious intent/functionality.
- **input_tokens:** The token length of the prompt.
- **response:** The response from the LLMs.
- **label:** The automated labeling results from ChatGPT-4.
- **check:** The results of manual sampling checks on the label.


### These fields are specific to the text-to-code scenario:
- **level:** The difficulty level of text-to-code.
- **level description:** The description and explanation of the level.
- **jid:** The ID of the jailbreak template (in level 3).

### These fields are specific to the code-to-code scenario:
- **cid:** The ID of the malicious code sample we collected.
- **original code:** The complete malicious code sample we collected.
- **language:** The programming language of the malicious code.
- **code lines:** The number of lines in the malicious code.
- **source:** The source of the malicious code.

### These fields are specific to the code-to-code scenario's code completion task:
- **code to be completed:** The remaining malicious code after being hollowing out.
- **missing part:** The hollowed out code (the code that needs to be completed).
- **completion level:** The level of code completion (token-level, line-level, multiline-level, function-level).
- **completion position:** The position of code completion (next token, fill-in-middle).

## Paper Link

📝[Arxiv](https://arxiv.org/abs/2409.15154)
📝[ACM Digital Library](https://dl.acm.org/doi/10.1145/3691620.3695480)

## Dataset
🌟 [Github](https://github.com/qing-yuan233/RMCBench/tree/main/data)
🤗 [Hugging Face](https://huggingface.co/datasets/zhongqy/RMCBench)


## Citation

```
@inproceedings{10.1145/3691620.3695480,
author = {Chen, Jiachi and Zhong, Qingyuan and Wang, Yanlin and Ning, Kaiwen and Liu, Yongkun and Xu, Zenan and Zhao, Zhe and Chen, Ting and Zheng, Zibin},
title = {RMCBench: Benchmarking Large Language Models' Resistance to Malicious Code},
year = {2024},
isbn = {9798400712487},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3691620.3695480},
doi = {10.1145/3691620.3695480},
numpages = {12},
keywords = {large language models, malicious code, code generation},
location = {Sacramento, CA, USA},
series = {ASE '24}
}
```