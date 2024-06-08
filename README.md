# RMCBench

Benchmarking Large Language Models' Resistance to Malicious Code

## Why do we need to do this study?

The large language models be used to generate malicious code!!!

This is a hidden danger to the security of LLMs content.

<img src="README.assets/good-and-bad-4-1.png" alt="good-and-bad-4-1" style="zoom: 25%;" />



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



## Paper Link

```
hold on
```



## Citation

```
hold on
```

