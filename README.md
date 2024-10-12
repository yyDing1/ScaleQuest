<p align="center">
<h1 align="center">Unleashing Reasoning Capability of LLMs<br>via Scalable Question Synthesis from Scratch</h1>

<p align="center">
    <a href="https://github.com/yyDing1/ScaleQuest/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/yyDing1/ScaleQuest"></a>
    <a href="https://huggingface.co/collections/dyyyyyyyy/scalequest-670a7dc2623c91990f28913b"><img alt="Pretrained Models" src="https://img.shields.io/badge/🤗 HuggingFace-Data & Models-green"></a>
    <a href=""><img alt="Paper" src="https://img.shields.io/badge/📄-Paper-orange"></a>
    <a href="https://opennlg.cn/"><img src="https://img.shields.io/badge/Organization-OpenNLG%20Group-blueviolet"></a>
</p>

We introduce ScaleQuest, a scalable, cost-effective, and novel data synthesis method that utilizes small-size open-source models to generate questions from scratch without the need for seed data with complex augmentation constraints.

This repository contains our complete data synthesis method, including:

1. Training a question generator through question fine-tuning (code in the `qft_train` folder).
2. Constructing preference data (code in the `question_optim` folder) and performing question preference optimization (code in the `qpo_train` folder).
3. Using the trained question generator to synthesize questions (code in the `data_generation` folder).
4. Applying a filtering process to the generated questions (code in the `question_filtering` folder).
5. Generating responses (code in the `data_generation` folder) and applying a reward filtering strategy (code in the `reward_filtering` folder).
6. For instruction-tuning and evaluation, we directly use the DART-Math framework.

We randomly sampled 100 generated data points and placed them in `data_samples/samples.jsonl`

## Method Overview

![](img/method.png)

## Model Performance

![](img/results.png)


