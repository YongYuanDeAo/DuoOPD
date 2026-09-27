---
license: mit
task_categories:
- question-answering
- feature-extraction
- text-classification
- summarization
language:
- en
tags:
- biology
- chemistry
size_categories:
- 10K<n<100K
configs:
- config_name: v2
  data_files: 
  - split: test
    path: "data/v2/sciknoweval_test_v2.jsonl"

---
<div align="center">

<h1 align="center">  <img src="figure/logo.svg" width="5%" height="5%"> SciKnowEval </h1>
<h1 align="center"> Evaluating Multi-level Scientific Knowledge of Large Language Models </h1>
</div>

> Please refer to our [repository](https://github.com/HICAI-ZJU/sciknoweval) and [paper](https://arxiv.org/abs/2406.09098) for more details.

<div align="center">
<p align=right><b>博学之 ，审问之 ，慎思之 ，明辨之 ，笃行之。</b></p>
<p align=right>—— 《礼记 · 中庸》 <i>Doctrine of the Mean</i></p>

<div align=center><img src="figure/sciknoweval.png" width="80%" height="100%" /></div>

<p></p>
</div>

The <b>Sci</b>entific <b>Know</b>ledge <b>Eval</b>uation (<b>SciKnowEval</b>) benchmark for Large Language Models (LLMs) is inspired by the profound principles outlined in the “<i>Doctrine of the Mean</i>” from ancient Chinese philosophy. This benchmark is designed to assess LLMs based on their proficiency in **Studying Extensively**, **Enquiring Earnestly**, **Thinking Profoundly**, **Discerning Clearly**, and **Practicing Assiduously**. Each of these dimensions offers a unique perspective on evaluating the capabilities of LLMs in handling scientific knowledge.

## 🆕 News

- **\[Sep 2024\]** We released an [evaluation report of OpenAI o1](http://scimind.ai/sciknoweval/o1) with SciKnowEval.

- **\[Sep 2024\]** We have updated the SciKnowEval paper in [arXiv](https://arxiv.org/abs/2406.09098v2).

- **\[Jul 2024\]** We have recently added the Physics and Materials to SciKnowEval. You can access the dataset [here](https://huggingface.co/datasets/hicai-zju/SciKnowEval) and check out the leaderboard [here](http://scimind.ai/sciknoweval).

- **\[Jun 2024\]** We released the SciKnowEval Dataset and Leaderboard for Biology and Chemistry.


## 📌 Table of Contents

- [⌚️ Overview](#2)
- [🏹 QuickStart](#3)
  - [⬇️ Installation](#3.1)
  - [📜 Prepare data](#3.2)
  - [🛒 Prepare models](#3.3)
  - [🚀 Evaluate](#3.4)
- [🏅 Leaderboard](#4)
- [📝 Cite](#6)
- [✨ Acknowledgements](#7)
---

<h2 id="2">⌚️  Overview</h2>

<h3 id="2.1">✡️ Evaluated Abilities</h3>

* 📖 **L1**: **Studying extensively** (i.e., *knowledge memory*). This dimension evaluates the breadth of an LLM’s knowledge across various scientific domains. It measures the model’s ability to remember a wide range of scientific concepts.

* ❓ **L2**: **Enquiring earnestly** (i.e., *knowledge comprehension*). This aspect focuses on the LLM’s capacity for deep enquiry and exploration within scientific contexts, such as analyzing scientific texts, identifying key concepts, and questioning relevant information.

* 💡 **L3**: **Thinking profoundly** (i.e., *knowledge reasoning*). This criterion examines the model’s capacity for critical thinking, logical deduction, numerical calculation, function prediction, and the ability to engage in reflective reasoning to solve problems.

* 🔨 **L4**: **Discerning clearly** (i.e., *knowledge discernment*). This aspect evaluates the LLM’s ability to make correct, secure, and ethical decisions based on scientific knowledge, including assessing the harmfulness and toxicity of information, and understanding the ethical implications and safety concerns related to scientific endeavors.

* 🔬 **L5**: **Practicing assiduously** (i.e., *knowledge application*). The final dimension assesses the LLM’s capability to apply scientific knowledge effectively in real-world scenarios, such as analyzing complex scientific problems and creating innovative solutions.

<h3 id="2.2">🎯 Domains and Tasks</h3>
<div align=center><img src="figure/task.png" width="80%" height="100%" /></div>
<div align=center><img src="figure/task2.png" width="80%" height="100%" /></div>

<h3 id="2.3">📊 Data Stats</h3>

<div align=center><img src="figure/stats.png" width="80%" height="100%" /></div>

<h3 id="2.4">🛠️ Data Construction</h3>

<div align=center><img src="figure/data_collection.png" width="70%" height="100%" /></div>

* 🤖 *<b>Generating New QAs from Literature Corpus:</b>*
This method involves collecting scientific papers from sources like BioRxiv, PubMed, and textbook databases such as LibreTexts. Large Language Models (LLMs) are used to automate the generation of QA pairs by designing effective prompts based on domain experts' advice. These prompts guide the LLMs to extract relevant knowledge from literature and generate QA pairs that ensure answers are explicitly found in the original text without adding external information.

* 🔩 *<b>Refactoring the Existing QAs:</b>*
Additional QAs are sampled from existing scientific benchmarks like MedMCQA, SciEval, and others. LLMs are employed to refactor these QAs by rewriting questions and reordering options to avoid data contamination and leakage. In cases where QAs lack explicit annotations for their corresponding levels in SciKnowEval, LLMs automatically categorize the data into distinct levels.

* ⚗️ *<b>Transforming Scientific Databases:</b>*
This approach transforms data from biological and chemical databases (e.g., PubChem, UniProtKB) into textual formats suitable for LLM evaluation. It starts with quality screening, such as filtering invalid chemical structures, followed by the use of multiple question templates to convert structured data (like sequence annotations) into natural language QA pairs, including multiple-choice and true/false formats.

* ✅ *<b>Quality Control</b>*: To further ensure the accuracy and reliability of the dataset, each task within our dataset undergoes validation by two domain experts in biology and chemistry. Experts evaluate the relevance and correctness of the scientific problems and solutions.

<h2 id="3">🏹 QuickStart</h2>
<h3 id="3.1">⬇️ Step 1: Installation</h3>

To evaluate LLMs on SciKnowEval, first clone the repository:
```bash
git clone https://github.com/HICAI-ZJU/SciKnowEval.git
cd SciKnowEval
```
Next, set up a conda environment to manage the dependencies:
```bash
conda create -n sciknoweval python=3.10.9
conda activate sciknoweval
```
Then, install the required dependencies:
```bash
pip install -r requirements.txt
```


<h3 id="3.2">📜 Step 2 : Prepare data</h3>

#### Getting Started with SciKnowEval Benchmark

1. **Download the SciKnowEval Benchmark Data**: To begin evaluating language models using the SciKnowEval benchmark, you should first download our dataset. There are two available sources:

    * 🤗 **HuggingFace Dataset Hub**: Access and download the dataset directly from our HuggingFace page: [https://huggingface.co/datasets/hicai-zju/SciKnowEval](https://huggingface.co/datasets/hicai-zju/SciKnowEval)

    * **Repository Data Folder**: The dataset is organized by level (L1~L5) and task within the `./raw_data/` folder of this repository. You may download parts separately and consolidate them into a single JSON file as needed.

2. **Prepare Your Model’s Predictions**: Utilize the official evaluation script `eval.py` provided in this repository to assess your model. You are required to prepare your model's predictions in the following JSON format, where each entry must preserve all the original attributes (which can be found in the dataset you downloaded) of the data such as question, choices, answerKey, type, domain, level, task, and subtask. Add your model's predicted answer under the "response" field.

Example JSON format for model evaluation:
```json
[
  {
    "question": "What triggers the activation of platelet integrins?",
    "choices": {
      "text": ["White blood cells", "Collagen exposure", "Adrenaline release", "Nutrient absorption"],
      "label": ["A", "B", "C", "D"]
    },
    "answerKey": "B",
    "type": "mcq-4-choices",
    "domain": "Biology",
    "details": {
      "level": "L2",
      "task": "Cellular Function",
      "subtask": "Platelet Activation"
    },
    "response": "B"  // Insert your model's prediction here
  },
  // Additional entries...
]
```

#### ❗Key Points to Remember
- **Preserve All Original Fields**: Ensure each JSON object retains all the original data fields to maintain the integrity of the evaluation.
- **Model Predictions**: Place your model’s predictions in the "response" field of each JSON object.

By following these guidelines, you can effectively use the SciKnowEval benchmark to evaluate the performance of language models across various scientific tasks and levels.


<h3 id="3.3">🛒 Step 3: Prepare models</h3>

**1. For relation extraction tasks, we need to calculate the text similarity with `word2vec` model. We use *GoogleNews-vectors* pretrained model as the default model.**

- Download `GoogleNews-vectors-negative300.bin.gz` from [this link](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) to local.

> The relation extraction evaluation code was initially developed by the [AI4S Cup](https://bohrium.dp.tech/competitions/3793785610?tab=datasets) team, thanks for their great work!🤗

**2. For tasks that use GPT for scoring, we use OpenAI API to assess answers.**

- Please set your OpenAI API key in the `OpenAI_API_KEY` environment variable. Use `export OPENAI_API_KEY="YOUR_API_KEY"` to set the environment variable.

- If you do not set the `OPENAI_API_KEY` environment variable, the evaluation will automatically **skip the tasks that require GPT scoring**.

- 📣 We select `gpt-4o` as the default evaluator !


<h3 id="3.4">🚀 Step 4: Evaluate</h3>

You can run `eval.py` to evaluate your model:

```bash
data_path="your/model/predictions.json"
word2vec_model_path="path/to/GoogleNews-vectors-negative300.bin"
gen_evaluator="gpt-4o" # the correct model name in OpenAI
output_path="path/to/your/output.json"

export OPENAI_API_KEY="YOUR_API_KEY"
python eval.py \
  --data_path $data_path \
  --word2vec_model_path $word2vec_model_path \
  --gen_evaluator $gen_evaluator \
  --output_path $output_path
```
 

<h2 id="4">🏅 Leaderboard</h2>

The latest leaderboards are shown [here](http://scimind.ai/sciknoweval/#overall).

| Models          | Biology |    |    |    |    |    | Chemistry |    |    |    |    |    |  Overall Rank  |
|-----------------|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:------------:|
|                 | L1   | L2   | L3   | L4   | L5   | All  | L1   | L2   | L3   | L4   | L5   | All  |  |
| 🥇 **GPT-4o**      | **2.00** | **2.25** | **6.00** | 4.00  | **1.20** | **3.28** | **1.00** | _2.29_ | _4.00_ | 7.00  | 3.75  | **3.46** | 1  |
| 🥈 **Gemini1.5-Pro-latest** | 4.50  | 5.12  | _6.14_ | **2.67** | 6.60  | 5.36  | _2.67_ | 4.00  | **3.57** | **1.33** | 11.75 | _4.67_ | 2  |
| 🥉 **GPT-4-Turbo-2024-04-09** | 4.00  | 5.50  | 7.86  | _3.33_ | 4.00  | 5.48  | 3.00  | **1.57** | 7.29  | _4.67_ | 7.75  | 4.83  | 3   |
| **Claude3-sonnet-20240229** | 5.50  | _4.12_ | 8.43  | 4.00  | _2.00_ | _5.00_ | 6.00  | 4.43  | 7.86  | 8.00  | 6.00  | 6.33  | 4  |
| **GPT-3.5-Turbo-0125** | _2.50_ | 7.62  | 11.86 | 4.67  | 7.60  | 8.04  | 9.00  | 7.86  | 8.29  | 7.00  | 8.00  | 8.04  | 5     |
| **Llama3-8B-Inst** | 8.50  | **5.50** | 11.71 | **7.67** | 10.80 | **8.80** | **6.00** | **6.29** | 8.57  | **7.33** | 14.25 | 8.38  | 6  |
| **Qwen1.5-14B-Chat** | **5.50** | 10.38 | 8.71  | 9.00  | **8.40** | 8.96  | 9.33  | 7.14  | **6.43** | 8.00  | 10.50 | 7.88  | 7  |
| **ChemDFM-13B**     | 6.50  | 11.12 | 12.00 | 9.67  | 12.40 | 11.08 | 6.67  | 9.43  | 8.29  | 8.33  | _1.75_ | **7.33** | 8  |
| **ChemLLM-20B-Chat** | 12.50 | 6.62  | 10.14 | 14.67 | 13.00 | 10.32 | 10.00 | 7.71  | 11.00 | 16.33 | 4.00  | 9.42  | 9     |
| **Qwen1.5-7B-Chat** | 9.00  | 10.50 | 13.71 | 8.00  | 10.60 | 11.00 | 10.67 | 9.86  | 9.29  | 11.67 | 13.50 | 10.62 | 10  |
| **MolInst-Llama3-8B** | 13.50 | 9.88  | **7.86** | 12.00 | 18.20 | 11.52 | 9.33  | 9.57  | 7.43  | 9.33  | 17.75 | 10.25 | 11  |
| **ChatGLM3-6B**    | 12.00 | 14.25 | 11.43 | 10.00 | 12.00 | 12.32 | 15.33 | 15.00 | 15.00 | 12.33 | 12.75 | 14.33 | 12 |
| **Galactica-30B**   | 11.00 | 13.75 | 8.43  | 16.67 | 16.80 | 13.00 | 7.67  | 16.43 | 13.00 | 16.67 | 16.00 | 14.29 | 13    |
| **Gemma1.1-7B-Inst** | 16.00 | 16.75 | 11.71 | 14.67 | 12.80 | 14.24 | 17.00 | 15.86 | 12.57 | 11.00 | 7.25  | 13.00 | 14 |
| **Llama2-13B-Chat** | 19.00 | 11.38 | 17.14 | 10.67 | 10.60 | 13.36 | 18.67 | 13.86 | 15.57 | 10.33 | 14.00 | 14.54 | 15  |
| **Mistral-7B-Inst** | 11.00 | 13.12 | 14.71 | 12.67 | 18.20 | 14.30 | 14.33 | 14.14 | 15.29 | **7.33** | 19.00 | 14.46 | 16  |
| **SciGLM-6B**       | 16.00 | 14.12 | 11.43 | 16.00 | 16.60 | 14.24 | 16.00 | 15.29 | 13.14 | 17.67 | 15.25 | 15.04 | 17    |
| **ChemLLM-7B-Chat** | 15.00 | 15.88 | 13.86 | 14.33 | 16.60 | 15.20 | 15.33 | 14.86 | 15.43 | 16.00 | 7.75  | 14.04 | 18    |
| **Galactica-6.7B**  | 17.50 | 16.50 | 11.86 | 18.00 | 19.20 | 16.00 | 13.00 | 17.86 | 13.00 | 13.00 | 18.50 | 15.33 | 19    |
| **LlaSMol-Mistral-7B** | 19.50 | 16.75 | 14.14 | 19.67 | 17.20 | 16.68 | 19.33 | 18.71 | 16.29 | 20.00 | **1.25** | 15.33 | 20  |


<h2 id="6">📝 Cite</h2>

```
@misc{feng2024sciknoweval,
    title={SciKnowEval: Evaluating Multi-level Scientific Knowledge of Large Language Models},
    author={Kehua Feng and Keyan Ding and Weijie Wang and Xiang Zhuang and Zeyuan Wang and Ming Qin and Yu Zhao and Jianhua Yao and Qiang Zhang and Huajun Chen},
    year={2024},
    eprint={2406.09098},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

<h2 id="7"> ✨ Acknowledgements </h2>

Special thanks to the authors of [LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset](https://github.com/OSU-NLP-Group/LLM4Chem), and the organizers of the [AI4S Cup - LLM Challenge](https://bohrium.dp.tech/competitions/3793785610?tab=datasets) for their inspiring work.

The sections evaluating molecular generation in [`evaluation/utils/generation.py`](./evaluation/utils/generation.py), as well as [`evaluation/utils/relation_extraction.py`](./evaluation/utils/relation_extraction.py), are grounded in their research. Grateful for their valuable contributions ☺️!

### Other Related Projects

- [SciEval](https://github.com/OpenDFM/SciEval)
- [SciBench](https://github.com/mandyyyyii/scibench)
- [SciAssess](https://github.com/sci-assess/SciAssess)
