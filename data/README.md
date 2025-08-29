---
license: cc-by-4.0
task_categories:
- visual-question-answering
language:
- en
pretty_name: VRSBench
size_categories:
- 10K<n<100K
tags:
- coral reef image, vision-language models
---

# CoralVQA: A Large-Scale Visual Question Answering Dataset for Coral Reef Image Understanding

<center>
    <img src="fig_example.png" alt="CoralVQA: A Large-Scale Visual Question Answering Dataset for Coral Reef Image Understanding.">
</center>


In this work, we introduce CoralVQA, the first large-scale VQA dataset for coral reef analysis. It contains 12,805 real-world coral images from 67 coral genera collected from 3 oceans, along with 277,653 question-answer pairs that comprehensively assess ecological and health-related conditions. To construct this dataset, we develop a semi-automatic data construction pipeline in collaboration with marine biologists to ensure both scalability and professional-grade data quality. CoralVQA presents novel challenges and provides a comprehensive benchmark for studying vision-language reasoning in the context of coral reef images. By evaluating several state-of-the-art LVLMs, we reveal key limitations and opportunities. These insights form a foundation for future LVLM development, with a particular emphasis on supporting coral conservation efforts.

### Using  Datasets

Detailed file documentation is described in the following.

- CoralVQA\_Image.zip: The file contains 12,805 real-world coral images from 67 coral genera.
- CoralVQA\_train.jsonl: The JSONL file consists of all question-answer pairs of the training datasets. The JSONL file is organized following the LLAVA data format. Each row is a question-answer pair. Each question-answer pair contains an ID, image path, conversations, and response to conversations.
- CoralVQA\_test.jsonl: The JSONL file comprises 27,984 question-answer pairs. Each row contains four components: ID, image path, question, and question.
- CoralVQA\_cross_region.jsonl: The cross-region question-answer dataset is designed to evaluate the generalization capability of vision-language models. The file contains 22,943 question-answer pairs in standard JSONL format.
- Code: The folder contains both fine-tuning and evaluation code for all methods, including Mini-Gemini(7B), Qwen2.5VL(7B), BLIP3, InternVL2.5(8B).

Attribute explanations in annotation files:

- oquadratid:the corresponding coral images
- y,x: positional coordinates of coral genera
- label_name: names of coral genera
- label: short names

## Key Contributions
- To the best of our knowledge, CoralVQA is the first large-scale VQA dataset dedicated to coral reef understanding. It contains 12,805 real coral images across 67 genera from 3 oceans, and 277,653 question–answer pairs from 16 dimensions.
- We design a semi-automatic vision-language coral data collection pipeline, which includes six steps: dataset collection, label cleaning and re-annotating, attribute extraction, question generation, and human verification. Our pipeline can be widely applied to other marine domains, such as visual reasoning tasks in mangrove ecosystem imagery.
- We systematically evaluate the performance of several state-of-the-art LVLMs on CoralVQA across multiple coral-related tasks, serving as a baseline and highlighting opportunities for future research. 

### Model Training
For the above three tasks, we benchmark state-of-the-art models, includingMini-Gemini(7B), Qwen2.5VL(7B), BLIP3, InternVL2.5(8B), to demonstrate the potential of LVMs for coral reef image understanding. To ensure a fair comparison, we reload the models that are initially trained on large-scale image-text alignment datasets, and then finetune each method using the training set of our CoralVQA dataset.  All comparison models are trained on single node with 4 NVIDIA H20 GPUs. All models are run in an environment with PyTorch 2.1.0, Python 3.10, and CUDA 12.1. For a fair comparison, all visual-language models are fine-tuned for three epochs with an initial learning rate of 2e-4 and a cosine learning rate decay schedule for optimization.

### Broader Impact of Dataset
To the best of our knowledge, CoralVQA is the first large-scale VQA dataset dedicated to coral reef understanding. The CoralVQA dataset provides a comprehensive benchmark for developing and evaluating vision-language models oriented towards coral reef monitoring and conservation. On the one hand, vision-language models trained on the CoralVQA dataset can assist researchers in rapidly identifying coral genera, coral growth conditions, and coral health status, thereby enhancing ecological monitoring efficiency. On the other hand, an interactive coral reef visual question answering system can further facilitate public learning of coral reef knowledge.

### Discussion of Limitations
The taxonomic resolution of the CoralVQA dataset is at the genus level for corals. However, annotating corals at the species level presents significant challenges. First, different species within the same coral genus exhibit highly similar morphological features, making it difficult even for experts to distinguish them rapidly based on image data alone. Second, species-level annotation requires substantially more time and effort.

### Licensing Information
The dataset is released under the [Creative Commons Attribution Non Commercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en), which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited.

### Citation Information

```bibtex
@article{han2025coralvqa,
  title={CoralVQA: A Large-Scale Visual Question Answering Dataset for Coral Reef Image Understanding},
  author={Han, Hongyong and Wang, Wei and Zhang, Gaowei and Li, Mingjie and Wang, Yi},
  journal={arXiv preprint arXiv:2507.10449},
  year={2025}
}
```

## Contact
hanhongyong@bupt.edu.cn, BUPT