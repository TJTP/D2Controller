# D2Controller

Code and data for "[Dynamic Demonstrations Controller for In-Context Learning](https://arxiv.org/abs/2310.00385)".

##  Pilot Experiments
<div align="center">
<img src="figs/pilot.jpg" width="700px">
</div>

- Increasing the number of demonstrations does not necessarily improve the model performance
- The optimal k-shot setting differs depending on specific datasets and models

##  Method

Based on the above analyses, we propose a method named D2Controller:

<div align="center">
<img src="figs/D2Controller.png" width="700px">
</div>

-  D2Controller not only boosts ICL performance but also saves time and space during inference of the LLMs
-  We apply D2Controller to eight different sizes of LLMs and realize an average of 5.4% relative improvement across ten datasets. We also extend D2Controller to previous ICL models and yield competitive results


## Environment Setup
Create a new virtual environment with `Python==3.9.16`
```
conda create --name d2controller python=3.9.16
```

Install `requirement.txt`
```
pip install -r requirement.txt
```

## LLMs

Please download LLMs (except GPT-3) from the [HuggingFace](https://huggingface.co/). The LLMs we use in our paper include [`gpt2-medium`](https://huggingface.co/gpt2-medium), [`gpt2-large`](https://huggingface.co/gpt2-large), [`gpt2-xl`](https://huggingface.co/gpt2-xl), [`Cerebras-GPT-2.7B`](https://huggingface.co/cerebras/Cerebras-GPT-2.7B), [`Cerebras-GPT-6.7B`](https://huggingface.co/cerebras/Cerebras-GPT-6.7B), [`opt-13b`](https://huggingface.co/facebook/opt-13b) and [`opt-30b`](https://huggingface.co/facebook/opt-30b). Put the model files under `llm/` directory. For example
```

llm/
  |--gpt2-medium/
      |--config.json
      |--merges.txt
      |--pytorch_model.bin
      |--tokenizer.json
      |--vocab.json
  |--gpt2-large/
      |--...
      ...
  ...
```
## Running Code
### Preprocess 
run the bash script `do_preprocess.sh` to transform original dataset files
```
bash scripts/do_preprocess.sh
```


### Select k-shot setting
To obtain selected $k$-shot settings, run the script `run_selectk.sh`
```
bash scripts/run_selectk.sh
```

For OPT-30B model, we use the script `run_selectk_multi.sh` to obtain results
```
bash scripts/run_selectk_multi.sh
```
Notice that you should allocate names and directories for datasets and models in the script.


### Obtain ICL results
To obtain ICL results based on the select $k$-shot setting, run the script `run_icl.sh`
```
bash scripts/run_icl.sh
```

### Find Oracle k-shot setting
To obtain Oracle $k$-shot setting on validation set, run the script `run_oracle_k.sh`
```
bash scripts/run_oracle_k.sh
```

## GPT-3
We will release the code and scripts for GPT-3 later.

## Results

**Main Results**
<details>

| **Model** | **Parameters** | **Setting** |**SST-2** | **SST-5** | **DBPedia** | **MR** | **CR** | **MPQA** | **Subj** | **AGNews** | **RTE** | **CB** | **AVG** |
|-----------|:---------:|---------|:---------:|:---------:|:----------:|:---------:|:---------:|:---------:|:----------:|:---------:|:---------:|:----------:|:---------:|
| GPT-2 | 0.3B | Default | 58.1&plusmn;13.1 | 24.1&plusmn;7.4 | 60.6&plusmn;7.2 | 54.2&plusmn;10.6 | 50.6&plusmn;0.4 | 59.6&plusmn;15.8 | 53.4&plusmn;5.3 | 48.7&plusmn;8.5 | 51.3&plusmn;1.7 | 48.6&plusmn;6.4 | 50.9 |
| GPT-2 | 0.3B | D2Controller |  74.1&plusmn;9.3 | 31.6&plusmn;8.6 | 60.6&plusmn;7.2 | 53.8&plusmn;7.0 | 67.7&plusmn;11.4 | 57.1&plusmn;9.7 | 53.8&plusmn;4.2 | 48.7&plusmn;8.5 | 48.7&plusmn;2.9 | 48.6&plusmn;6.4 | 54.5 |
| GPT-2 | 0.3B | Oracle | 74.1&plusmn;9.3 | 31.6&plusmn;8.6 | 60.6&plusmn;7.2 | 56.0&plusmn;9.9 | 67.7&plusmn;11.4 | 64.5&plusmn;16.0 | 58.6&plusmn;12.8 | 49.4&plusmn;18.4 | 51.3&plusmn;1.7 | 50.0&plusmn;9.2 | 56.4 |
| GPT-2 | 0.8B | Default | 71.8&plusmn;12.1 | 37.8&plusmn;6.8 | 63.4&plusmn;6.0 | 71.1&plusmn;15.6 | 80.5&plusmn;11.4 | 65.8&plusmn;11.3 | 59.9&plusmn;12.2 | 65.6&plusmn;17.2 | 53.1&plusmn;3.4 | 37.1&plusmn;14.5 | 60.6 |
| GPT-2 | 0.8B | D2Controller | 65.9&plusmn;15.2 | 37.5&plusmn;5.1 | 63.4&plusmn;6.0 | 71.1&plusmn;15.6 | 80.5&plusmn;11.4 | 70.5&plusmn;5.2 | 69.4&plusmn;12.4 | 65.6&plusmn;17.2 | 53.1&plusmn;3.4 | 47.5&plusmn;3.2 | 62.4 |
| GPT-2 | 0.8B | Oracle | 71.8&plusmn;12.1 | 39.6&plusmn;5.1 | 63.4&plusmn;6.0 | 71.1&plusmn;15.6 | 80.5&plusmn;11.4 | 74.5&plusmn;8.8 | 69.4&plusmn;12.4 | 65.6&plusmn;17.2 | 53.8&plusmn;4.4 | 49.3&plusmn;3.7 | 63.9 |
| GPT-2 | 1.5B | Default | 70.3&plusmn;6.6 | 35.4&plusmn;8.4 | 82.0&plusmn;2.0 | 52.0&plusmn;3.8 | 52.0&plusmn;3.2 | 66.7&plusmn;8.2 | 57.3&plusmn;10.5 | 78.2&plusmn;6.7 | 53.1&plusmn;1.7 | 52.9&plusmn;6.3 | 60.0 |
| GPT-2 | 1.5B | D2Controller | 81.3&plusmn;5.4 | 35.4&plusmn;8.4 | 82.0&plusmn;2.0 | 72.2&plusmn;13.9 | 66.2&plusmn;16.7 | 83.9&plusmn;1.5 | 64.1&plusmn;11.3 | 78.2&plusmn;6.7 | 53.1&plusmn;2.9 | 52.9&plusmn;6.3 | 67.0 |
| GPT-2 | 1.5B | Oracle | 81.3&plusmn;5.4 | 40.6&plusmn;5.4 | 82.0&plusmn;2.0 | 72.2&plusmn;13.9 | 66.2&plusmn;16.7 | 83.9&plusmn;1.5 | 64.1&plusmn;11.3 | 81.3&plusmn;7.5 | 53.1&plusmn;2.9 | 57.9&plusmn;9.8 | 68.2 |
| Cerebras-GPT | 2.7B | Default | 65.5&plusmn;13.8 | 28.4&plusmn;4.3 | 81.8&plusmn;1.4 | 65.1&plusmn;11.2 | 85.8&plusmn;4.2 | 64.2&plusmn;11.6 | 69.3&plusmn;14.4 | 69.5&plusmn;3.2 | 48.1&plusmn;1.1 | 52.5&plusmn;9.5 | 63.0 |
| Cerebras-GPT | 2.7B | D2Controller | 77.3&plusmn;7.7 | 34.3&plusmn;4.8 | 81.8&plusmn;1.4 | 76.0&plusmn;7.7 | 87.4&plusmn;1.5 | 81.6&plusmn;2.1 | 74.2&plusmn;7.6 | 77.3&plusmn;4.1 | 48.0&plusmn;1.1 | 54.6&plusmn;2.7 | 69.3 |
| Cerebras-GPT | 2.7B | Oracle |  80.7&plusmn;9.1 | 34.3&plusmn;4.8 | 81.8&plusmn;1.4 | 76.0&plusmn;7.7 | 87.4&plusmn;1.5 | 82.9&plusmn;3.0 | 74.2&plusmn;7.6 | 77.3&plusmn;4.1 | 49.6&plusmn;2.3 | 55.7&plusmn;5.0 | 70.0 |
| Cerebras-GPT | 6.7B | Default | 83.4&plusmn;8.5 | 38.3&plusmn;1.8 | 87.0&plusmn;2.4 | 88.0&plusmn;1.1 | 89.0&plusmn;3.1 | 75.2&plusmn;10.3 | 72.0&plusmn;14.5 | 79.2&plusmn;2.4 | 52.3&plusmn;2.3 | 52.5&plusmn;8.0 | 71.7 |
| Cerebras-GPT | 6.7B | D2Controller | 82.0&plusmn;11.3 | 39.5&plusmn;3.7 | 87.0&plusmn;2.4 | 86.8&plusmn;1.9 | 90.5&plusmn;0.9 | 83.8&plusmn;3.3 | 79.2&plusmn;12.5 | 80.2&plusmn;1.5 | 52.8&plusmn;2.5 | 57.9&plusmn;7.2 | 74.0 |
| Cerebras-GPT | 6.7B | Oracle |  88.6&plusmn;2.7 | 43.6&plusmn;1.6 | 87.0&plusmn;2.4 | 88.0&plusmn;1.1 | 90.6&plusmn;2.8 | 83.8&plusmn;3.3 | 79.2&plusmn;12.5 | 80.2&plusmn;1.5 | 53.4&plusmn;1.7 | 57.9&plusmn;3.0 | 75.2 |
| OPT | 13B | Default | 81.2&plusmn;6.7 | 43.3&plusmn;4.6 | 92.3&plusmn;2.1 | 87.8&plusmn;2.7 | 91.4&plusmn;3.3 | 75.0&plusmn;6.7 | 79.1&plusmn;12.7 | 81.9&plusmn;2.9 | 54.4&plusmn;4.2 | 58.9&plusmn;8.1 | 74.5 |
| OPT | 13B | D2Controller | 90.2&plusmn;5.8 | 43.3&plusmn;4.6 | 92.3&plusmn;2.1 | 87.8&plusmn;2.7 | 91.3&plusmn;2.1 | 72.0&plusmn;9.4 | 91.6&plusmn;2.0 | 82.6&plusmn;1.5 | 55.8&plusmn;3.1 | 58.9&plusmn;8.1 | 76.6 |
| OPT | 13B | Oracle |  90.9&plusmn;3.7 | 48.0&plusmn;2.8 | 92.3&plusmn;2.1 | 91.8&plusmn;0.6 | 93.3&plusmn;1.2 | 78.6&plusmn;7.3 | 91.6&plusmn;2.0 | 82.6&plusmn;1.5 | 55.8&plusmn;3.1 | 73.2&plusmn;12.4 | 79.8 |
| OPT | 30B | Default | 92.3&plusmn;1.3 | 40.9&plusmn;1.8 | 91.7&plusmn;3.7 | 91.8&plusmn;2.1 | 87.3&plusmn;3.3 | 78.8&plusmn;6.2 | 76.1&plusmn;4.9 | 78.7&plusmn;3.6 | 63.0&plusmn;3.1 | 60.0&plusmn;8.2 | 76.1 |
| OPT | 30B | D2Controller | 92.3&plusmn;1.3 | 42.0&plusmn;2.8 | 91.7&plusmn;3.7 | 93.4&plusmn;1.1 | 87.3&plusmn;2.7 | 85.7&plusmn;3.8 | 83.4&plusmn;8.6 | 76.7&plusmn;4.5 | 61.6&plusmn;2.8 | 60.0&plusmn;8.2 | 77.4 |
| OPT | 30B | Oracle | 92.8&plusmn;1.6 | 45.2&plusmn;3.1 | 91.7&plusmn;3.7 | 93.4&plusmn;1.1 | 87.7&plusmn;3.9 | 85.7&plusmn;3.8 | 83.4&plusmn;8.6 | 78.7&plusmn;3.6 | 63.0&plusmn;3.1 | 60.0&plusmn;8.2 | 78.1 |

</details>

---

**D2Controller is beneficial to other ICL models**
<details>
  
| **Model** | **GPT-2 0.3B** | **GPT-2 0.8B** | **GPT-2 1.5B** | **Cerebras-GPT 2.7B** | **Cerebras-GPT 6.7B** |
|-----------|:---------:|:---------:|:---------:|:----------:|:---------:|
| KATE |  66.7 | 69.4 | 67.7 | 71.6 | 77.6 |
| KATE+D2Controller | 68.8 | 70.5 | 69.4 | 74.7 | 77.9 |
| GlobalE | 59.5 | 67.7 | 69.8 | - | - |
| GlobalE+D2Controller |  61.5 | 68.7 | 71.6 | - | - |
| Contextual Calibration | 59.5 | 64.2 | 63.9 | 67.2 | 72.5 |
| Contextual Calibration+D2Controller | 60.8 | 66.6 | 65.4 | 68.7 | 73.5 |
| kNN Prompting | 74.8 | 76.0 | 77.3 | 77.8 | 79.0 |
| kNN Prompting+D2Controller | 75.8 | 77.1 | 78.2 | 78.1 | 79.7 |

</details>

---

**D2Controller can achieve competitive results on a small number of in-context example groups**
<details>

<div align="center">
<img src="figs/ICL.jpg" width="400px">
</div>

</details>

---

**IICScore is effective in selecting evaluation examples**
<details>

| **Model** | **GPT-2 0.3B** | **GPT-2 0.8B** | **GPT-2 1.5B** | **Cerebras-GPT 2.7B** | **Cerebras-GPT 6.7B** |
|-----------|:---------:|:---------:|:---------:|:----------:|:---------:|
| Random | 54.1 | 59.2 | 63.5 | 68.0 | 72.9 |
| D2Controller-ED | 54.4 | 59.2 | 64.0 | 67.1 | 72.6 |
| D2Controller-Cos | 54.9 | 59.3 | 62.2 | 68.3 | 72.4 |
| D2Controller |  54.5 | 62.4 | 66.9 | 69.3 | 74.0 |

</details>

---

**Dynamically selecting k performs better than using the maximum k**
<details>

| **Model** | **GPT-2 0.3B** | **GPT-2 0.8B** | **GPT-2 1.5B** | **Cerebras-GPT 2.7B** | **Cerebras-GPT 6.7B** |
|-----------|:---------:|:---------:|:---------:|:----------:|:---------:|
| kmax-shot setting | 54.1 | 58.7 | 66.0 | 65.4 | 73.0 |
| D2Controller | 54.5 | 62.4 | 66.9 | 69.3 | 74.0 |

</details>

---

## Citation
If you find this helpful, please cite our paper.

```bibtex
@misc{zhao2023dynamic,
      title={Dynamic Demonstrations Controller for In-Context Learning}, 
      author={Fei Zhao and Taotian Pang and Zhen Wu and Zheng Ma and Shujian Huang and Xinyu Dai},
      year={2023},
      eprint={2310.00385},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
