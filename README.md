<div align=center><img src="logo.jpg" style="zoom:30%;" /></div>

# D2Controller
Released code for our paper: [Dynamic Demonstrations Controller for In-Context Learning](https://arxiv.org/abs/2310.00385)


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
# Running Code
### Preprocess 
run the bash script `do_preprocess.sh` to transform original dataset files
```
bash do_preprocess.sh
```

### Run ICL
To obtain results on validation set, run script `run_icl.sh`
```
bash run_icl.sh
```
Notice that you should allocate names and directories for datasets and models in the script.

### Select k-shot setting
To obtain selected $k$-shot settings, run script `run_selectk.sh`
```
bash run_selectk.sh
```
Notice that you should allocate names and directories for datasets and models in the script.

For OPT-30B model, we use script `run_selectk_multi.sh` to obtain results
```
bash run_selectk_multi.sh
```

## GPT-3
We will release the code and scripts for GPT-3 later.

## Citation
```
@misc{zhao2023dynamic,
      title={Dynamic Demonstrations Controller for In-Context Learning}, 
      author={Fei Zhao and Taotian Pang and Zhen Wu and Zheng Ma and Shujian Huang and Xinyu Dai},
      year={2023},
      eprint={2310.00385},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
