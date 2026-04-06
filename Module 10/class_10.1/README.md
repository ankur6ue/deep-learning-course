Problem1: Minstral models are created using Transformer build 5 and use the corresponding tokenizer backends and other configs. Latest vllm only supports Transformer build 4.57, and hence can't load Minstral models directly, unless the mistral backbone is explicitly specified. Since llmcompressor is a vllm package, and uses Transformers 4.57, it can't load the ministral models without modifications to the config.
```angular2html
vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --tokenizer-mode mistral \
    --config-format mistral \
    --load-format mistral \
    --host 0.0.0.0 \
    --port 8002 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --enforce-eager
```

Problem 2: ministral models have a vision backbone which is redundant for text only tasks

Solution
First create two uv environments, one with the latest transformer version (build 5.x), call it .venv-ministral-hf, and another with vllm, that includes the 4.5.x transformer build

Step 1:
Isolate the text-only backbone from the ministral model. 
Run extract_lm_backbone.py in the .venv-ministral-hf environment. This will extract the text backbone and save it as a separate model

This new text only model will now work correctly in the .venv-ministral-hf environment

Step 2:
Patch the configs to make it transformer 4.57 compatible and copy the original tokenizer files (important)
Run patch_ministral_config.py

Step 3:
Test using vllm
```angular2html
vllm serve /home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours   --tokenizer /home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours   --tokenizer-mode mistral   --config-format hf   --load-format auto   --generation-config vllm   --host 0.0.0.0   --port 8002   --max-model-len 4096   --tensor-parallel-size 1   --enforce-eager

```

Note: You would have run the original (full model) like this:
```angular2html
vllm serve /home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16 \
  --host 0.0.0.0 \
  --port 8003 \
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  --language-model-only \
  --max-model-len 4096 \
  --tensor-parallel-size 1 \
  --enforce-eager
```
Test using vllm in python (.venv-ministral environment)
python run-text-only-vllm.py

Test using transformers in python (.venv-ministral-hf environment)
python run-text-only-transformers.py
Note the tokenizer needs to point to the original bf16 directory.. not sure why
The .venv-ministral environment doesn't work with running the model on transformers

Step 4:
Run NVFP4 quantization
```angular2html
python ministral14b_compression.py --method=nvfp4 --targets=Linear --output-dir="/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours-NVFP4"

```
Note: It uses the tokenizer from the BF16 version

Step 5: 
Host the quantized model using vllm
```angular2html
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

vllm serve "/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours-NVFP4" \
  --tokenizer "/home/ankur/dev/models/Ministral-3-14B-Instruct-2512-BF16-TextOnly-Ours-NVFP4" \
  --tokenizer-mode mistral \
  --config-format hf \
  --load-format auto \
  --generation-config vllm \
  --host 0.0.0.0 \
  --port 8003 \
  --max-model-len 4096 \
  --tensor-parallel-size 1 \
  --enforce-eager
```
This approach doesn't work because of incompatibility between the transformers version used 

make a new environment with latest de vllm
```angular2html
uv venv ~/dev/.venv-llmcompressor
source ~/dev/.venv-llmcompressor/bin/activate

uv pip install --upgrade pip
uv pip install torch torchvision torchaudio
uv pip install git+https://github.com/huggingface/transformers.git
uv pip install datasets accelerate sentencepiece protobuf safetensors
uv pip install git+https://github.com/vllm-project/llm-compressor.git
```
may have to uninstall torchvision to eliminate errors
