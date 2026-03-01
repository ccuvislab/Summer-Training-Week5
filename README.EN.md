
# 📑 Outline
### [1. Local Ollama Setup](#i)

### [2. Local LLaVA Setup and Experiment](#ii)
  - [2-1. LLaVA Setup](#ii-1)
  
  - [2-2. LLaVA Experiment Assignment](#ii-2)
### [3. Local LLM Setup and Experiment](#iii)
  - [3-1. LLM Setup](#iii-1)
  
  - [3-2. LLM Experiment Assignment](#iii-2)
### [4. Remote TWCC Ollama Setup and Experiment](#iv)
  - [4-1. Remote TWCC Ollama Setup](#iv-1)
  
  - [4-2. Experiment Assignment](#iv-2)

<a name="i"></a>
## 🔍 1. Local Ollama Setup
Ollama offers the convenience of easy deployment and containerization for LLM inference.

Before installing Ollama, ensure that GPU drivers such as NVIDIA CUDA are already installed.
For graphics card driver installation, you can refer to the following websites:

  - 📝 [Installation Tutorial](https://vocus.cc/article/67015afefd897800016a47e0) 

  - 📝 [PyTorch Official Website](https://pytorch.org/get-started/locally/)

Once the environment issues are resolved, go to the Ollama official website, select the installation package corresponding to your operating system, and install it.

  - 📝 [Ollama Download](https://ollama.com/download)

After installation, you can start using Ollama. (Note: Ollama now offers a native Windows installer, so Docker is no longer a strict requirement for Windows users.)

  - 📝 [Complete Guide to Using Ollama](https://github.com/datawhalechina/handy-ollama/blob/main/docs/C4/2.%20%E5%9C%A8%20Python%20%E4%B8%AD%E4%BD%BF%E7%94%A8%20Ollama%20API.md)

Currently, Ollama can only run LLMs in the `.gguf` format. If you want to convert other models from HuggingFace that do not have a `.gguf` file yet, you can refer to [this article](https://medium.com/playtech/%E4%BD%BF%E7%94%A8llama-cpp%E5%B0%87huggingface-%E5%8F%96%E5%BE%97%E7%9A%84llm%E6%A8%A1%E5%9E%8B%E8%BD%89%E7%82%BA-gguf%E6%A0%BC%E5%BC%8F-879c3bd3505c).

Currently, only supercomputers can run full, uncompressed LLMs. For inference, the vast majority use quantized models (Quantization), which are compressed versions of the models:

  - 📝 [Background Knowledge of Quantized Models 1](https://vocus.cc/article/6803b975fd8978000153e4ad)
  - 📝 [Background Knowledge of Quantized Models 2](https://chih-sheng-huang821.medium.com/ai%E6%A8%A1%E5%9E%8B%E5%A3%93%E7%B8%AE%E6%8A%80%E8%A1%93-%E9%87%8F%E5%8C%96-quantization-966505128365)


<a name="ii"></a>
## 🔍 2. Local LLaVA Setup and Experiment

<a name="ii-1"></a>
### 🚀 2-1. LLaVA Setup

LLaVA is a Vision-Language Model (VLM) that can perform inference using prompts and images.

  - 🤖 [llava-v1.5-7B.gguf](https://huggingface.co/second-state/Llava-v1.5-7B-GGUF/tree/main)

To run a model using Ollama, you need to prepare the model itself and a `Modelfile`.
The `Modelfile` is a file used to create and share models. It contains the generation parameters and is where the main prompt engineering will be implemented.
You only need a notepad to prepare a `Modelfile`; after writing it, simply change the file extension to `.modelfile` (or leave it without an extension as just `Modelfile`).

  - 📝 [Official Modelfile Templates and Examples](https://ollama.readthedocs.io/en/modelfile/)

Once the model itself and the `Modelfile` are prepared, it is best to place them in the same folder (the `FROM` parameter in the `Modelfile` mainly corresponds to the file name and location of the LLM).
Then, use `cmd` or Anaconda Prompt to navigate to that folder and execute:
```
ollama create my-model -f Modelfile
```
`my-model` is your custom LLM name, and `Modelfile` is the name of your prepared Modelfile (**including the file extension**).

>[!WARNING]
> **# Note: If there is an error during creation, it means there might be an error in the Modelfile, mostly formatting issues.**

To check if the model has been successfully built, you can enter the following in `cmd` or Anaconda Prompt:

```
ollama list
```
This command will display the names of all models that have been built and can be run. If the model name created by `ollama create` is in there, the model build was successful.

The corresponding reference code can be found in the `ollama_create.ipynb` notebook in the `llava` folder.

If you want to run the vanilla LLaVA, you can refer to the `modelfile` (`llava1_6_7b_Q4_vanilla.modelfile`) in the `llava` folder.

After all the above preparations are complete, you can refer to the `llava.ipynb` notebook in the `llava` folder to try running it with images and prompts.

<a name="ii-2"></a>
### 📌 2-2. LLaVA Experiment Assignment
> [!IMPORTANT]
> Use your prompt to try to identify the vehicle details (model, color, brand...) from the car photos in the `llava` folder, and take a screenshot of the results.

<a name="iii"></a>
## 🔍 3. Local LLM Setup and Experiment 

<a name="iii-1"></a>
### 🚀 3-1. LLM Setup
Basically, all setup contents are not significantly different from the previous chapter. The difference lies in the use of the Ollama API for VLM versus LLM (for details, refer to the `LLM_ollama.ipynb` notebook in the `LLM` folder).
Unlike the previous experiments, this chapter mainly revolves around Prompt Engineering.

Regarding Prompt Engineering, you can refer to [this guide](https://www.promptingguide.ai/zh):

Broadly speaking, Prompt Engineering is about making the LLM generate the content you desire, which can be achieved through text content you have prepared yourself.

Below is an example experiment of using Prompt Engineering to create a simple chatbot:

  - 📝 [Official Modelfile Templates and Examples](https://ollama.readthedocs.io/en/modelfile/)

> [!TIP]
> Refer to `prompt_example.txt` in the `LLM` folder, which is a Q&A pair generated using GPT. Based on the template above, customize the contents of `system`, `user`, and `assistant`, and adjust the `Temperature` value to achieve your desired effect.

<a name="list1"></a>
You can refer to the following recommended Chinese LLMs to conduct experiments:

  - 🤖 [Qwen1.5-7B-Chat-GGUF](https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GGUF/tree/main)

  - 🤖 [Qwen2-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/tree/main)

  - 🤖 [Yi-1.5-6B-Chat-GGUF](https://huggingface.co/bartowski/Yi-1.5-6B-Chat-GGUF/tree/main)

  - 🤖 [DeepSeek-R1-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main)

  - 🤖 [Chinese-Alpaca-2-7B-GGUF](https://huggingface.co/TheBloke/Chinese-Alpaca-2-7B-GGUF/tree/main)

  - 🤖 [blossom-v3-baichuan2-7B-GGUF](https://huggingface.co/TheBloke/blossom-v3-baichuan2-7B-GGUF/tree/main)

  - 🤖 [Llama-3-Taiwan-8B-Instruct-GGUF](https://huggingface.co/chienweichang/Llama-3-Taiwan-8B-Instruct-GGUF/tree/main)

  - 🤖 [Breeze-7B-Instruct-v1_0-GGUF](https://huggingface.co/YC-Chen/Breeze-7B-Instruct-v1_0-GGUF/tree/main)

  - 🤖 [Gpt-oss-20b](https://huggingface.co/unsloth/gpt-oss-20b-GGUF/tree/main)

  - 🤖 [gemma-7b](https://huggingface.co/google/gemma-7b-GGUF/tree/main)


<a name="iii-2"></a>
### 📌 3-2. LLM Experiment Assignment
> [!IMPORTANT]
> Refer to the `LLM_Experiment.xlsx` spreadsheet in the `LLM` folder. This contains environmental inspection reports and annotations (left and right columns respectively). Try modifying and using the already annotated environmental inspection reports as the text required for Prompt Engineering. 
> You can refer to the `LLM_ollama.ipynb` notebook, pair it with different LLMs ([refer to the recommended list above](#list1)), and input the unannotated data (the last two) to generate their annotations. Take a screenshot of the annotation results once completed.

<a name="iv"></a>
## 🔍 4. Remote TWCC Ollama Setup and Experiment
<a name="iv-1"></a>
### 🚀 4-1. Remote TWCC Ollama Setup
This chapter primarily explains how to use a remote server (using TWCC) to set up Ollama, and use a client to transmit prompts to the server. You then wait for the server to finish processing and send the generated data back to the client.

This approach avoids the issue of insufficient local performance (allowing you to run larger LLMs) and can be deployed to various other client applications at the same time.

> [!TIP]
> For detailed setup operations, refer to [**TWCC_ollama.pdf**](/TWCC_ollama.pdf)

<a name="iv-2"></a>
### 📌 4-2. Experiment Assignment
> [!IMPORTANT]
> Set up Ollama on TWCC and run an LLM. Take screenshots of the server backend and the generation results on the client side.
