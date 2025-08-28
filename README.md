# 📑Outline
### [1.本地端ollama建置](#i)

### [2.本地端llava建置與實驗](#ii)
  - [2-1.llava建置](#ii-1)
  
  - [2-2.llava實驗作業](#ii-2)
### [3.本地端LLM建置與實驗](#iii)
  - [3-1.LLM建置](#iii-1)
  
  - [3-2.LLM實驗作業](#iii-2)
### [4.遠端TWCC ollama建置與實驗](#iv)
  - [4-1.遠端TWCC ollama建置](#iv-1)
  
  - [4-2.實驗作業](#iv-2)

<a name="i"></a>
## 🔍1.本地端ollama建置:  
ollama在LLM inference上具易部署且容器化的便利性。

安裝ollama之前，要先確保NVIDIA CUDA等GPU驅動已經完成。
安裝顯卡驅動可以參考以下網站:

  - 📝[安裝教學:](https://vocus.cc/article/67015afefd897800016a47e0) 


  - 📝[pytorch官網:](https://pytorch.org/get-started/locally/)

環境問題解決之後，到ollama官網選擇作業環境對應的安裝包並安裝。

  - 📝[ollama下載:](https://ollama.com/download)

安裝完後便可以使用ollama。(windows版需要事先安裝Docker)

  - 📝[ollama使用完整指南:](https://github.com/datawhalechina/handy-ollama/blob/main/docs/C4/2.%20%E5%9C%A8%20Python%20%E4%B8%AD%E4%BD%BF%E7%94%A8%20Ollama%20API.md)


ollama目前只能跑.gguf檔的LLM，若要將huggingface上其他尚未有.gguf檔轉檔，可以參考[這篇:](https://medium.com/playtech/%E4%BD%BF%E7%94%A8llama-cpp%E5%B0%87huggingface-%E5%8F%96%E5%BE%97%E7%9A%84llm%E6%A8%A1%E5%9E%8B%E8%BD%89%E7%82%BA-gguf%E6%A0%BC%E5%BC%8F-879c3bd3505c)

目前只有超級電腦能跑完整的LLM，在inference上大部分所使用的都是量化模型(Quantization)，也就是壓縮過後的模型:

  - 📝[量化模型的背景知識1](https://vocus.cc/article/6803b975fd8978000153e4ad)
  - 📝[量化模型的背景知識2](https://chih-sheng-huang821.medium.com/ai%E6%A8%A1%E5%9E%8B%E5%A3%93%E7%B8%AE%E6%8A%80%E8%A1%93-%E9%87%8F%E5%8C%96-quantization-966505128365)




<a name="ii"></a>
## 🔍2.本地端llava建置與實驗

<a name="ii-1"></a>
### 🚀2-1.llava建置

llava為VLM，可以利用prompt與圖片進行推理。

  - 🤖[llava-v1.5-7B.gguf](https://huggingface.co/second-state/Llava-v1.5-7B-GGUF/tree/main)

要使用ollama跑模型，需要準備模型本身以及modelfile。
modelfile為建立和共享模型的文件，包含生成的參數以及主要的提示工程都會在這裡實現。
modelfile的準備僅需要利用記事本，在寫完之後改副檔名為modelfile即可。

  - 📝[modelfile的官方模板與範例](https://ollama.readthedocs.io/en/modelfile/)



模型本身以及modelfile準備好之後，最好是放在同個資料夾(modelfile中的FROM參數主要是對應到LLM的檔案名稱與位置)。
之後用cmd或是anaconda prompt導引至該資料夾並:
```
ollama create my-model -f Modelfile
```
my-model為你自訂的LLM名稱，Modelfile為寫好的modelfile檔名(**包含副檔名**)

>[!WARNING]
> **#註:若是create有錯誤，代表modelfile可能有錯，大多是格式錯誤**

若要確認model是否有建置成功，可以用cmd或是anaconda prompt輸入:

```
ollama list
```
此指令會顯示所有已建好並可以跑的模型名稱，若ollama create的模型名已在裡面，代表模型建置成功。

對應參考程式碼在資料夾llava裡的[ollama_create筆記本](llava/ollama_create.ipynb)裡。

若想要跑原版llava，可以參考在資料夾llava裡的[modelfile](llava/llava1_6_7b_Q4_vanilla.modelfile)。

以上都準備完成後，可以參考資料夾llava裡的[llava筆記本](llava/llava.ipynb)嘗試跑圖片與prompt。

<a name="ii-2"></a>
### 📌2-2.llava實驗作業
> [!IMPORTANT]
> 利用你的prompt嘗試分辨資料夾llava裡的車種照片的車種(型號、顏色、廠牌...)，並截圖結果。

<a name="iii"></a>
## 🔍3.本地端LLM建置與實驗 

<a name="iii-1"></a>
### 🚀3-1.LLM建置
基本上所有建置內容與上述章節沒有太大差別，差別在於使用ollama API時對於VLM與LLM的不同(詳細可參考資料夾LLM裡的[LLM_ollama筆記本](LLM/LLM_ollama.ipynb))
與前面的實驗不同，此章節主要是圍繞在提示工程(Prompt Engineering)上面。

有關提示工程，可以參考以下[這篇:](https://www.promptingguide.ai/zh)


概括而言，提示工程在於讓LLM生成自己想要的內容，其可以透過自己準備好的文本內容來達成。

以下為利用提示工程創造一個簡單聊天機器人的範例實驗:

  - 📝[modelfile的官方模板與範例:](https://ollama.readthedocs.io/en/modelfile/)


> [!TIP]
> 參考資料夾LLM裡的[prompt_example.txt](LLM/prompt_example.txt)，這是利用gpt生成的對答組。根據上述模板，自定義system、user、assistant的內容，調整Temperature的數值達到你想樣的效果。

<a name="list1"></a>
可參考以下推薦的中文LLM來進行實驗:

  - 🤖[Qwen1.5-7B-Chat-GGUF:](https://huggingface.co/Qwen/Qwen1.5-7B-Chat-GGUF/tree/main)


  - 🤖[Qwen2-7B-Instruct-GGUF:](https://huggingface.co/Qwen/Qwen2-7B-Instruct-GGUF/tree/main)


  - 🤖[Yi-1.5-6B-Chat-GGUF](https://huggingface.co/bartowski/Yi-1.5-6B-Chat-GGUF/tree/main)


  - 🤖[DeepSeek-R1-GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main)


  - 🤖[Chinese-Alpaca-2-7B-GGUF](https://huggingface.co/TheBloke/Chinese-Alpaca-2-7B-GGUF/tree/main)


  - 🤖[blossom-v3-baichuan2-7B-GGUF](https://huggingface.co/TheBloke/blossom-v3-baichuan2-7B-GGUF/tree/main)


  - 🤖[Llama-3-Taiwan-8B-Instruct-GGUF](https://huggingface.co/chienweichang/Llama-3-Taiwan-8B-Instruct-GGUF/tree/main)


  - 🤖[Breeze-7B-Instruct-v1_0-GGUF](https://huggingface.co/YC-Chen/Breeze-7B-Instruct-v1_0-GGUF/tree/main)


  - 🤖[Gpt-oss-20b](https://huggingface.co/unsloth/gpt-oss-20b-GGUF/tree/main)


<a name="iii-2"></a>
### 📌3-2.LLM實驗作業
> [!IMPORTANT]
> 參考資料夾LLM裡的[LLM_Experiment試算表](LLM/LLM_Experiment.xlsx)，此為環境稽查報告與標註(分別是左欄與右欄)，嘗試修改與利用已經標註好的環境稽查報告當作提示工程需要的文本，
可參考[LLM_ollama筆記本](LLM/LLM_ollama.ipynb)配合不同的LLM([參考上述推薦list](#list1))並輸入尚未標註好的資料(倒數兩個)生成他們的標註，完成後截圖標註結果。

<a name="iv"></a>
## 🔍4.遠端TWCC ollama建置與實驗
<a name="iv-1"></a>
### 🚀4-1.遠端TWCC ollama建置
本章節主要是講述如何使用遠端伺服器(使用TWCC)建置ollama，並使用客戶端來傳輸prompt給伺服器端，等待伺服器端處理完後將生成的資料傳給客戶端。

這樣就能避免本地端性能不足的問題(可以跑更大的LLM)，同時可以部屬給其他不同的應用端。

> [!TIP]
> 相關設置操作詳細參考[**TWCC_ollama.pdf**](/TWCC_ollama.pdf)

<a name="iv-2"></a>
### 📌4-2.實驗作業
> [!IMPORTANT]
> 在TWCC設置ollama並跑LLM，截圖伺服器端後台與客戶端生成結果。







