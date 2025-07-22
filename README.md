# Outline
### 1.本地端ollama建置
### 2.本地端llava建置與實驗
### 3.本地端LLM建置與實驗
### 4.遠端TWCC ollama建置與實驗

## 1.本地端ollama建置
ollama在LLM inference上具易部署且容器化的便利性。

安裝ollama之前，要先確保NVIDIA CUDA等GPU驅動已經完成。
安裝顯卡驅動可以參考以下網站:

安裝教學: 
> https://vocus.cc/article/67015afefd897800016a47e0

pytorch官網:
> https://pytorch.org/get-started/locally/

環境問題解決之後，到ollama官網選擇作業環境對應的安裝包並安裝。

ollama下載:
> https://ollama.com/download

安裝完後便可以使用ollama。(需要事先安裝Docker)

ollama使用完整指南:
> https://github.com/datawhalechina/handy-ollama/blob/main/docs/C4/2.%20%E5%9C%A8%20Python%20%E4%B8%AD%E4%BD%BF%E7%94%A8%20Ollama%20API.md

ollama目前只能跑.gguf檔的LLM，若要將huggingface上其他尚未有.gguf檔轉檔，可以參考以下這篇:
> https://medium.com/playtech/%E4%BD%BF%E7%94%A8llama-cpp%E5%B0%87huggingface-%E5%8F%96%E5%BE%97%E7%9A%84llm%E6%A8%A1%E5%9E%8B%E8%BD%89%E7%82%BA-gguf%E6%A0%BC%E5%BC%8F-879c3bd3505c

目前只有超級電腦能跑完整的LLM，在inference上大部分所使用的都是量化模型(Quantization)，也就是壓縮過後的模型:

量化模型的背景知識:
> https://vocus.cc/article/6803b975fd8978000153e4ad

> https://chih-sheng-huang821.medium.com/ai%E6%A8%A1%E5%9E%8B%E5%A3%93%E7%B8%AE%E6%8A%80%E8%A1%93-%E9%87%8F%E5%8C%96-quantization-966505128365





## 2.本地端llava建置與實驗

### 2-1.llava建置

llava為VLM，可以利用prompt與圖片進行推理。

llava-v1.5-7B.gguf :
> https://huggingface.co/second-state/Llava-v1.5-7B-GGUF/tree/main

要使用ollama跑模型，需要準備模型本身以及modelfile。
modelfile為建立和共享模型的文件，包含生成的參數以及主要的提示工程都會在這裡實現。
modelfile的準備僅需要利用記事本，在寫完之後改檔名為modelfile即可。

modelfile的官方模板與範例:
>https://ollama.readthedocs.io/en/modelfile/


模型本身以及modelfile準備好之後，最好是放在同個資料夾(modelfile中的FROM參數主要是對應到LLM的檔案名稱與位置)。
之後用cmd或是anaconda prompt導引至該資料夾並:
```
ollama create my-model -f Modelfile
```
my-model為你自訂的LLM名稱，Modelfile為寫好的modelfile檔名(**包含副檔名**)

**#註:上述為windows用法，linux版不須加副檔名**

**#註:若是create有錯誤，代表modelfile可能有錯，大多是格式錯誤**

若要確認model是否有建置成功，可以用cmd或是anaconda prompt輸入:

```
ollama list
```
此指令會顯示所有已建好並可以跑的模型名稱，若ollama create的模型名已在裡面，代表模型建置成功。

對應參考程式碼在資料夾llava裡的ollama_create筆記本裡。

若想要跑原版llava，可以參考在資料夾llava裡的modelfile。

以上都準備完成後，可以參考資料夾llava裡的llava筆記本嘗試跑圖片與prompt。

### 2-2.llava實驗

#### 實驗說明
利用你的prompt嘗試分辨資料夾llava裡的車種照片的車種(型號、顏色、廠牌...)，並截圖結果。

## 3.本地端LLM建置與實驗













