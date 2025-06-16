git clone https://github.com/SparkAudio/Spark-TTS.git
cd Spark-TTS

conda create -n sparktts -y python=3.12
conda activate sparktts
pip install ipykernel
ipython kernel install --user --name sparktts --display-name "sparktts"

pip install -r requirements.txt


from huggingface_hub import snapshot_download

snapshot_download("SparkAudio/Spark-TTS-0.5B", local_dir="pretrained_models/Spark-TTS-0.5B")


# If you are in mainland China, you can set the mirror as follows:
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com


vim run_tts.py

from gradio_client import Client
from datasets import load_dataset
import os
import shutil

# 初始化 Gradio 客户端
client = Client("https://d94e474df91206b4f4.gradio.live")

# 加载数据集
dataset = load_dataset("svjack/Omni_Ethan_Ad_wav")["train"]

# 设置输出文件夹
output_dir = "male_generated_audio_prompts"
os.makedirs(output_dir, exist_ok=True)

# 遍历数据集并调用 API
for idx, example in enumerate(dataset):
    prompt_text = example["prompt"]
    
    # 调用 Gradio API
    result = client.predict(
        text=prompt_text,
        gender="male",
        pitch=3,
        speed=3,
        api_name="/voice_creation"
    )
    
    # 输出文件名（按字典序命名，例如 prompt_0000.txt 和 prompt_0000.wav）
    output_base = os.path.join(output_dir, f"prompt_{idx:04d}")
    
    # 保存 prompt 到 .txt 文件
    txt_path = f"{output_base}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    
    # 拷贝 WAV 文件到输出目录
    wav_path = result  # 假设 result 返回的是本地文件路径
    if os.path.exists(wav_path):
        shutil.copy(wav_path, f"{output_base}.wav")
        print(f"Saved: {output_base}.txt and {output_base}.wav")
    else:
        print(f"WAV file not found for index {idx}: {wav_path}")

# 加载数据集
dataset = load_dataset("svjack/Mavuika_Ad_Prompt")["train"]

# 设置输出文件夹
output_dir = "female_generated_audio_prompts"
os.makedirs(output_dir, exist_ok=True)

# 遍历数据集并调用 API
for idx, example in enumerate(dataset):
    prompt_text = example["prompt"]
    
    # 调用 Gradio API
    result = client.predict(
        text=prompt_text,
        gender="female",
        pitch=3,
        speed=3,
        api_name="/voice_creation"
    )
    
    # 输出文件名（按字典序命名，例如 prompt_0000.txt 和 prompt_0000.wav）
    output_base = os.path.join(output_dir, f"prompt_{idx:04d}")
    
    # 保存 prompt 到 .txt 文件
    txt_path = f"{output_base}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    
    # 拷贝 WAV 文件到输出目录
    wav_path = result  # 假设 result 返回的是本地文件路径
    if os.path.exists(wav_path):
        shutil.copy(wav_path, f"{output_base}.wav")
        print(f"Saved: {output_base}.txt and {output_base}.wav")
    else:
        print(f"WAV file not found for index {idx}: {wav_path}")
