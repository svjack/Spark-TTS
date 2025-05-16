#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spark TTS Fine-tuning Script (0.5B model)
Automatically generated from Colab notebook
Original notebook: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Spark_TTS_(0_5B).ipynb
"""

'''
pip install unsloth
pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf 'datasets>=3.4.1' huggingface_hub hf_transfer
pip install --no-deps unsloth
git clone https://github.com/SparkAudio/Spark-TTS
pip install omegaconf einx

pip uninstall torch torchaudio torchvision -y
pip install torch torchaudio torchvision
pip install tf-keras
pip install soundfile soxr einops librosa
'''

import os
import sys
import re
import locale
import torch
import numpy as np
from typing import Dict, Any
import torchaudio.transforms as T
from datasets import load_dataset
from trl import SFTTrainer
#from transformers import TrainingArguments
#from huggingface_hub import snapshot_download
#from unsloth import FastModel, is_bfloat16_supported
import soundfile as sf
from IPython.display import Audio, display

from unsloth import FastLanguageModel, is_bfloat16_supported, FastModel
from transformers import TrainingArguments
from trl import SFTTrainer

# Setup audio tokenizer
sys.path.append('Spark-TTS')
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.audio import audio_volume_normalize

from huggingface_hub import snapshot_download

def install_dependencies():
    """Install required dependencies"""
    # Commented out IPython magic to ensure Python compatibility.
    # %%capture
    if "COLAB_" not in "".join(os.environ.keys()):
        os.system("pip install unsloth")
    else:
        # Do this only in Colab notebooks! Otherwise use pip install unsloth
        os.system("pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo")
        os.system("pip install sentencepiece protobuf 'datasets>=3.4.1' huggingface_hub hf_transfer")
        os.system("pip install --no-deps unsloth")

    # Extras for Spark TTS
    os.system("git clone https://github.com/SparkAudio/Spark-TTS")
    os.system("pip install omegaconf einx")


class SparkTTSFineTuner:
    def __init__(self, max_seq_length=2048):
        self.max_seq_length = max_seq_length
        self.model = None
        self.tokenizer = None
        self.audio_tokenizer = None

    def setup_models(self):
        """Download and setup the model and tokenizer"""
        # Download model and code
        snapshot_download("unsloth/Spark-TTS-0.5B", local_dir="Spark-TTS-0.5B")

        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name="Spark-TTS-0.5B/LLM",
            max_seq_length=self.max_seq_length,
            dtype=torch.float32,  # Spark seems to only work on float32 for now
            full_finetuning=True,  # We support full finetuning now!
            load_in_4bit=False,
            # token="hf_...",  # use one if using gated models like meta-llama/Llama-2-7b-hf
        )

        # Add LoRA adapters
        self.model = FastModel.get_peft_model(
            self.model,
            r=128,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
            lora_alpha=128,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",     # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )

        # Setup audio tokenizer
        sys.path.append('Spark-TTS')
        from sparktts.models.audio_tokenizer import BiCodecTokenizer
        from sparktts.utils.audio import audio_volume_normalize
        self.audio_tokenizer = BiCodecTokenizer("Spark-TTS-0.5B", "cuda")

    def extract_wav2vec2_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """Extract wav2vec2 features"""
        if wavs.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, but got shape {wavs.shape}")

        wav_np = wavs.squeeze(0).cpu().numpy()
        processed = self.audio_tokenizer.processor(
            wav_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = processed.input_values
        input_values = input_values.to(self.audio_tokenizer.feature_extractor.device)

        model_output = self.audio_tokenizer.feature_extractor(input_values)

        if model_output.hidden_states is None:
            raise ValueError("Wav2Vec2Model did not return hidden states. Ensure config `output_hidden_states=True`.")

        num_layers = len(model_output.hidden_states)
        required_layers = [11, 14, 16]
        if any(l >= num_layers for l in required_layers):
            raise IndexError(f"Requested hidden state indices {required_layers} out of range for model with {num_layers} layers.")

        feats_mix = (
            model_output.hidden_states[11] + model_output.hidden_states[14] + model_output.hidden_states[16]
        ) / 3

        return feats_mix

    def formatting_audio_func(self, example):
        """Format audio data for training"""
        text = f"{example['source']}: {example['text']}" if "source" in example else example["text"]
        audio_array = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]

        target_sr = self.audio_tokenizer.config['sample_rate']

        if sampling_rate != target_sr:
            resampler = T.Resample(orig_freq=sampling_rate, new_freq=target_sr)
            audio_tensor_temp = torch.from_numpy(audio_array).float()
            audio_array = resampler(audio_tensor_temp).numpy()

        if self.audio_tokenizer.config["volume_normalize"]:
            audio_array = audio_volume_normalize(audio_array)

        ref_wav_np = self.audio_tokenizer.get_ref_clip(audio_array)

        audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).float().to(self.audio_tokenizer.device)
        ref_wav_tensor = torch.from_numpy(ref_wav_np).unsqueeze(0).float().to(self.audio_tokenizer.device)

        feat = self.extract_wav2vec2_features(audio_tensor)

        batch = {
            "wav": audio_tensor,
            "ref_wav": ref_wav_tensor,
            "feat": feat.to(self.audio_tokenizer.device),
        }

        semantic_token_ids, global_token_ids = self.audio_tokenizer.model.tokenize(batch)

        global_tokens = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze().cpu().numpy()]  # Squeeze batch dim
        )
        semantic_tokens = "".join(
            [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze().cpu().numpy()]  # Squeeze batch dim
        )

        inputs = [
            "<|task_tts|>",
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>",
            global_tokens,
            "<|end_global_token|>",
            "<|start_semantic_token|>",
            semantic_tokens,
            "<|end_semantic_token|>",
            "<|im_end|>"
        ]
        inputs = "".join(inputs)
        return {"text": inputs}

    def prepare_dataset(self):
        """Prepare the dataset for training"""
        dataset = load_dataset("svjack/genshin_impact_ganyu_audio_sample", split="train").rename_column('prompt', 'text')
        dataset = dataset.map(self.formatting_audio_func, remove_columns=["audio"])
        print("Moving Bicodec model and Wav2Vec2Model to cpu.")
        self.audio_tokenizer.model.cpu()
        self.audio_tokenizer.feature_extractor.cpu()
        torch.cuda.empty_cache()
        return dataset

    def train(self):
        """Train the model"""
        dataset = self.prepare_dataset()

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                # num_train_epochs=1,  # Set this for 1 full training run.
                # max_steps=60,
                max_steps=300,
                learning_rate=2e-4,
                fp16=False,  # We're doing full float32 so disable mixed precision
                bf16=False,  # We're doing full float32 so disable mixed precision
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none",  # Use this for WandB etc
            ),
        )

        # Show current memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        trainer_stats = trainer.train()

        # Show final memory and time stats
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    def generate_speech_from_text(
        self,
        text: str,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1,
        max_new_audio_tokens: int = 2048,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> np.ndarray:
        """
        Generates speech audio from text using default voice control parameters.

        Args:
            text (str): The text input to be converted to speech.
            temperature (float): Sampling temperature for generation.
            top_k (int): Top-k sampling parameter.
            top_p (float): Top-p (nucleus) sampling parameter.
            max_new_audio_tokens (int): Max number of new tokens to generate (limits audio length).
            device (torch.device): Device to run inference on.

        Returns:
            np.ndarray: Generated waveform as a NumPy array.
        """
        FastModel.for_inference(self.model)  # Enable native 2x faster inference

        prompt = "".join([
            "<|task_tts|>",
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>"
        ])

        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)

        print("Generating token sequence...")
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_audio_tokens,  # Limit generation length
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,  # Stop token
            pad_token_id=self.tokenizer.pad_token_id  # Use models pad token id
        )
        print("Token sequence generated.")

        generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]

        predicts_text = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]
        # print(f"\nGenerated Text (for parsing):\n{predicts_text}\n") # Debugging

        # Extract semantic token IDs using regex
        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
        if not semantic_matches:
            print("Warning: No semantic tokens found in the generated output.")
            return np.array([], dtype=np.float32)

        pred_semantic_ids = torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)  # Add batch dim

        # Extract global token IDs using regex
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)
        if not global_matches:
            print("Warning: No global tokens found in the generated output (controllable mode). Might use defaults or fail.")
            pred_global_ids = torch.zeros((1, 1), dtype=torch.long)
        else:
            pred_global_ids = torch.tensor([int(token) for token in global_matches]).long().unsqueeze(0)  # Add batch dim

        pred_global_ids = pred_global_ids.unsqueeze(0)  # Shape becomes (1, 1, N_global)

        print(f"Found {pred_semantic_ids.shape[1]} semantic tokens.")
        print(f"Found {pred_global_ids.shape[2]} global tokens.")

        # Detokenize using BiCodecTokenizer
        print("Detokenizing audio tokens...")
        # Ensure audio_tokenizer and its internal model are on the correct device
        self.audio_tokenizer.device = device
        self.audio_tokenizer.model.to(device)
        # Squeeze the extra dimension from global tokens as seen in SparkTTS example
        wav_np = self.audio_tokenizer.detokenize(
            pred_global_ids.to(device).squeeze(0),  # Shape (1, N_global)
            pred_semantic_ids.to(device)            # Shape (1, N_semantic)
        )
        print("Detokenization complete.")

        return wav_np

    def save_model(self, output_dir="lora_model"):
        """Save the trained model"""
        self.model.save_pretrained(output_dir)  # Local saving
        self.tokenizer.save_pretrained(output_dir)
        # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
        # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

    def save_merged_model(self, output_dir="model", save_method="merged_16bit", hf_repo=None, token=None):
        """Save merged model (16bit or 4bit)"""
        if save_method == "merged_16bit":
            self.model.save_pretrained_merged(output_dir, self.tokenizer, save_method="merged_16bit")
            if hf_repo:
                self.model.push_to_hub_merged(hf_repo, self.tokenizer, save_method="merged_16bit", token=token)
        elif save_method == "merged_4bit":
            self.model.save_pretrained_merged(output_dir, self.tokenizer, save_method="merged_4bit")
            if hf_repo:
                self.model.push_to_hub_merged(hf_repo, self.tokenizer, save_method="merged_4bit", token=token)
        elif save_method == "lora":
            self.model.save_pretrained_merged(output_dir, self.tokenizer, save_method="lora")
            if hf_repo:
                self.model.push_to_hub_merged(hf_repo, self.tokenizer, save_method="lora", token=token)


def main():
    # Install dependencies
    #install_dependencies()

    # Initialize and setup the fine-tuner
    fine_tuner = SparkTTSFineTuner(max_seq_length=2048)
    fine_tuner.setup_models()

    # Train the model
    fine_tuner.train()

    # Generate speech example
    input_text = "愿帝君保佑你，愿你的每个梦都踏实而香甜。"
    chosen_voice = None  # None for single-speaker

    print(f"Generating speech for: '{input_text}'")
    text = f"{chosen_voice}: " + input_text if chosen_voice else input_text
    generated_waveform = fine_tuner.generate_speech_from_text(input_text)

    if generated_waveform.size > 0:
        output_filename = "generated_speech_controllable.wav"
        sample_rate = fine_tuner.audio_tokenizer.config.get("sample_rate", 16000)
        sf.write(output_filename, generated_waveform, sample_rate)
        print(f"Audio saved to {output_filename}")

        # Optional: Play audio
        display(Audio(generated_waveform, rate=sample_rate))
    else:
        print("Audio generation failed (no tokens found?).")

    # Save the model
    fine_tuner.save_model(output_dir = "lora_model_merged_300")
    #fine_tuner.save_merged_model(save_method="merged_16bit")


if __name__ == "__main__":
    main()

'''
sudo apt-get update && sudo apt-get install cbm ffmpeg git-lfs

pip install unsloth
pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf 'datasets>=3.4.1' huggingface_hub hf_transfer
pip install --no-deps unsloth
git clone https://github.com/SparkAudio/Spark-TTS
pip install omegaconf einx

pip uninstall torch torchaudio torchvision -y
pip install torch torchaudio torchvision
pip install tf-keras
pip install soundfile soxr einops librosa

git clone https://huggingface.co/svjack/Spark-TTS-0.5B-GanYu-Merged-Early
git clone https://huggingface.co/unsloth/Spark-TTS-0.5B
'''


import sys
sys.path.append('Spark-TTS')

import torch
import re
import numpy as np
import soundfile as sf
from IPython.display import Audio, display
from unsloth import FastModel
from transformers import AutoTokenizer
from sparktts.models.audio_tokenizer import BiCodecTokenizer

class SparkTTSLoRAInference:
    def __init__(self, model_name="lora_model_merged_300/"):
        """初始化模型和tokenizer"""
        # 加载基础模型和LoRA适配器
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=torch.float32,
            load_in_4bit=False,
        )
        #self.model.load_adapter(lora_path)  # 加载LoRA权重

        # 初始化音频tokenizer
        self.audio_tokenizer = BiCodecTokenizer("Spark-TTS-0.5B", "cuda")
        FastModel.for_inference(self.model)  # 启用优化推理模式

        # 打印设备信息
        print(f"Model loaded on device: {next(self.model.parameters()).device}")

    def generate_speech_from_text(
            self,
            text: str,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 1,
            max_new_audio_tokens: int = 2048,
            device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ) -> np.ndarray:
        """
        Generates speech audio from text using default voice control parameters.
        Args:
            text (str): The text input to be converted to speech.
            temperature (float): Sampling temperature for generation.
            top_k (int): Top-k sampling parameter.
            top_p (float): Top-p (nucleus) sampling parameter.
            max_new_audio_tokens (int): Max number of new tokens to generate (limits audio length).
            device (torch.device): Device to run inference on.
        Returns:
            np.ndarray: Generated waveform as a NumPy array.
        """
        FastModel.for_inference(self.model)  # Enable native 2x faster inference
        prompt = "".join([
            "<|task_tts|>",
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_global_token|>"
        ])
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(device)
        print("Generating token sequence...")
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_audio_tokens,  # Limit generation length
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,  # Stop token
            pad_token_id=self.tokenizer.pad_token_id  # Use models pad token id
        )
        print("Token sequence generated.")
        generated_ids_trimmed = generated_ids[:, model_inputs.input_ids.shape[1]:]
        predicts_text = self.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=False)[0]
        # print(f"\nGenerated Text (for parsing):\n{predicts_text}\n") # Debugging
        # Extract semantic token IDs using regex
        semantic_matches = re.findall(r"<\|bicodec_semantic_(\d+)\|>", predicts_text)
        if not semantic_matches:
            print("Warning: No semantic tokens found in the generated output.")
            return np.array([], dtype=np.float32)
        pred_semantic_ids = torch.tensor([int(token) for token in semantic_matches]).long().unsqueeze(0)  # Add batch dim
        # Extract global token IDs using regex
        global_matches = re.findall(r"<\|bicodec_global_(\d+)\|>", predicts_text)
        if not global_matches:
            print("Warning: No global tokens found in the generated output (controllable mode). Might use defaults or fail.")
            pred_global_ids = torch.zeros((1, 1), dtype=torch.long)
        else:
            pred_global_ids = torch.tensor([int(token) for token in global_matches]).long().unsqueeze(0)  # Add batch dim
        pred_global_ids = pred_global_ids.unsqueeze(0)  # Shape becomes (1, 1, N_global)
        print(f"Found {pred_semantic_ids.shape[1]} semantic tokens.")
        print(f"Found {pred_global_ids.shape[2]} global tokens.")
        # Detokenize using BiCodecTokenizer
        print("Detokenizing audio tokens...")
        # Ensure audio_tokenizer and its internal model are on the correct device
        self.audio_tokenizer.device = device
        self.audio_tokenizer.model.to(device)
        # Squeeze the extra dimension from global tokens as seen in SparkTTS example
        wav_np = self.audio_tokenizer.detokenize(
            pred_global_ids.to(device).squeeze(0),  # Shape (1, N_global)
            pred_semantic_ids.to(device)            # Shape (1, N_semantic)
        )
        print("Detokenization complete.")
        return wav_np

tts = SparkTTSLoRAInference("Spark-TTS-0.5B-GanYu-Merged-Early")

generated_waveform = tts.generate_speech_from_text("愿帝君保佑你，愿你的每个梦都踏实而香甜。", max_new_audio_tokens = 2048)
if generated_waveform.size > 0:
    output_filename = "infer1.wav"
    sample_rate = tts.audio_tokenizer.config.get("sample_rate", 16000)
    sf.write(output_filename, generated_waveform, sample_rate)
    print(f"Audio saved to {output_filename}")
    # Optional: Play audio
    display(Audio(generated_waveform, rate=sample_rate))

generated_waveform = tts.generate_speech_from_text("天叔大人长年为民操劳的智慧，凝光大人运筹帷幄的远见，都是璃月繁荣的基石。能与二位共事是我的荣幸，愿这份同心协力的契约精神永远守护璃月港的万家灯火。", max_new_audio_tokens = 2048)
if generated_waveform.size > 0:
    output_filename = "infer2.wav"
    sample_rate = tts.audio_tokenizer.config.get("sample_rate", 16000)
    sf.write(output_filename, generated_waveform, sample_rate)
    print(f"Audio saved to {output_filename}")
    # Optional: Play audio
    display(Audio(generated_waveform, rate=sample_rate))

generated_waveform = tts.generate_speech_from_text("芙宁娜大人，冒昧打扰。这封来自璃月七星的文书，还请您过目。枫丹与璃月的合作事宜，凝光大人特意嘱咐我当面转达谢意。", max_new_audio_tokens = 2048)
if generated_waveform.size > 0:
    output_filename = "infer3.wav"
    sample_rate = tts.audio_tokenizer.config.get("sample_rate", 16000)
    sf.write(output_filename, generated_waveform, sample_rate)
    print(f"Audio saved to {output_filename}")
    # Optional: Play audio
    display(Audio(generated_waveform, rate=sample_rate))
