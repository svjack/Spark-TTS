```ps
BBdown BV1tyRhY1Ew8 --skip-ai false
BBdown BV1tyRhY1Ew8 --sub-only --skip-ai false
```

```python
from pydub import AudioSegment
import os
import re
from datetime import timedelta

def process_audio_with_srt(wav_path, srt_path, output_folder, max_duration_sec=10):
    """
    根据SRT字幕分割WAV音频文件，并合并相邻片段
    参数:
        wav_path: 输入WAV文件路径
        srt_path: 输入SRT字幕文件路径
        output_folder: 输出文件夹路径
        max_duration_sec: 最大合并时长(秒)
    返回:
        无，直接生成分割后的WAV和TXT文件
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 解析SRT文件
    def parse_srt(srt_content):
        pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n([\s\S]*?)\n\n'
        matches = re.findall(pattern, srt_content + '\n\n')
        segments = []
        for match in matches:
            idx, start, end, text = match
            # 清理文本中的多余空格和换行
            text = ' '.join(text.strip().splitlines())
            segments.append({
                'index': int(idx),
                'start': start.replace(',', '.'),
                'end': end.replace(',', '.'),
                'text': text
            })
        return segments

    # 时间字符串转毫秒
    def time_str_to_ms(time_str):
        h, m, s = time_str.split(':')
        s, ms = s.split('.')
        return int(timedelta(
            hours=int(h),
            minutes=int(m),
            seconds=int(s),
            milliseconds=int(ms)
        ).total_seconds() * 1000)

    # 读取并解析SRT文件
    with open(srt_path, 'r', encoding='utf-8') as f:
        srt_content = f.read()
    segments = parse_srt(srt_content)

    # 加载音频文件
    audio = AudioSegment.from_wav(wav_path)

    # 分割并合并相邻片段
    merged_segments = []
    current_group = []
    current_duration = 0

    for seg in segments:
        start_ms = time_str_to_ms(seg['start'])
        end_ms = time_str_to_ms(seg['end'])
        duration_ms = end_ms - start_ms
        
        # 检查是否需要新建合并组
        if current_duration + duration_ms > max_duration_sec * 1000 and current_group:
            merged_segments.append(current_group)
            current_group = [seg]
            current_duration = duration_ms
        else:
            current_group.append(seg)
            current_duration += duration_ms
    
    if current_group:
        merged_segments.append(current_group)

    # 生成输出文件
    for i, group in enumerate(merged_segments):
        # 计算合并片段的起止时间
        start_ms = time_str_to_ms(group[0]['start'])
        end_ms = time_str_to_ms(group[-1]['end'])
        
        # 提取音频片段
        segment_audio = audio[start_ms:end_ms]
        
        # 合并文本
        merged_text = '\n'.join([seg['text'] for seg in group])
        
        # 生成文件名（按字典序）
        filename = f"segment_{i:04d}"
        wav_output = os.path.join(output_folder, f"{filename}.wav")
        txt_output = os.path.join(output_folder, f"{filename}.txt")
        
        # 保存文件
        segment_audio.export(wav_output, format="wav")
        with open(txt_output, 'w', encoding='utf-8') as f:
            f.write(merged_text)
    
    print(f"处理完成! 共生成 {len(merged_segments)} 个合并片段")

# 使用示例
process_audio_with_srt(
    wav_path="Wang_Leehom_Music_Class_Wav/[P01]1-1呼吸与声音的健康.wav",
    srt_path="Wang_Leehom_Music_Class_Wav/[P01]1-1呼吸与声音的健康.srt",
    output_folder="output_segments",
    max_duration_sec=15  # 合并最长15秒的相邻片段
)
```
