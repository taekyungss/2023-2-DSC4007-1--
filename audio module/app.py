import torch
import gradio as gr
import time
import os
import tempfile
from pathlib import Path
from datetime import datetime

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.utils import is_flash_attn_2_available
from transformers.pipelines.audio_utils import ffmpeg_read

from openai import OpenAI
from dotenv import load_dotenv

'''

STT

'''
BATCH_SIZE = 16
MAX_AUDIO_MINS = 30  # maximum audio input in minutes

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
use_flash_attention_2 = is_flash_attn_2_available()

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v2", torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
    use_flash_attention_2=use_flash_attention_2
)

if not use_flash_attention_2:
    # use flash attention from pytorch sdpa
    model = model.to_bettertransformer()

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")

model.to(device)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={"language": "ko", "task": "transcribe"},
    return_timestamps=True
)
pipe_forward = pipe._forward


def transcribe(inputs):
    if inputs is None:
        raise gr.Error("입력된 오디오 파일이 없습니다! 요청을 하기 전에 오디오 파일을 녹음하거나 업로드하십시오.")

    with open(inputs, "rb") as f:
        inputs = f.read()

    inputs = ffmpeg_read(inputs, pipe.feature_extractor.sampling_rate)
    audio_length_mins = len(inputs) / pipe.feature_extractor.sampling_rate / 60

    if audio_length_mins > MAX_AUDIO_MINS:
        raise gr.Error(
            f"허락된 최대 오디오 길이는 {MAX_AUDIO_MINS} 분 입니다."
            f"입력된 오디오 길이는 {round(audio_length_mins, 3)} 분 입니다"
        )

    inputs = {"array": inputs, "sampling_rate": pipe.feature_extractor.sampling_rate}

    def _forward_time(*args, **kwargs):
        global runtime
        start_time = time.time()
        result = pipe_forward(*args, **kwargs)
        runtime = time.time() - start_time
        runtime = round(runtime, 2)
        return result

    pipe._forward = _forward_time
    text = pipe(inputs, batch_size=BATCH_SIZE)["text"]

    yield text, runtime

'''

tts

'''
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("API_KEY")

client = OpenAI()
def tts(text, model, voice):
    response = client.audio.speech.create(
        model=model,  # "tts-1","tts-1-hd"
        voice=voice,  # 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'
        input=text,
    )
    '''
    # Create a temp file to save the audio
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_file.write(response.content)

    # Get the file path of the temp file
    temp_file_path = temp_file.name
    print(temp_file_path)
    '''
    speech_file_path = Path(__file__).parent / f"{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.mp3"

    print(speech_file_path)
    response.stream_to_file(speech_file_path)
    return speech_file_path


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.HTML(
            """
                <div style="text-align: center; max-width: 700px; margin: 0 auto;">
                  <div style="display: inline-flex; align-items: center; gap: 0.8rem; font-size: 1.75rem; " >
                    <h1 style="font-weight: 900; margin-bottom: 7px; line-height: normal; style: "> 취약계층을 위한 AI음성챗봇  </h1>
                  </div>
                </div>
            """
        )
        gr.HTML(
            f"""
            <ol> <STT>  openAI: Whisper-large-v2 </ol>
            """
        )
        #STT
        audio = gr.components.Audio(type="filepath", label="음성 입력")
        button = gr.Button("전사하기")
        with gr.Row():
            runtime = gr.components.Textbox(label="Whisper 전사시간(s)")
        with gr.Row():
            transcription = gr.components.Textbox(label="Whisper 전사내용", show_copy_button=True)
        button.click(
            fn=transcribe,
            inputs=audio,
            outputs=[transcription, runtime],
        )
        gr.HTML(
            f"""
                  <ol> <TTS>  openAI: TTS-1 </ol>
                  """
        )
        #TTS
        with gr.Row():
            model = gr.Dropdown(choices=['tts-1', 'tts-1-hd'], label='모델', value='tts-1')
            voice = gr.Dropdown(choices=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'], label='보이스옵션',
                                value='nova')
        #text = gr.Textbox(label="텍스트입력",
                       #   placeholder="텍스트를 입력하고 텍스트 음성 변환 버튼을 누르거나 Enter 키를 누릅니다.")
        btn = gr.Button("음성합성하기")
        output_audio = gr.Audio(label="음성 출력")

        transcription.submit(fn=tts, inputs=[transcription, model, voice], outputs=output_audio, api_name="tts")
        btn.click(fn=tts, inputs=[transcription, model, voice], outputs=output_audio, api_name="tts")

        #샘플
        gr.Markdown("## 음성샘플")
        gr.Examples(
            [["task1_01.wav"], ["task1_02.wav"]],
            audio,
            outputs=[transcription, runtime],
            fn=transcribe,
            cache_examples=False,
        )
    demo.queue(max_size=10).launch(share=True)