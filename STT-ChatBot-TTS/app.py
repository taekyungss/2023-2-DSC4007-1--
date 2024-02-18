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

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# llm
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
import pandas as pd

import time
import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("API_KEY")

'''---------------------- STT ----------------------'''

BATCH_SIZE = 16
MAX_AUDIO_MINS = 30  # maximum audio input in minutes

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
use_flash_attention_2 = is_flash_attn_2_available()

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3", torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
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
    #if inputs is None:
        #raise gr.Error("입력된 오디오 파일이 없습니다! 요청을 하기 전에 오디오 파일을 녹음하거나 업로드하십시오.")

    with open(inputs, "rb") as f:
        inputs = f.read()

    print(inputs)
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

    return text, runtime



'''---------------------- LLM ----------------------'''

data = pd.read_csv("./data.csv")
chat_template = f"""필수 : 나는 복지상담을 진행하는 음성봇 이며 이름은 아이이다.
한번의 답변에 한가지 질문만 한다. 
공손하고 예의바르게 말을 해야한다. 
전화통화를 하는 것처럼 대화를 해야한다. 
답변은 2줄이내로 진행한다. 
해당 전화는 본 챗봇이 먼저 전화를 건 상황이다.
총 6번이상의 대화가 오고가도록 한다

0. 상대방의 이름은 {data["이름"][1]}이다. 해당 이름으로 부르며 대화를 한다.

1. 첫번째 답변은 안녕하세요 000님, 지금 현재 {data["문제"][1]}로 확인이 되는데 어려움은 없는지 질문한다.

2. {data["문제"][1]}이 건강문제라면 어디 아프신곳이 있냐고 질문한다.

3. {data["문제"][1]}에 대해 구체적인 상황은 어떠한지 질문을 한다.

4. 상대방의 문제에 대해 어느정도 질문을 했으면 주변에 도움을 요청할만한 보호자가 없는지 물어본다.

5. 질문이 최소 5번 이상 오가고 난 이후에, 상대방의 어려움에 관련된 복지 서비스를 추천해준다.


"""

chat_ai  = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'], temperature=0.5, model="gpt-4-1106-preview")

def response(message, history):  ## human, ai 리스트 반환으로 지정
    # 종료하겠습니다를 입력하면 최종 요약 출력
    if message == "종료하겠습니다":
        return summarize(history)

    history_langchain_format = []
    # 프롬프트 추가
    history_langchain_format.append(SystemMessage(content=chat_template))
    # 이전 대화 기억
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))
    print(history_langchain_format)
    gpt_response = chat_ai(history_langchain_format)
    return gpt_response.content, [[message,gpt_response.content]]


# 요약 llm 모델 추가
# summarize_llm과 history로 출력
def summarize(history):
    for i, (human, ai) in enumerate(history):
        history[i] = (f"사용자 : {human}", f"AI : {ai}")

    summarize_llm = OpenAI(
        temperature=0.3, model="gpt-3.5-turbo-instruct", max_tokens=512
    )


    summarize_template = """
    필수 : AI와 사용자 모두의 대화를 합쳐서 요약할거야.
    0. 사용자의 문제상황이 뭔지 정확하게 파악하고 요약해야해.

    {texts} 대화내용을 기반으로 요약문 작성해줘.
    """

    summarize_prompt = PromptTemplate(
        template=summarize_template, input_variables=["texts"]
    )
    summarize_chain = LLMChain(prompt=summarize_prompt, llm=summarize_llm)

    return summarize_chain.run(history)


'''---------------------- TTS ----------------------'''

client = OpenAI()


def tts(text, model, voice):
    response = client.audio.speech.create(
        model=model,  # "tts-1","tts-1-hd"
        voice=voice,  # 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'
        input=text,
    )


    speech_file_path = Path(__file__).parent / f"{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.mp3"

    print(speech_file_path)
    start_time = time.time()
    response.stream_to_file(speech_file_path)
    runtime = time.time() - start_time

    runtime = round(runtime, 2)

    return speech_file_path, runtime


if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Column():
            gr.HTML(
                """
                    <div style="text-align: center; max-width: 700px; margin: 0 auto;">
                      <div style="display: inline-flex; align-items: center; gap: 0.8rem; font-size: 1.75rem; " >
                        <h1 style="font-weight: 900; margin-bottom: 7px; line-height: normal; style: "> 취약계층을 위한 AI음성챗봇  </h1>
                      </div>
                    </div>
                """
            )

                #STT
        with gr.Column(variant='compact'):
            gr.HTML(
                """
                    <div style="text-align: start; max-width: 1450px; margin: 0 auto;">
                      <div style="display: inline-flex; align-items: center; gap: 0.8rem; font-size: 1.75rem; " >
                        <h2 style="font-weight: 900; line-height: normal; style: "> Whisper-large-v3   </h2>
                      </div>
                    </div>
                """
            )
            with gr.Row():

                audio = gr.components.Audio(type="filepath", label="음성 입력", sources=["microphone"])
                transcription = gr.components.Textbox(label="Whisper 전사내용", show_copy_button=True,visible=False)
                runtime = gr.components.Textbox(visible=False, label="Whisper 전사시간(s)")
                audio.change(
                            fn=transcribe,
                            inputs=audio,
                            outputs=[transcription, runtime],
                        )

        with gr.Column(variant='compact'):
            gr.HTML(
                """
                    <div style="text-align: start;max-width: 1450px; margin: 0 auto;">
                      <div style="display: inline-flex; align-items: center; gap: 0.8rem; font-size: 1.75rem; " >
                        <h2 style="font-weight: 900; line-height: normal; style: "> ChatGPT-4-Turbo   </h2>
                      </div>
                    </div>
                """
            )
            with gr.Row():

                chatbot = gr.Chatbot(height=240)
                chatbot_text = gr.components.Textbox(visible=False)
                transcription.change(response, inputs=[transcription,chatbot], outputs=[chatbot_text, chatbot])

        with gr.Column(variant='compact'):
            gr.HTML(
                """
                    <div style="text-align: start; max-width: 1450px; margin: 0 auto;">
                      <div style="display: inline-flex; align-items: center; gap: 0.8rem; font-size: 1.75rem; " >
                        <h2 style="font-weight: 900; line-height: normal; style: "> TTS-1   </h2>
                      </div>
                    </div>
                """
            )
            with gr.Row():

                model = gr.Dropdown(choices=['tts-1', 'tts-1-hd'], label='모델', value='tts-1')
                voice = gr.Dropdown(choices=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'], label='보이스옵션',value='nova')
                output_audio = gr.Audio(label="음성 출력", autoplay=True)
                chatbot_text.change(fn=tts, inputs=[chatbot_text, model, voice], outputs=[output_audio,runtime])

    demo.queue(max_size=10).launch()