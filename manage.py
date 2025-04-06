import torch
import torchaudio
import gradio as gr
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cpu")

def tts(text, audio):
    wav, sr = torchaudio.load(audio)
    speaker = model.make_speaker_embedding(wav, sr)
    cond = make_cond_dict(text=text, speaker=speaker, language="en-us")
    conditioning = model.prepare_conditioning(cond)
    codes = model.generate(conditioning)
    wav_out = model.autoencoder.decode(codes).cpu()
    path = "output.wav"
    torchaudio.save(path, wav_out[0], model.autoencoder.sampling_rate)
    return path

gr.Interface(fn=tts,
             inputs=[
                 gr.Textbox(label="Text to Speak"),
                 gr.Audio(source="upload", type="filepath", label="Speaker Audio")
             ],
             outputs=gr.Audio(label="Generated Voice"),
             title="Zonos TTS Voice Cloner"
).launch()