from flask import Flask, request, render_template
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import torchaudio.transforms as T
from IPython.display import Audio
import torch
import scipy

app = Flask(__name__)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for generating music based on text input
@app.route('/generate_music', methods=['POST'])
def generate_music():
    text_input = request.form['text_input']

    # Load the MusicGen model
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    # Move the model to GPU if available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Unconditional Generation
    unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
    audio_values_unconditional = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=1024)

    # Save the generated unconditional audio as a .wav file
    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write("musicgen_out_unconditional.wav", rate=sampling_rate, data=audio_values_unconditional[0, 0].cpu().numpy())

    # Text-Conditional Generation
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

    # Process the text input
    inputs = processor(
        text=[text_input],
        padding=True,
        return_tensors="pt",
    )

    # Generate audio based on the text input
    audio_values_conditional = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=1024)

    # Save the generated conditional audio as a .wav file
    scipy.io.wavfile.write("musicgen_out_conditional.wav", rate=sampling_rate, data=audio_values_conditional[0, 0].cpu().numpy())

    return render_template('result.html', audio_path="musicgen_out_conditional.wav")

if __name__ == '__main__':
    app.run(debug=False)
