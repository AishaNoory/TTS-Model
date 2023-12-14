import os
import tarfile
import requests
import shutil
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import load_tts_samples
from trainer import Trainer, TrainerArgs

# Define output_path at the top level of your script
output_path = r"C:/Users/busta/OneDrive/Desktop/TTS"  # Adjust this path as needed
os.makedirs(output_path, exist_ok=True)  # Create the directory if it does not exist

# Define the LJ Speech dataset download URL and destination directory
lj_speech_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
lj_speech_dir = os.path.join(output_path, "LJSpeech-1.1")

# Audio config
audio_config = VitsAudioConfig(
    sample_rate=44100, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
)

# Character config
character_config = CharactersConfig(
    characters_class="TTS.tts.models.vits.VitsCharacters",
    characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890",
    punctuations=" !,.?-",
    pad="<PAD>",
    eos="<EOS>",
    bos="<BOS>",
    blank="<BLNK>",
)

# Model config
config = VitsConfig(
    audio=audio_config,
    characters=character_config,
    run_name="vits_vctk",
    batch_size=16,
    eval_batch_size=4,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=0,
    epochs=1000,
    text_cleaner="basic_cleaners",
    use_phonemes=False,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    save_best_after=1000,
    save_checkpoints=True,
    save_all_best=True,
    mixed_precision=True,
    max_text_len=250,
    output_path=output_path,
    datasets=[
        {
            "formatter": "ljspeech",
            "dataset_name": "LJSpeech",
            "meta_file_train": os.path.abspath(os.path.join(lj_speech_dir, "metadata.csv")),
            "path": os.path.abspath(lj_speech_dir),
            "meta_file_val": os.path.abspath(os.path.join(lj_speech_dir, "metadata.csv")),
            "ignored_speakers": None,
            "language": None,
        }
    ],
    cudnn_benchmark=False,
    test_sentences=[
        ["Ini adalah tes suara"],
        ["Beberapa hal yang perlu diperhatikan ketika melatih sebuah model adalah parameter."],
        ["Selamat pagi"]
    ],
    eval_split_size=0.9
)

# Audio processor is used for feature extraction and audio I/O
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
tokenizer, config = TTSTokenizer.init_from_config(config)

# Formatter function for LJ Speech dataset
def formatter(root_path, meta_file_train, **kwargs):
    items = []
    with open(meta_file_train, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split("|")
        if len(parts) == 3:
            audio_id, transcription, _ = parts
            items.append({
                "text": transcription,
                "audio_file": os.path.join(root_path, "wavs", audio_id + ".wav"),
                "root_path": root_path
            })
    return items

if __name__ == '__main__':
    try:
        # Download and extract the dataset
        response = requests.get(lj_speech_url, stream=True)
        with open(os.path.join(output_path, "LJSpeech-1.1.tar.bz2"), "wb") as tar_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    tar_file.write(chunk)

        with tarfile.open(os.path.join(output_path, "LJSpeech-1.1.tar.bz2"), "r:bz2") as tar:
            tar.extractall(output_path)

        # Load the dataset
        train_samples, eval_samples = load_tts_samples(
            datasets=config.datasets,
            eval_split=True,
            formatter=formatter,
        )

        # Initialize the model
        model = Vits(config, ap, tokenizer, speaker_manager=None)

        # Initialize the trainer
        trainer = Trainer(
            TrainerArgs(),

            config,
            output_path,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
           
        )

        # Start training
        trainer.fit()

       

    except Exception as e:
        print(f"An error occurred: {e}")
        
 #test the model
        (response1, response2) = model.test_run()
        print(response1)
        print(response2)
        trainer.test(model=model, test_samples=["Hello world, I am testing this input."])
        print(trainer.test_run())
        