# Bring your own domain audio

The notebooks use NVIDIA's public dummy LibriSpeech sample so every participant
can complete the workflow without credentials.

For a real adaptation run, prepare 16 kHz mono WAV files and a JSON Lines
manifest with one record per file:

```json
{"audio_filepath":"/absolute/path/example.wav","text":"the exact transcript","duration":7.4}
```

Split speakers between train and validation sets. The workshop's tiny fine-tune
is a mechanics exercise, not evidence that a model improved on a domain. Report
accuracy only from a held-out set representative of the target accents, noise,
terminology and recording conditions.
