# medicalclaw

This repository contains a BioViL-based chest X-ray zero-shot diagnosis workflow.

## Environment

- Recommended Python: 3.12
- Verified environment: CUDA-enabled PyTorch 2.8.0 with `torchvision` 0.23.0

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Running BioViL

Run the main script and provide an absolute path to a chest X-ray image when prompted:

```powershell
python BioViL.py
```

## Transformers compatibility note

This project has been updated to work with newer `transformers` releases.
The text tokenizer path now uses the tokenizer callable API instead of the
legacy `batch_encode_plus` method in `health_multimodal/text/data/io.py`.

This keeps the current prompt tokenization flow compatible with the tokenizer
objects returned by recent Hugging Face model/tokenizer loading code.
