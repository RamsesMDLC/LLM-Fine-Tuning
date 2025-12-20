````markdown
# LLM Fine-Tuning (Chat Templates) — SmolLM3 Examples

A small, practical notebook-style project demonstrating how to run **SmolLM3-3B** with **Hugging Face Transformers** in “chat mode” using:
- the high-level `pipeline("text-generation")` API (automatic chat-template handling), and
- `AutoTokenizer.apply_chat_template()` (manual chat template formatting).

The attached source also includes short conceptual notes explaining key model/training terms (decoder-only, GQA, NoPE, YARN, APO, etc.). :contentReference[oaicite:0]{index=0}

---

## What’s in this repo

This project is organized as a step-by-step walkthrough:

- **Part 0 — Model background**
  - Short description of SmolLM3, its long-context support, and training stack. :contentReference[oaicite:1]{index=1}
- **Part 1 — Simple automated chat**
  - Uses `transformers.pipeline` with a list-of-dicts chat format (`role`/`content`) so Transformers handles the chat template automatically. :contentReference[oaicite:2]{index=2}
- **Part 2 — Advanced automated chat**
  - Adds common generation parameters (e.g., `max_new_tokens`, `temperature`, `top_p`, `repetition_penalty`) and demonstrates multi-turn conversation handling. :contentReference[oaicite:3]{index=3}
- **Part 3 — Working with SmolLM3 chat templates in code**
  - Uses `AutoTokenizer.apply_chat_template()` to format a conversation explicitly. :contentReference[oaicite:4]{index=4}

---

## Requirements

- Python 3.10+ (recommended)
- A working PyTorch installation (CPU or GPU)
- Hugging Face Transformers
- Hugging Face Hub (recommended for auth / downloads)

If you run this in **Google Colab**, the code also demonstrates using `google.colab.userdata` for securely retrieving secrets (e.g., tokens). :contentReference[oaicite:5]{index=5}

---

## Installation

### Option A — Local (venv)

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows (PowerShell)

pip install -U pip
pip install torch transformers huggingface_hub
````

### Option B — Google Colab

In Colab, install dependencies in a cell:

```bash
pip install -U transformers huggingface_hub
```

---

## Authentication (Hugging Face)

Some environments/models may require a Hugging Face token (or you may want higher rate limits).

### Colab (using `userdata`)

Store a secret (e.g., `HF_TOKEN`) in Colab, then:

```python
from google.colab import userdata
from huggingface_hub import login

login(userdata.get("HF_TOKEN"))
```

### Local

```bash
huggingface-cli login
```

---

## Usage

### 1) Simple automated chat (Transformers pipeline)

This is the simplest way to chat with SmolLM3-3B. You provide messages as a list of dicts and `pipeline` handles chat formatting.

```python
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    "HuggingFaceTB/SmolLM3-3B",
    device_map="auto"
)

messages = [
    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a CEO"},
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]

out = pipe(messages, max_new_tokens=128, temperature=0.7)

# The pipeline returns a list; each item contains a "generated_text" conversation.
print(out[0]["generated_text"][-1])
```

**Note:** The original notebook shows a `KeyboardInterrupt` during generation when running on CPU (SmolLM3-3B can be slow without a GPU). If you hit this, switch to a GPU runtime, reduce `max_new_tokens`, or use a smaller model. 

---

### 2) Advanced generation configuration + multi-turn chat

You can pass a “generation config” dict into the pipeline call:

```python
generation_config = {
    "max_new_tokens": 200,
    "temperature": 0.8,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}

conversation = [
    {"role": "system", "content": "You are a helpful math tutor."},
    {"role": "user", "content": "Can you help me with calculus?"},
]

# First response
resp = pipe(conversation, **generation_config)
conversation = resp[0]["generated_text"]

# Follow-up turn
conversation.append({"role": "user", "content": "What is a derivative?"})
resp2 = pipe(conversation, **generation_config)

for msg in resp2[0]["generated_text"]:
    print(f"{msg['role']}: {msg['content']}")
```

---

### 3) Manual chat template formatting (Tokenizer)

If you need full control over how the chat prompt is constructed, you can apply the chat template yourself:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

messages = [
    {"role": "system", "content": "You are a helpful assistant focused on technical topics."},
    {"role": "user", "content": "Can you explain what a chat template is?"},
    {"role": "assistant", "content": "A chat template structures conversations between users and AI models..."},
]

formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(formatted)
```

---

## Generation parameters (quick reference)

Common parameters used in the examples: 

* `max_new_tokens`: maximum number of *generated* tokens (recommended over `max_length`)
* `temperature`: randomness; lower is more deterministic
* `do_sample`: enable sampling (vs. greedy decoding)
* `top_p`: nucleus sampling threshold
* `repetition_penalty`: discourages repetitive outputs

---

## Troubleshooting

### Slow / hanging generation on CPU

SmolLM3-3B is large enough that CPU inference can be very slow. If you see long waits or need to interrupt:

* use a GPU runtime (`device_map="auto"` will pick it up),
* reduce `max_new_tokens` (e.g., 32–64),
* reduce batch size / keep prompts short. 

### Out-of-memory (GPU)

* reduce `max_new_tokens`,
* use a smaller model,
* ensure no other large models are loaded in the same session.

---

## Project goals / non-goals

### Goals

* Show how to run SmolLM3-3B in chat mode using Transformers.
* Demonstrate both automatic and manual chat-template handling.
* Provide a minimal, educational baseline for experimentation. 
```
```
