# Minibob: Teaching neural network to play Alias

This repo contains source code and notes for the prototype bot to play word game Alias with. There are Python scripts for data parsing, ML model training and building a public app.

---

## Intro

The goal of this repo to build a telegram bot that capable guess word user described. It simulates word game Alias for two players but only from one side, so user describes and bot guess an answer. Another constrain is language, we will teach Russian bot.

> Why Russian if article in English?

I am just developing my writing skills =) 

One possible solution is deep neural networks to generate an answer on user word description (next prompt). We could use pretrained models like [GPT](https://jalammar.github.io/illustrated-gpt2/) or [T5](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) models which are widely used for text generation tasks. We will use T5, there are several models for Russian.

The next step is domain data because we have to finetune (teach) our model for generation on specific prompt. 

```bash
"известный русский напиток" -> T5 (magic) -> "водка"
```

There are several word games where the user is required to solve word puzzles. They could be in the format of a crossword or just plain question-answer pair. Some sites also store keys for games, so we could parse pairs to train our model. One important thing is questions are usually much harder than prompts we use for Alias game. It's expensive to collect human annotations for Alias, so it could be good proxy data for our purpose. 

Last step is to develop some interface for user interaction. We will use telegram as a poor man tool.

Finally, let's create a name for our bot, let it be "`minibob`", the mini part stands for light version and Bob (Alice companion) well known char in computer science.

> "Alice and Bob are fictional characters commonly used as placeholders in discussions about cryptographic systems and protocols, ..." - Wikipedia

Other sections describe how to reproduce app and give additional notes. 

## Collect word games dataset for russian language

You can skip this section, ready to use [dataset](https://huggingface.co/datasets/artemsnegirev/ru-word-games) you can find on 🤗 hub:

```python
from datasets import load_dataset

dataset = load_dataset("artemsnegirev/ru-word-games")
```

### How to

To scrape data from scratch use `parsing` module:

1. parsers use selenium lib, so you have to [download](https://chromedriver.chromium.org/downloads) chromedriver for your browser version and place it at path `./chromedriver/chromedriver`
2. install required packages

    ```
    python -m venv venv
    source venv/bin/activate
    pip install -r parsing/requirements.txt
    ```

3. run parsers
    ```python
    python parsing/main.py
    ```

You could face some problems during parsing. If it failed, don't worry, progress is saved and you can start script as usual to continue.

You could also add some new parsers and extend functionality, it will be awesome!

### Additional notes

I found a great website with answers for most popular Russian word games, every game's page has similar HTML structure. I implemented BaseParser for specific Parser, its output is DataRecord:

```python
@dataclass
class DataRecord:
    prompt: str
    answer: str
```

Output is standartized, and the answer is lower case. I also added subset field (game name) for filtering some examples in future.

Some exampels:

```json
{"subset": "350_zagadok", "answer": "баян", "prompt": "Давно небритый анекдот."}
{"subset": "bashnya_slov", "answer": "бегемот", "prompt": "Громкие животные"}
{"subset": "crosswords", "answer": "аббат", "prompt": "Начальник мoнаxoв-катoликoв"}
```

That is we need to teach our minibob do staff!

Follow this scripts for more details:
 - parsing/main.py (parsing loop)
 - parsing/parsers.py (implementation of parsers)

### Dataset info

Dataset contains more than 100k examples of pairs word-description, where description is kind of crossword question. 

Key stats:
- Number of examples: 133223
- Number of sources: 8
- Number of unique answers: 35024

| subset       | count |
|--------------|-------|
| 350_zagadok  | 350   |
| bashnya_slov | 43522 |
| crosswords   | 39290 |
| guess_answer | 1434  |
| ostrova      | 1526  |
| top_seven    | 6643  |
| ugadaj_slova | 7406  |
| umnyasha     | 33052 |

## Teaching minibob solve puzzles

You can skip this section, ready to use [model](https://huggingface.co/artemsnegirev/minibob) you can find on 🤗 hub:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("artemsnegirev/minibob")
model = AutoModelForSeq2SeqLM.from_pretrained("artemsnegirev/minibob")
```

### How to

To train model we need `transformers` library, it's fast to prototype and easy to work with pretrained Transformer models.

Training pipeline is here:
minibob/train_minibob.ipynb

It's better to open notebook in Google Colab, which gives you free to use GPU. Training on a local computer with CPU will continue too much long. Switch `Runtime mode` to GPU (Hardware accelerator -> GPU) and execute cells one by one.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/artemsnegirev/minibob/blob/main/minibob/train_minibob.ipynb)

In the end you can [push your model](https://huggingface.co/docs/transformers/v4.27.2/en/model_sharing) on Huggingface Hub!

## Additional notes

For my experiments I used [ruT5-base](https://huggingface.co/ai-forever/ruT5-base) model that was trained by SberDevices team. It is an encoder-decoder [Transformer](https://jalammar.github.io/illustrated-transformer/) and works fine for this task. I added additional prefix `guess word`, but I think it will work well without it.

I used only several subsets:

```python
subsets = ["350_zagadok", "ostrova", "ugadaj_slova", "umnyasha"]
```

And also used only NOUNs for training, it is easy to guess and adds additional constraints for output. Training dataset consists of ~25k examples. I hold out 10% as a test set to measure network generalization. I measure `exact_match`:

```
true_answer     model_prediction    exact_match
дом             дома                0    
дом             дом                 1
```

I added feature - set ellipses (...) as a placeholder for model prediction, I found it useful, and T5 already can do it:

```bash
usr: обратная ... монеты
bob: сторона
```

Result `exact_match` is 0.19, that is small, but little better than the model before finetuning. After all, our main goal is to teach play Alias where prompts are much easier. So I handcrafted test cases as I played Alias, there are some of them:

```json
{"answer": "время", "prompt": "измеряется в часах, минутах и секундах"},
{"answer": "жизнь", "prompt": "противоположность смерти"},
{"answer": "день", "prompt": "время суток идущее после утра"},
{"answer": "рука", "prompt": "верхняя конечность человека"},
{"answer": "работа", "prompt": "место, куда человек устраивается после учебы"},
{"answer": "слово", "prompt": "составляющая предложений"}
```

I asked Minibob to generate 5 candidates without [sampling](https://huggingface.co/blog/how-to-generate). If the candidates list contains an answer, I count as a correct prediction otherwise not correct.

Test cases `exact_match` score is 0.65. Much better!

You could try to train minibob over all subsets and publish it to Huggingface Hub!

### Hyperparams info

To use larger batch fp16 and gradient_accumulation=2, it comes to 64 batch size and works for better regularization. AdamW with learning rate of 1e-4, warmup 1% and decayed params by factor of 0.1. Total training steps over 8 epochs is ~2.9k, that is equal to ~185k training examples.

## Telegram integration to meet minibob with real world

Here I don't give already live demo bot, but you can build (and it's much better) it with already ready components.

### How to

To build telegram bot we will use `aiogram` and some other libs for minibob model inference. We will wrapp our Python bot into Docker image to run it anywhere. 

If you dont't want install Docker on your system try to run pure in Python, note that I use prebuild image:

```Dockerfile
FROM huggingface/transformers-inference:4.24.0-pt1.13-cpu
```

Let's push it to public!

1) Create new bot (for example `minibob_<any number>_bot`) and obtain your bot token ([how to](https://core.telegram.org/bots/tutorial#obtain-your-bot-token)) and put it in `.env` file:

    ```bash
    TG_BOT_TOKEN="<your_telegram_token>"
    ```

2) Put model name you will use for your bot, note that it should be listed in Huggingface Hub:

    ```bash
    # replace with your model name if it's needed
    HF_MODEL_NAME="artemsnegirev/minibob"

    # if you use some private model for any reason
    # but for "artemsnegirev/minibob" it is not required
    HF_TOKEN="<huggingface read token>"
    ```

3) Build docker image:

    ```bash
    docker build -t minibob .
    ```

4) Run docker container:

    ```bash
    docker run --env-file .env -v $(pwd)/.cache:/app/.cache minibob
    ```
5) Now you supported machine revolution 🤖️

## Final notes

We learned how to solve simple data parsing tasks using Selenium and keep the codebase extendable, after we deep dived into finetuning large pretrained model (ruT5-base in our case), meets with Google Colab and found several tricks to make a batch size larger, finally teached Minibob and learned how to publish the model to the hub. Cherry on the cake is pushing the model to toy production using Docker.

Bot implemented as simple as possible. There are no game management rules: remember which words are already guessed, sample a word from the predefined list (about cars, tech, etc.), restart progress, skip a word to describe and so on. It is room for improvement. Although it is already funny.

```
usr: это человек который приходит починить туалет и раковину
bob: мои предположения - починщик, ремонтник, ремонтер, няня, сантехник
```

Няня 😂

```
usr: это мифическое животное похожее на лошадь с рогом и вокруг радуга
bob: мои предположения - ящер, орел, аист, лев, аид
```

Аид 😈🔱

## References

More details:
- https://jalammar.github.io/illustrated-gpt2/
- https://jalammar.github.io/illustrated-transformer/

How to:
- https://shivanandroy.com/fine-tune-t5-transformer-with-pytorch/
- https://huggingface.co/blog/how-to-generate

Artifacts:
- https://huggingface.co/datasets/artemsnegirev/ru-word-games
- https://huggingface.co/datasets/artemsnegirev/minibob
