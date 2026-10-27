"""The CPU smoke run's tiny Glimmer (``anjo.sleep.train.smoke``): a random-weight
``MuseGlimmerForConditionalGeneration`` with Glimmer's layout at toy sizes (a text decoder with the
attention output gate and the 3-sliding-then-1-NoPE layer pattern, a vision tower, its adapter and
projection), and a tokenizer, saved to one folder.

The tokenizer is a word-level one that frames a chat the way Glimmer's does (``<|start|>role
<|message|>words<|eot|>``, the reply's header `` to=user``, a BOS the tokenizer puts first itself),
or, given a folder holding Glimmer's own tokenizer and chat template, that tokenizer, with the
model's vocabulary sized to it.

Like ``colab_train.py`` it runs as a file in the training environment, so it imports nothing from
Anjo, and torch, tokenizers and transformers load only inside ``main``. Nothing is downloaded.
Usage: ``python smoke_model.py <folder> [<tokenizer folder>]``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

BOS, EOS, EOT, UNK = "<|begin_of_text|>", "<|end_of_text|>", "<|eot|>", "<|unk|>"
START, MESSAGE = "<|start|>", "<|message|>"
IMAGE, VIDEO = "<|patch|>", "<|video|>"
SPECIAL = (BOS, EOS, EOT, UNK, START, MESSAGE, IMAGE, VIDEO)
# The smoke rows' words (``smoke.py`` writes its rows from these) and the header's; any other is UNK.
WORDS = (
    "system", "user", "assistant", "you", "are", "her", "she", "remembers", "what", "did", "i", "say",
    "about", "the", "garden", "tomatoes", "were", "late", "this", "year", "and", "rain", "came",
    "early", "we", "planted", "them", "in", "may", "so", "maybe", "next", "time", "plant", "sooner",
    "that", "is", "fine", "no", "idea", "to", "=",
)
CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}<|start|>{{ message['role'] }}"
    "{% if message['role'] == 'assistant' %} to=user{% endif %}<|message|>{{ message['content'] }}<|eot|>"
    "{% endfor %}{% if add_generation_prompt %}<|start|>assistant{% endif %}"
)
LAYERS = 4  # one block of Glimmer's pattern: three sliding-window RoPE layers, then a full NoPE one
TEXT = {
    "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": LAYERS,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "sliding_window": 32,
    "max_position_embeddings": 16_384, "rope_parameters": {"rope_theta": 500_000.0, "rope_type": "default"},
}
VISION_WIDTH = 32
VISION = {
    "hidden_size": VISION_WIDTH, "intermediate_size": 64, "num_attention_heads": 2, "num_hidden_layers": 2,
    "pos_emb_height": 4, "pos_emb_width": 4, "max_position_embeddings": 16,
    "rope_parameters": {"rope_theta": 10_000.0, "rope_type": "default"},
}


def _word_level() -> object:  # pragma: no cover - runs only in the training environment
    from tokenizers import Tokenizer, models, pre_tokenizers, processors  # type: ignore[import-not-found, unused-ignore]
    from transformers import PreTrainedTokenizerFast  # type: ignore[import-not-found, unused-ignore]

    vocab = {token: index for index, token in enumerate((*SPECIAL, *WORDS))}
    word_level = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK))
    word_level.pre_tokenizer = pre_tokenizers.Whitespace()
    word_level.post_processor = processors.TemplateProcessing(
        single=f"{BOS} $A", pair=f"{BOS} $A {BOS} $B", special_tokens=[(BOS, vocab[BOS])])
    return PreTrainedTokenizerFast(
        tokenizer_object=word_level, bos_token=BOS, eos_token=EOS, unk_token=UNK,
        additional_special_tokens=[EOT, START, MESSAGE, IMAGE, VIDEO], chat_template=CHAT_TEMPLATE,
    )


def main(folder: Path, glimmer: Path | None = None) -> int:  # pragma: no cover - runs only in the training environment
    import torch  # type: ignore[import-not-found, unused-ignore]
    from transformers import (  # type: ignore[import-not-found, unused-ignore]
        AutoTokenizer,
        MuseGlimmerConfig,
        MuseGlimmerForConditionalGeneration,
    )

    tokenizer: Any = _word_level() if glimmer is None else AutoTokenizer.from_pretrained(glimmer)
    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in (BOS, EOS, IMAGE, VIDEO)}
    text = {**TEXT, "vocab_size": len(tokenizer), "bos_token_id": ids[BOS], "eos_token_id": ids[EOS]}  # Glimmer's: 202,048
    merged = VISION_WIDTH * 4  # the vision tower's width after its 2x2 patch merge
    config = MuseGlimmerConfig(
        text_config=text, vision_config=VISION, image_token_id=ids[IMAGE], video_token_id=ids[VIDEO],
        out_hidden_size=merged, projector_hidden_size=64,
    )
    torch.manual_seed(0)
    model = MuseGlimmerForConditionalGeneration(config).to(torch.float32)
    folder.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return 0


if __name__ == "__main__":  # pragma: no cover - run as a file in the training environment
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: smoke_model.py <folder> [<tokenizer folder>]")
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) == 3 else None))
