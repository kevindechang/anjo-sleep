"""The CPU smoke run's tiny Glimmer (``anjo.sleep.train.smoke``): a random-weight
``MuseGlimmerForConditionalGeneration`` with Glimmer's layout at toy sizes (a text decoder with the
attention output gate and the 3-sliding-then-1-NoPE layer pattern, a vision tower, its adapter and
projection), and a word-level tokenizer with a chat template, saved to one folder.

Like ``colab_train.py`` it runs as a file in the training environment, so it imports nothing from
Anjo, and torch, tokenizers and transformers load only inside ``main``. Nothing is downloaded.
Usage: ``python smoke_model.py <folder>``.
"""
from __future__ import annotations

import sys
from pathlib import Path

BOS, EOS, UNK = "<|begin_of_text|>", "<|eot|>", "<|unk|>"
HEADER_START, HEADER_END = "<|header_start|>", "<|header_end|>"
IMAGE, VIDEO = "<|patch|>", "<|video|>"
SPECIAL = (BOS, EOS, UNK, HEADER_START, HEADER_END, IMAGE, VIDEO)
# The smoke rows' words (``smoke.py`` writes its rows from these); any other word is UNK.
WORDS = (
    "system", "user", "assistant", "you", "are", "her", "she", "remembers", "what", "did", "i", "say",
    "about", "the", "garden", "tomatoes", "were", "late", "this", "year", "and", "rain", "came",
    "early", "we", "planted", "them", "in", "may", "so", "maybe", "next", "time", "plant", "sooner",
    "that", "is", "fine", "no", "idea",
)
CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "<|header_start|>{{ message['role'] }}<|header_end|>{{ message['content'] }}<|eot|>"
    "{% endfor %}{% if add_generation_prompt %}<|header_start|>assistant<|header_end|>{% endif %}"
)
VOCAB = 256
LAYERS = 4  # one block of Glimmer's pattern: three sliding-window RoPE layers, then a full NoPE one
TEXT = {
    "vocab_size": VOCAB, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": LAYERS,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "sliding_window": 32,
    "max_position_embeddings": 16_384, "rope_parameters": {"rope_theta": 500_000.0, "rope_type": "default"},
}
VISION_WIDTH = 32
VISION = {
    "hidden_size": VISION_WIDTH, "intermediate_size": 64, "num_attention_heads": 2, "num_hidden_layers": 2,
    "pos_emb_height": 4, "pos_emb_width": 4, "max_position_embeddings": 16,
    "rope_parameters": {"rope_theta": 10_000.0, "rope_type": "default"},
}


def main(folder: Path) -> int:  # pragma: no cover - runs only in the training environment
    import torch  # type: ignore[import-not-found, unused-ignore]
    from tokenizers import Tokenizer, models, pre_tokenizers  # type: ignore[import-not-found, unused-ignore]
    from transformers import (  # type: ignore[import-not-found, unused-ignore]
        MuseGlimmerConfig,
        MuseGlimmerForConditionalGeneration,
        PreTrainedTokenizerFast,
    )

    vocab = {token: index for index, token in enumerate((*SPECIAL, *WORDS))}
    word_level = Tokenizer(models.WordLevel(vocab=vocab, unk_token=UNK))
    word_level.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=word_level, bos_token=BOS, eos_token=EOS, unk_token=UNK,
        additional_special_tokens=[HEADER_START, HEADER_END, IMAGE, VIDEO], chat_template=CHAT_TEMPLATE,
    )
    text = {**TEXT, "bos_token_id": vocab[BOS], "eos_token_id": vocab[EOS]}
    merged = VISION_WIDTH * 4  # the vision tower's width after its 2x2 patch merge
    config = MuseGlimmerConfig(
        text_config=text, vision_config=VISION, image_token_id=vocab[IMAGE], video_token_id=vocab[VIDEO],
        out_hidden_size=merged, projector_hidden_size=64,
    )
    torch.manual_seed(0)
    model = MuseGlimmerForConditionalGeneration(config).to(torch.float32)
    folder.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    return 0


if __name__ == "__main__":  # pragma: no cover - run as a file in the training environment
    if len(sys.argv) != 2:
        raise SystemExit("usage: smoke_model.py <folder>")
    sys.exit(main(Path(sys.argv[1])))
