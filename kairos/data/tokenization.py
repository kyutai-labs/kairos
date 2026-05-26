from functools import reduce

import sentencepiece


def tokenize(
    text: str,
    tokenizer: sentencepiece.SentencePieceProcessor,
    bos: bool = True,
    eos: bool = False,
) -> list[int]:
    nl_piece = tokenizer.encode("\n")[-1]
    tokens = tokenizer.encode(text.split("\n"))
    tokens = reduce(lambda a, b: a + [nl_piece] + b, tokens)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    if eos:
        tokens.append(tokenizer.eos_id())
    return tokens


def detokenize(
    tokens: list[int],
    tokenizer: sentencepiece.SentencePieceProcessor,
    strip_bos: bool = False,
    strip_eos: bool = False,
) -> str:
    tokens = [i for i in tokens if i != tokenizer.pad_id() and i != tokenizer.unk_id()]
    if strip_bos and tokenizer.bos_id() in tokens[1:]:
        tokens = tokens[: tokens.index(tokenizer.bos_id(), 1)]
    if strip_eos and tokenizer.eos_id() in tokens[1:]:
        tokens = tokens[: tokens.index(tokenizer.eos_id(), 1)]
    text = tokenizer.decode(tokens)
    return text.replace("\n ", "\n")


def tokenize_conversation(
    messages: list[dict[str, str]],
    tokenizer: sentencepiece.SentencePieceProcessor,
    csz: int,
) -> dict[str, list[int]] | None:
    tokens: list[int] = []
    mask: list[int] = []
    for i, turn in enumerate(messages):
        role = turn["role"]
        content = turn["content"]
        text = f"<|im_start|>{role}\n{content}<|im_end|>\n"

        if len(text) > 100000:
            return None
        ti = tokenize(text, tokenizer, bos=(i == 0), eos=False)
        tokens.extend(ti)
        mask.extend([int(role == "assistant")] * len(ti))

    tokens.extend([tokenizer.pad_id()] * (csz + 1 - len(tokens)))
    mask.extend([0] * (csz + 1 - len(mask)))
    return {"text": tokens, "mask": mask}
