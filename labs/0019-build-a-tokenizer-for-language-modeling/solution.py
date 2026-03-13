def train_tokenizer(corpus, vocab_size):
    """
    Builds a tokenizer by mapping unique characters in the corpus to IDs.
    """
    # 1. Identify all unique characters in the training corpus
    unique_chars = sorted(list(set(''.join(corpus))))
    
    # 2. Ensure we handle characters not in the training corpus (fallback)
    # We'll use the first 256 Unicode characters as a safe default 
    # if the corpus is small, or just map what we find.
    vocab = {ch: i for i, ch in enumerate(unique_chars) if i < vocab_size}
    inverse_vocab = {i: ch for ch, i in vocab.items()}

    def encode(text: str) -> list[int]:
        res = []
        for char in text:
            if char in vocab:
                res.append(vocab[char])
            else:
                # Fallback for unknown characters: 
                # map to their ordinal value modulo vocab_size
                res.append(ord(char) % vocab_size)
        return res

    def decode(tokens: list[int]) -> str:
        res = []
        for t in tokens:
            if t in inverse_vocab:
                res.append(inverse_vocab[t])
            else:
                # Fallback: convert ID back to character
                res.append(chr(t))
        return "".join(res)

    return encode, decode