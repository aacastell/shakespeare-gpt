from transformers import GPT2Tokenizer

EXPECTED_VOCAB_SIZE = 50257

class GPT2TokenizerWrapper:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        vocab_size = self.tokenizer.vocab_size
        if vocab_size != EXPECTED_VOCAB_SIZE:
            raise ValueError(
                f"Tamaño de Vocabulario: {vocab_size}. Esperado {EXPECTED_VOCAB_SIZE}"
            )
        
    def encode(self, text:str):
        return self.tokenizer.encode(text)
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
    
    def get_vocab_size(self):
        return self.tokenizer.vocab_size