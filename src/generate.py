import torch
import torch.nn.functional as F

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    block_size: int=128,
    temperature: float = 1.0,
    top_k: int= None,
    device: str = "cpu"
    ):

    model.eval()

    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):

        input_cond = input_ids[:, -block_size:]

        logits = model(input_cond)

        logits = logits[:, -1, :]

        logits = logits/temperature

        if top_k is not None:
            values, _ =  torch.topk(logits, top_k)
            min_val = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_val, torch.full_like(logits, -float("inf")), logits)

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat((input_ids, next_token), dim=1)

    output_tokens = input_ids[0].tolist()
    return tokenizer.decode(output_tokens)