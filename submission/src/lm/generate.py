import argparse
import json
import os
import math

import tiktoken
import torch
from omegaconf import OmegaConf
from tqdm import trange
from lm.model import DecoderLM
from lm.utils import determine_device, enable_tf32
from lm.train import compute_language_modeling_loss


def softmax_with_temperature(
    logits: torch.FloatTensor, temperature: float
) -> torch.FloatTensor:
    """Turns logits into probabilities under softmax (with temperature)

    Args:
        logits: a 2d torch tensor of token ids (B x V)
        temperature: temperature of the softmax function

    Returns:
        a 2d torch tensor of token probabilities (B x V)
    """

    # to avoid division by 0
    temperature = max(temperature, 1e-5)
    logits = logits / temperature
    return logits.softmax(1)


@torch.inference_mode()
def generate(
    model: DecoderLM,
    device: str,
    tokenizer: tiktoken.Encoding,
    prefixes: list[str],
    batch_size: int,
    max_new_tokens: int = 32,
    temperature: float = 0.1,
) -> list[str]:
    """Generates completions conditioned on prefixes; computes perplexity

    Args:
        model: the language model
        device: device to put the tensors on
        tokenizer: the tokenizer
        prefixes: a list of strings as prefixes for generation
        batch_size: number of prefixes to batch together during generation
        max_new_tokens: the number of tokens to generate for each prefix
        temperature: temperature parameter of softmax

    Returns:
        a list of strings (continuations to prefixes)

    Note: you should implement a batched version of this function by
        left-padding tokenized prefixes with `tokenizer.eot_token` so that all
        sequences have equal length. `attention_mask` should be set to 0.0 for
        padding tokens, and 1.0 everywhere else.
    """

    generations = []
    all_losses = []
    
    for i in trange(0, len(prefixes), batch_size):
        batch_prefixes = prefixes[i : i + batch_size]
        tokenized_prefixes = [tokenizer.encode(prefix) for prefix in batch_prefixes]
        max_len = max(len(seq) for seq in tokenized_prefixes)
        
        input_ids = []
        attention_mask = []
        for seq in tokenized_prefixes:
            padding_len = max_len - len(seq)
            padded_seq = [tokenizer.eot_token] * padding_len + seq
            mask = [0.0] * padding_len + [1.0] * len(seq)
            
            input_ids.append(padded_seq)
            attention_mask.append(mask)
        input_ids = torch.LongTensor(input_ids).to(device)
        attention_mask = torch.FloatTensor(attention_mask).to(device)
        
        logits = model(input_ids, attention_mask)
        loss = compute_language_modeling_loss(input_ids, logits)
        all_losses.append(loss.item())
        
        for _ in range(max_new_tokens):
            # Get logits for the last position
            logits = model(input_ids, attention_mask)
            next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            
            # Apply temperature and sample
            probs = softmax_with_temperature(next_token_logits, temperature)
            next_tokens = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            # Extend attention mask (new tokens are always attended to)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), device=device)
            ], dim=1)
        
        for j, seq in enumerate(tokenized_prefixes):
            prefix_len = max_len  # All sequences are now the same length after padding
            generated_ids = input_ids[j, prefix_len:].tolist()
            generated_text = tokenizer.decode(generated_ids)
            generations.append(generated_text)
    
    mean_loss = sum(all_losses) / len(all_losses)
    perplexity = math.exp(mean_loss)

    print(f"Perplexity: {perplexity}")
    return generations


def main():
    enable_tf32()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=OmegaConf.load,
        required=True,
        help="the yaml config file used for model training",
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        required=True,
        help="a json file with a list of strings as prefixes for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="temperature in sampling"
    )

    args = parser.parse_args()
    config = args.config
    with open(args.prefixes) as f:
        prefixes = [json.loads(line)["prefix"] for line in f]
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature

    # initialize tokenizer and model
    model_path = os.path.join(config.output_dir, "model.pt")
    assert os.path.exists(model_path), f"no model checkpoint at {model_path}"
    tokenizer = tiktoken.get_encoding(config.tokenizer_encoding)
    device = determine_device() if config.device == "auto" else config.device
    model = DecoderLM(tokenizer.n_vocab, **config.model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # generate and save outputs
    model.eval()
    generations = generate(
        model,
        device,
        tokenizer,
        prefixes,
        config.batch_size,
        max_new_tokens,
        temperature,
    )

    generation_path = os.path.join(config.output_dir, "generation.jsonl")
    print(f"writing generations to {generation_path}")
    with open(generation_path, "w") as f:
        for prefix, generation in zip(prefixes, generations):
            json.dump({"prefix": prefix, "generation": generation}, f)
            f.write("\n")

    print("done!")


if __name__ == "__main__":
    main()
