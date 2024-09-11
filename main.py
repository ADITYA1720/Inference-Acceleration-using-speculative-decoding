import sys
import time
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTQConfig
from utils import Utils
from sampling import Autoregressive_Sampling, Speculative_Sampling

parser = argparse.ArgumentParser(description='Speculative Sampling')
parser.add_argument('--method', default="speculative", help='Sampling Method (autogressive / speculative)')
parser.add_argument('--prompt', required=True, help='Input prompt')
parser.add_argument('--max_new_tokens', type=int, required=True, help='No. of max new tokens')
parser.add_argument('--temperature', default=0, type=float, help='Temperature')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device : {device}")


target_model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-AWQ").to(device)
draft_model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-AWQ").to(device)
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-13B-AWQ")


if args.method == "speculative":

    print("Using target model:", target_model)
    print("Using draft model:", draft_model)
    lookahead = 4
    speculative_sampler = Speculative_Sampling(target_model, draft_model, tokenizer, args.prompt, args.temperature, lookahead)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    start_time = time.time_ns()
    tokens = speculative_sampler.speculative_sampling(target_model, draft_model, initial_prompt_seq=inputs.input_ids, target_len=args.max_new_tokens+len(inputs.input_ids), tokenizer=tokenizer, temperature=args.temperature, debug=False)
    end_time = time.time_ns()
    new_tokens = len(tokens[0]) - len(inputs.input_ids)
    time_taken = (end_time - start_time) / 1_000_000_000
    print(tokenizer.decode(tokens[0]))
    print()
    print("New tokens:", new_tokens)
    print(f"Latency (Speculative Sampling): {new_tokens/time_taken:.2f} tok/s")
    print("--------------------------------------------------------------------------------------------\n")

elif args.method == "autoregressive":

    print("Using target model:", target_model)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    auto_regressor = Autoregressive_Sampling(target_model, args.prompt, args.temperature)
    start_time = time.time_ns()
    tokens = auto_regressor.autoregressive_sampling(target_model, initial_prompt_seq=inputs.input_ids, target_len=args.max_new_tokens+len(inputs.input_ids), temperature=args.temperature)
    end_time = time.time_ns()
    new_tokens = len(tokens[0]) - len(inputs.input_ids)
    time_taken = (end_time - start_time) / 1_000_000_000
    print(tokenizer.decode(tokens[0]))
    print()
    print("New tokens:", new_tokens)
    print(f"Latency (Naive Autoregressive Sampling): {new_tokens/time_taken:.2f} tok/s")
    print("--------------------------------------------------------------------------------------------\n")