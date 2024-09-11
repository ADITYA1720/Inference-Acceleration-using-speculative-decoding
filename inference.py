import time
import os
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sampling import Autoregressive_Sampling, Speculative_Sampling


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device : {device}")
print("----------------------------------------------")

print("Logging into Hugging Face Hub")
from huggingface_hub import login
login(new_session=False, # Won’t request token if one is already saved on machine
write_permission=True, # Requires a token with write permission
token='YOUR HUGGING FACE TOKEN', # The name of your token
add_to_git_credential=True)
print("----------------------------------------------")


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
target_model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-AWQ").to(device)
draft_model =  AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-AWQ").to(device)

prompts_sample = [ "A AI and a human form an unlikely friendship while solving a mystery in a futuristic city.",
"Explain the steps to solve a Sudoku puzzle, and how to approach it using logical reasoning.",
"Describe how a neural network is trained using backpropagation, and explain the significance of each step.",
"Imagine if the Industrial Revolution occurred 500 years earlier. How would society, technology, and politics be different today?",
"Debate whether free will can exist in a deterministic universe. Provide arguments for both sides and attempt to reach a conclusion.",
"Generate Python code that simulates a simple turn-based game between two players, and include the option for AI to play as one of the players.",
"Predict what transportation will look like 100 years from now and how it will affect city planning and everyday life.",
"Compare and contrast the role of individualism and collectivism in Western and Eastern societies.",
"Create a backstory for a character who is a space explorer discovering an alien civilization for the first time.",
"Discuss the ethical implications of using AI in law enforcement and the potential risks of bias in decision-making."]


texts = prompts_sample

MAX_NEW_TOKENS = 128
TEMPERATURE = 0 # 0 for Deterministic

print("Target Model -", target_model.config._name_or_path)
print("Draft Model -", draft_model.config._name_or_path)
print("----------------------------------------------\n")

inputs_sample = tokenizer(random.choice(texts), return_tensors="pt").to(device)
tokens = target_model.generate(**inputs_sample, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
print("HF's generate")
print("Count of new tokens:", len(tokens[0]) - len(inputs_sample.input_ids))
print(tokenizer.decode(tokens[0]))
print("----------------------------------------------\n")

lookahead = 4
speculative_sampler = Speculative_Sampling(target_model=target_model, draft_model=draft_model, tokenizer=tokenizer, prompt=inputs_sample.input_ids, temperature=TEMPERATURE, lookahead=lookahead)
auto_regressor = Autoregressive_Sampling(target_model=target_model, prompt=inputs_sample.input_ids, temperature=TEMPERATURE)

tokens = auto_regressor.autoregressive_sampling(target_model, initial_prompt_seq=inputs_sample.input_ids, target_len=MAX_NEW_TOKENS+len(inputs_sample.input_ids), temperature=TEMPERATURE)
print("Naive Autoregressive with temperature")
print("Count of new tokens:", len(tokens[0]) - len(inputs_sample.input_ids))
print(tokenizer.decode(tokens[0]))
print("--------------------------------------------------------------------------------------------\n")

tokens = speculative_sampler.speculative_sampling(target_model, draft_model, initial_prompt_seq=inputs_sample.input_ids, max_new_tokens=MAX_NEW_TOKENS, tokenizer=tokenizer, temperature=TEMPERATURE, debug=False)
print("Speculative Sampling with temperature")
print("Count of new tokens:", len(tokens[0]) - len(inputs_sample.input_ids))
print(tokenizer.decode(tokens[0]))
print("--------------------------------------------------------------------------------------------\n")
print()

print("Autoregressive Sampling.")
tokens = auto_regressor.autoregressive_sampling(target_model, initial_prompt_seq=inputs_sample.input_ids, target_len=MAX_NEW_TOKENS+len(inputs_sample.input_ids), temperature=TEMPERATURE)

time_taken = 0
new_tokens = 0
for i in tqdm(range(len(texts))):
  text = texts[i]
  inputs = tokenizer(text, return_tensors="pt").to(device)
  start_len = len(inputs.input_ids)

  start_time = time.time_ns()
  tokens = auto_regressor.autoregressive_sampling(target_model, initial_prompt_seq=inputs.input_ids, target_len=MAX_NEW_TOKENS+len(inputs.input_ids), temperature=TEMPERATURE)
  end_time = time.time_ns()

  new_tokens += len(tokens[0]) - start_len
  time_taken += (end_time - start_time) / 1_000_000_000

print(f"Latency (Autoregressive Sampling): {new_tokens/time_taken:.2f} tok/s")
print("--------------------------------------------------------------------------------------------\n")


print("Speculative Sampling...")
tokens = speculative_sampler.speculative_sampling(target_model, draft_model, initial_prompt_seq=inputs_sample.input_ids, max_new_tokens=MAX_NEW_TOKENS, tokenizer=tokenizer, temperature=TEMPERATURE, debug=False)

time_taken = 0
new_tokens = 0
for i in tqdm(range(len(texts))):
  text = texts[i]
  inputs = tokenizer(text, return_tensors="pt").to(device)
  start_len = len(inputs.input_ids)

  start_time = time.time_ns()
  tokens = speculative_sampler.speculative_sampling(target_model, draft_model, initial_prompt_seq=inputs.input_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, tokenizer=tokenizer, debug=False)
  end_time = time.time_ns()

  new_tokens += len(tokens[0]) - start_len
  time_taken += (end_time - start_time) / 1_000_000_000

print(f"Latency (Speculative Sampling): {new_tokens/time_taken:.2f} tok/s")
print("--------------------------------------------------------------------------------------------\n")









