# Inference-Acceleration-using-speculative-decoding

## This repo is inspired from the paper by Google Deepmind
## 
    # https://arxiv.org/pdf/2302.01318

## Run the file using
##
    python generate.py -h

## Autoregressive Decoding
##
    python main.py  --method autoregressive \
                    --prompt "Describe how a neural network is trained using backpropagation, and explain the significance of each step." \
                    --max_new_tokens 128 \
                    --temperature 0.1

## Speculative Decoding
##
    python main.py  --method speculative \
                    --prompt "Describe how a neural network is trained using backpropagation, and explain the significance of each step." \
                    --max_new_tokens 128 \
                    --temperature 0.1
## How to improve the performance of your speculative decoding technique

1. The draft model must be significantly small compared to that of the target model.
2. Both models should use the same tokenizer
3. Efficient batching can improve performance however may lead to memory management issues.

## Advantages

1. Speculative Sampling offers a 1.5X - 3.0X speedup on naive autoregression

## Caveats

1. The size of the draft model if is comparable to the target model can reduce speed due to significant overhead by the draft model.
2. Using different tokenizers for both models will drastically decrease performance
