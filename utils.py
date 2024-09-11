import torch

class Utils:
    def __init__(self, model, temperature=1.0):
        """
        model: the draft model used for sampling
        temperature: the sampling temperature, default is 1.0
        """
        self.model = model
        self.temperature = temperature

    def get_distribution(self, logits):
        """
        Computes the probability distribution with temperature scaling.
        """
        probs = torch.softmax(logits / (self.temperature + 1e-10), dim=-1)
        return probs

    def sample_token(self, logits):
        """
        Samples a token from the logits using the distribution from get_distribution.
        """
        probs = self.get_distribution(logits)
        return torch.multinomial(probs, num_samples=1)[0]

    def sample_from_draft_model(self, prompt, new_tokens):
        """
        Generates a sequence of new tokens using the autoregressive process.
        
        prompt: the initial input sequence
        new_tokens: the number of tokens to sample
        """
        response = prompt.detach().clone()
        out_logits = []

        for _ in range(new_tokens):
            sample_token_logits = self.model(response).logits[:, -1, :]
            sample_token = self.sample_token(sample_token_logits)
            
            # Concatenate the sampled token to the response
            response = torch.cat([response, sample_token[None, ...]], dim=-1)
            out_logits.append(sample_token_logits)

        # Stack logits of each sample step
        out_logits = torch.stack(out_logits, dim=-1)
        return response, out_logits
