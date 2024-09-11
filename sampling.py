import torch
from utils import Utils
import time

class Autoregressive_Sampling():
    def __init__(self, model, prompt, temperature=1.0):
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.utils = Utils(model, temperature)

    def autoregressive_sampling(self, target_len, temperature=1.0):
        n = self.prompt.shape[-1]
        response = self.prompt.detach.clone()

        while n < target_len:
            sample_token_logits = self.model(response).logits[:, -1, :]
            sample_token = self.utils.sample(sample_token_logits, temperature = temperature)
            response = torch.concat([response, sample_token[None,...]], dim=-1)
            n += 1
        return response
    
class Speculative_Sampling():
    def __init__(self, target_model, draft_model, tokenizer, prompt, temperature, lookahead):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.temperature  = temperature
        self.lookahead = lookahead
        self.utils= Utils(draft_model, temperature)

    def speculative_sampling(self, debug=True):
        '''Batch size should be 1'''
        assert self.prompt.shape[0] == 1 

        #n here is the length of the initial sequence
        n = self.prompt.shape[-1]
        # .detach() creates a new tensor, which shares the same data but without tracking the computation graph
        # .clone() creates a copy of the tensor such that the changes made to this tensor don't affect the original tensor
        response = self.prompt.detach().clone()

        while n< self.target_len:
            n_orig = n
            N = response.shape[-1]
            #Get output from the draft model - lightweight model
            draft_outputs, draft_logits = self.sample_from_draft_model(self.draft_model, response, new_tokens=self.lookahead, temperature=self.temperature)

            if debug:
                print(f"Possible continuations: {self.tokenizer.decode(draft_outputs[0,n_orig:], skip_special_tokens=True)}")

            target_logits = self.target_model(draft_outputs).logits[:, -self.lookahead-1:, :]

            target_model_distribution = self.utils.get_distribution(target_logits, self.temperature)
            draft_model_distribution = self.utils.get_distribution(target_logits, self.temperature)

            flag = 1

            for t in range(self.lookahead):
                numerator = target_model_distribution[:, t, draft_outputs[0, N+t]]
                denominator = draft_model_distribution[:, t, draft_outputs[0, N+t]]
                ratio = (numerator / denominator)

                uniform_distribution = torch.rand_like(numerator)
                ones_tensor = torch.ones_like(numerator)

                # Rejection Sampling
                ## Acceptance
                if (uniform_distribution < torch.min(ones_tensor, ratio)).any():
                    response = torch.concat([response, draft_outputs[:, N+t].unsqueeze(dim=-1)], dim=-1)
                    n += 1

                else:
                    new_dist = (target_model_distribution[:, t, :] - draft_model_distribution[:, t, :])
                    new_dist = torch.max(torch.zeros_like(new_dist), new_dist)
                    new_dist = new_dist / new_dist.sum(dim=-1, keepdim=True)
                    token_id = torch.multinomial(new_dist, num_samples=1)[0]
                    response = torch.concat([response, token_id[None,...]], dim=-1)
                    accepted_flag = 0
                    break

            if accepted_flag == 1:
                sample_token = self.utils.sample(target_logits[:, -1, :], temperature=self.temperature)
                response = torch.concat([response, sample_token[None,...]], dim=-1)
            
            if debug:
                print(f"Accepted continuations: {self.tokenizer.decode(response[0,n_orig:], skip_special_tokens=True)}")

            n += 1

        return response