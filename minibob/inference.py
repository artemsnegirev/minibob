from typing import List

from transformers import T5ForConditionalGeneration as T5Model, T5Tokenizer


class InferencePipe:
    def __init__(self, model_name, token, cache_dir) -> None:
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
            model_name, use_auth_token=token, cache_dir=cache_dir
        )
        
        self.model: T5Model = T5Model.from_pretrained(
            model_name, use_auth_token=token, cache_dir=cache_dir, low_cpu_mem_usage=True
        )
        
    def __call__(self, text: str) -> List[str]:
        prefix = 'guess word:'

        prompt = f'{prefix} {text}'
        prompt = prompt.replace('...', '<extra_id_0>')

        inputs = self.tokenizer(
            [prompt], max_length=32, truncation=True, return_tensors='pt'
        )

        # TODO: add generation config for `num_beams` and `max_new_tokens`

        outputs = self.model.generate(
            inputs.input_ids, 
            num_beams=5, 
            max_new_tokens=8,
            do_sample=False,
            num_return_sequences=5
        )

        candidates = []
        for tokens in outputs:
            decoded: str = self.tokenizer.decode(tokens, skip_special_tokens=True)
            decoded = decoded.strip().lower()
            candidates.append(decoded)

        return candidates
