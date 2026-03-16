from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMJudge:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", device="cpu"):
        self.device = device
        print(f"Loading Judge Model: {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            trust_remote_code=True
        ).to(device)
        self.model.eval()
        print("Judge Model Loaded.")

    def evaluate(self, prompt_text, candidates):
        """
        prompt_text: What the user (AI trainer) said.
        candidates: List of strings (responses from our EvoAI).
        Returns: Index of the best candidate (0 to len(candidates)-1).
        """
        
        # Format the candidates
        options_text = ""
        for i, cand in enumerate(candidates):
            # Clean candidate to avoid messing up the prompt?
            # Truncate if too long (EvoAI might output infinite garbage)
            clean_cand = cand.strip()[:100]
            options_text += f"{i+1}. {clean_cand}\n"

        # Construct Prompt for the Judge
        # We want the judge to pick the best response.
        judge_prompt = f"""User said: "{prompt_text}"

Here are 5 candidate responses:
{options_text}

Which response is the most relevant or coherent? If all are bad, pick the one that contains real words or letters.
Output ONLY the number (1-5).
Answer:"""

        # Generate decision
        messages = [
            {"role": "system", "content": "You are a helpful judge selecting the best AI response."},
            {"role": "user", "content": judge_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=5, # We just need a number
                do_sample=False  # Deterministic
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Parse the number
        try:
            # Look for digits
            import re
            match = re.search(r'[1-5]', response)
            if match:
                choice = int(match.group())
                return choice - 1 # 0-indexed
            else:
                # If failed, return random? or 0.
                print(f"Judge output unclear: '{response}' -> Defaulting to 0")
                return 0
        except:
             return 0
