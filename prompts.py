from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load a pretrained GPT model and tokenizer
model_name = "gpt2"  # Change to any other model like "gpt-neo" or "gpt-j" as needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_random_prompts(batch_size=10, prompt_length=20):
    prompts = []
    for _ in range(batch_size):
        # Start generation with a random word or short phrase
        input_text = "cat"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        # Generate text
        outputs = model.generate(
            input_ids,
            max_length=prompt_length,
            do_sample=True,
            temperature=0.7,
            top_k=50
        )

        # Decode and store each generated prompt
        prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prompts.append(prompt)
    
    return prompts

# Generate a batch of random prompts
random_prompts = generate_random_prompts(batch_size=5)
for i, prompt in enumerate(random_prompts, 1):
    print(f"Prompt {i}: {prompt}")
