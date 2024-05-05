from transformers import (
    AutoModelForCausalLM,
     AutoTokenizer)


prompt = "Who is the best football player of all time?"
path ="mistralai/Mistral-7B-Instruct-v0.2"
device = "cuda" # the device to load the model onto
print(path)
model = AutoModelForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
print("mistral")
messages=[{"role": "user", "content": prompt}]
# response = client.chat.completions.create(
# 	model=LM,
# 	messages=[
# 		{"role": "user", "content": prompt},
# 	],
# 	temperature=temperature,
# 	n=n,
# 	frequency_penalty=0,
# 	presence_penalty=0,
# 	stop=stop
# )
encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
response = tokenizer.batch_decode(generated_ids)
print("response:", response)