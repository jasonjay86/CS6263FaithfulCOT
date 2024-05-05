from transformers import (
    AutoModelForCausalLM,
     AutoTokenizer)

def get_Mistral_answer(text):
    """
    Extracts the first sentence after the marker [/INST] in the text.

    Args:
        text: The input text string.

    Returns:
        The first sentence after [/INST], or None if not found.
    """
    # print(text)
    inst_index = text.find("[/INST]")
    if inst_index == -1:
        return None
    print (inst_index)

    # Split the text after [/INST] using a sentence separator
    sentences = text[inst_index+len("[/INST]"):].split(". ")
    print("func:", sentences)
    if sentences:
        return sentences[0]
    else:
        return None

prompt = "Who is the best NFL player of all time?"
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
completions = []
choices = response
# completion_objs = [choice.message for choice in choices]

for choice in choices:
    completions.append(get_Mistral_answer(choice))

# completions = [completion.content for completion in choices]
print("COmpletions = ", completions)