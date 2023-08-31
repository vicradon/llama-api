from typing import Union
import uvicorn
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastapi import FastAPI

app = FastAPI()






@app.get("/")
def read_root():
    model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_pipe = pipeline("sentiment-analysis", model=model_id)
    return {"Output": sentiment_pipe('I love you')}


model_name = 'VMware/open-llama-7b-open-instruct'


# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='sequential')

# prompt_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

# prompt = 'Explain in simple terms how the attention mechanism of a transformer model works'


# inputt = prompt_template.format(instruction=prompt)
# input_ids = tokenizer(inputt, return_tensors="pt").input_ids.to("cuda")

# output1 = model.generate(input_ids, max_length=512)
# input_length = input_ids.shape[1]
# output1 = output1[:, input_length:]
# output = tokenizer.decode(output1[0])

# print(output)





@app.post("/completion/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


if __name__ == '__main__':
    uvicorn.run("main:app", port=8000, log_level="info", reload=True)