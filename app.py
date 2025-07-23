from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import uvicorn

app = FastAPI()

model = T5ForConditionalGeneration.from_pretrained("hashtag_model")
tokenizer = T5Tokenizer.from_pretrained("hashtag_model")

class RequestText(BaseModel):
    text: str

@app.post("/generate")

def generate_tags(request: RequestText):
    inputs = tokenizer(request.text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=32)
    hashtags = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"hashtags": hashtags}

# This runs the server on Render
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)