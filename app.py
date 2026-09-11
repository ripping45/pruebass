from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Qwen Translation API")

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Cargando tokenizador y modelo...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cpu"
)

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "auto"
    target_lang: str = "es"

@app.post("/v1/translate")
def translate(request: TranslationRequest):
    prompt = f"Traduce el siguiente texto al idioma '{request.target_lang}'. Devuelve ÚNICAMENTE la traducción, sin explicaciones ni texto adicional:\n\n{request.text}"
    
    messages = [
        {"role": "system", "content": "Eres un traductor profesional ultra preciso."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    translation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return {"translated_text": translation.strip()}