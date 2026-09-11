from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import DEFAULT_INTERPRETER_SYSTEM_PROMPT

app = FastAPI(title="Qwen OpenAI-Compatible API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Cargando tokenizador y modelo...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="cpu"
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "qwen2.5-1.5b"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    input_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    # Keep caller-provided system prompts (including language-specific interpreter rules).
    if not any(message["role"] == "system" for message in input_messages):
        input_messages.insert(0, {
            "role": "system",
            "content": DEFAULT_INTERPRETER_SYSTEM_PROMPT,
        })
    
    # Aplica la plantilla oficial de chat de Qwen2.5
    text = tokenizer.apply_chat_template(
        input_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Parámetros optimizados para obediencia estricta y eliminación de artefactos
    generated_ids = model.generate(
        **model_inputs, 
        max_new_tokens=request.max_tokens or 256,
        do_sample=False,
        repetition_penalty=1.1,     # Evita pegar palabras o repetir estructuras
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {
        "id": "chatcmpl-qwen-local",
        "object": "chat.completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text.strip()
                },
                "finish_reason": "stop"
            }
        ]
    }