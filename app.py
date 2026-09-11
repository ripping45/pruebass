from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import json
import os
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from prompts import DEFAULT_INTERPRETER_SYSTEM_PROMPT

logger = logging.getLogger("uvicorn.error")

torch.set_num_threads(min(4, os.cpu_count() or 1))
torch.set_num_interop_threads(min(4, os.cpu_count() or 1))

app = FastAPI(title="Qwen OpenAI-Compatible API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-translator-lovat-three.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001",
    ],
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
model.eval()
generation_lock = threading.Lock()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "qwen2.5-1.5b"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    input_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    user_text = " ".join(message["content"] for message in input_messages if message["role"] == "user")
    logger.info(
        "Solicitud recibida: mensajes=%d, user_chars=%d, user_start=%r, user_end=%r",
        len(input_messages),
        len(user_text),
        user_text[:80],
        user_text[-80:],
    )

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
    max_new_tokens = request.max_tokens or 2048

    if request.stream:
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **model_inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "repetition_penalty": 1.0,
            "pad_token_id": tokenizer.eos_token_id,
            "use_cache": True,
            "streamer": streamer,
        }

        def generate() -> None:
            with generation_lock, torch.inference_mode():
                model.generate(**generation_kwargs)

        generation_thread = threading.Thread(target=generate, daemon=True)
        generation_thread.start()

        def event_stream():
            accumulated = ""
            for chunk in streamer:
                if not chunk:
                    continue
                accumulated += chunk
                payload = {
                    "id": "chatcmpl-qwen-local",
                    "object": "chat.completion.chunk",
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            logger.info("Respuesta generada: %r", accumulated.strip())
            final_payload = {
                "id": "chatcmpl-qwen-local",
                "object": "chat.completion.chunk",
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logger.info("Respuesta generada: %r", response_text.strip())

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