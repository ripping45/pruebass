# Qwen Translator API

API compatible con el formato de Chat Completions de OpenAI para interpretar y traducir texto usando `Qwen/Qwen2.5-1.5B-Instruct`.

## Ejecutar localmente

```bash
docker compose up --build -d
```

La API queda disponible en `http://localhost:8000` y la documentacion interactiva en `http://localhost:8000/docs`.

## Endpoint

```text
POST /v1/chat/completions
Content-Type: application/json
```

No se requiere autenticacion.

## Peticion

```json
{
	"model": "qwen2.5-1.5b",
	"messages": [
		{
			"role": "system",
			"content": "Translate from English to formal neutral Latin American Spanish. Output only the translation."
		},
		{
			"role": "user",
			"content": "The patient needs to schedule a follow-up appointment for next Tuesday."
		}
	],
	"max_tokens": 256
}
```

### Campos

| Campo | Tipo | Requerido | Descripcion |
| --- | --- | --- | --- |
| `model` | string | No | Identificador que se devuelve en la respuesta. Valor recomendado: `qwen2.5-1.5b`. |
| `messages` | array | Si | Lista de mensajes con `role` y `content`. |
| `messages[].role` | string | Si | Normalmente `system` o `user`. |
| `messages[].content` | string | Si | Instrucciones o texto que se va a interpretar. |
| `max_tokens` | integer | No | Maximo de tokens generados. Valor por defecto: `512`. |
| `temperature` | number | No | Se acepta por compatibilidad, pero actualmente se ignora porque la API usa `do_sample=false`. |

## Reglas de `messages`

- Usa `role: "system"` para indicar idioma fuente, idioma destino y reglas adicionales.
- Usa `role: "user"` para enviar el texto exacto que se debe traducir.
- Si no envias un mensaje `system`, la API agrega automaticamente el prompt de interprete definido en `prompts.py`.
- No envies `source`, `target`, `lang` o `text` como campos independientes; la API actual procesa `messages`.
- La respuesta contiene unicamente la traduccion final, sin explicaciones.

## Ejemplo con cURL

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d '{
		"model": "qwen2.5-1.5b",
		"messages": [
			{
				"role": "system",
				"content": "Translate from Spanish to professional American English. Output only the translation."
			},
			{
				"role": "user",
				"content": "Necesito programar una cita de seguimiento para el proximo martes."
			}
		],
		"max_tokens": 64
	}'
```

## Respuesta

```json
{
	"id": "chatcmpl-qwen-local",
	"object": "chat.completion",
	"model": "qwen2.5-1.5b",
	"choices": [
		{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "I need to schedule a follow-up appointment for next Tuesday."
			},
			"finish_reason": "stop"
		}
	]
}
```

El texto traducido se obtiene de `choices[0].message.content`.

## URL publica de desarrollo

Cuando el tunel de GitHub Codespaces este activo, usa la URL publica con la misma ruta:

```text
https://<tu-subdominio>-8000.app.github.dev/v1/chat/completions
```

La URL publica puede cambiar al reiniciar el tunel.
