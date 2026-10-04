"""Modelos de linguagem e embeddings.

LLM carrega o modelo na máquina com transformers, e LLMAPI conversa com um
endpoint no formato chat completions da OpenAI. As duas têm a mesma superfície,
e é por isso que o agente e a memória funcionam com qualquer uma delas sem saber
de onde vem a resposta. O resto do agentkit é função.
"""

from __future__ import annotations

import copy
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


class LLM:
    """Modelo de linguagem local, carregado com transformers."""

    def __init__(
        self,
        model: str,
        device: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.weights = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.weights.eval()
        # Nem todo tokenizador define token de preenchimento; o de fim serve.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tools: list[dict] = []
        self.last_usage: dict = {}
        self.usage: list[dict] = []

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Gera texto a partir de uma string, sem aplicar template de conversa.

        Toda geração preenche last_usage e acrescenta um item a usage. Esse
        registro de tokens e de tempo atravessa a unidade inteira.
        """
        temperature = self.temperature if temperature is None else temperature
        top_p = self.top_p if top_p is None else top_p
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        sampling = temperature > 0

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.weights.device)
        started = time.perf_counter()
        with torch.no_grad():
            output = self.weights.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=sampling,
                temperature=temperature if sampling else None,
                top_p=top_p if sampling else None,
                # O modelo publica top_k no generation_config, e sem amostragem
                # ele vira um argumento inválido que o transformers avisa.
                top_k=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        self.last_usage = {
            "tokens_in": int(inputs["input_ids"].shape[-1]),
            "tokens_out": int(new_tokens.shape[-1]),
            "seconds": round(time.perf_counter() - started, 3),
            "model": self.model,
        }
        self.usage.append(self.last_usage)
        return text

    def chat(self, messages: list[dict], **kwargs) -> str | dict:
        """Aplica o template de conversa às mensagens e gera a resposta.

        Com ferramentas ligadas por bind_tools, devolve a mensagem do assistente
        em vez do texto, com a chave tool_calls quando o modelo pede uma chamada.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tools=self.tools or None, tokenize=False, add_generation_prompt=True
        )
        text = self.generate(prompt, **kwargs)
        if not self.tools:
            return text
        calls = parse_tool_calls(text)
        if not calls:
            return {"role": "assistant", "content": text}
        return {
            "role": "assistant",
            "content": text.split("<tool_call>")[0].strip(),
            "tool_calls": calls,
        }

    def invoke(self, input: str | list[dict], **kwargs) -> str | dict:
        """Encaminha string para generate e lista de mensagens para chat.

        É o método usado no restante do pacote, para que o agente e a memória
        não precisem saber qual das duas formas o usuário escolheu.
        """
        if isinstance(input, str):
            return self.generate(input, **kwargs)
        return self.chat(input, **kwargs)

    def bind_tools(self, tools: list) -> "LLM":
        """Devolve uma cópia do modelo com as ferramentas ligadas às chamadas."""
        bound = copy.copy(self)
        bound.tools = [fn.tool_schema for fn in tools]
        return bound

    def generate_structured(
        self,
        prompt_or_messages: str | list[dict],
        schema: type,
        max_tokens: int | None = None,
    ) -> object:
        """Gera uma saída forçada pelo esquema e devolve o objeto validado."""
        try:
            from outlines import Generator, from_transformers
        except ImportError as exc:
            raise ImportError(
                "Instale o outlines com: pip install \"outlines>=1,<2\""
            ) from exc

        if isinstance(prompt_or_messages, str):
            prompt = prompt_or_messages
        else:
            prompt = self.tokenizer.apply_chat_template(
                prompt_or_messages, tokenize=False, add_generation_prompt=True
            )

        started = time.perf_counter()
        model = from_transformers(self.weights, self.tokenizer)
        generator = Generator(model, schema)
        text = generator(
            prompt,
            max_new_tokens=max_tokens or self.max_tokens,
            do_sample=False,
        )
        self.last_usage = {
            "tokens_in": len(self.tokenizer.encode(prompt)),
            "tokens_out": len(self.tokenizer.encode(text)),
            "seconds": round(time.perf_counter() - started, 3),
            "model": self.model,
        }
        self.usage.append(self.last_usage)
        return schema.model_validate_json(text)


def parse_tool_calls(text: str) -> list[dict]:
    """Lê os blocos tool_call do texto, no formato estilo Hermes usado pelo Qwen.

    Lista vazia é o sinal de que o texto é a resposta final.
    """
    calls = []
    for block in text.split("<tool_call>")[1:]:
        try:
            call = json.loads(block.split("</tool_call>")[0])
        except json.JSONDecodeError:
            continue
        if isinstance(call.get("name"), str):
            calls.append({"name": call["name"], "arguments": call.get("arguments", {})})
    return calls


OPENAI_BASE_URL = "https://api.openai.com/v1"


class LLMAPI:
    """Modelo servido por uma API no formato chat completions da OpenAI.

    Tem a mesma superfície do LLM — invoke, bind_tools, generate_structured e o
    registro em last_usage — para que o agente e a memória não precisem saber de
    onde vem a resposta.

    O formato não é um padrão de comitê: é a API da OpenAI, que os demais
    provedores passaram a imitar. Por isso apontar para outro serviço é questão
    de trocar a base_url e a chave.

        OpenAI   https://api.openai.com/v1
        Groq     https://api.groq.com/openai/v1

    Qualquer servidor que fale o mesmo formato serve, inclusive um rodando na
    própria máquina, como Ollama ou vLLM. Servidores locais costumam ignorar a
    chave, mas o argumento continua obrigatório: passe qualquer texto.

    A conversa viaja num formato mais verboso que o do curso, e as funções
    to_api_messages e to_api_tools fazem a tradução. Três diferenças importam: a
    chamada de ferramenta chega dentro de tool_calls com um identificador, os
    argumentos viajam como string JSON em vez de objeto, e a observação precisa
    apontar de volta para a chamada pelo tool_call_id.

    A chave vem do argumento api_key. A variável de ambiente OPENAI_API_KEY
    só é consultada quando a base_url é a da OpenAI, para que a chave de um
    provedor nunca seja enviada a outro.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = OPENAI_BASE_URL,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        parallel_tool_calls: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        # A variável de ambiente vale só para a OpenAI. Sem essa condição, apontar
        # para outro provedor sem informar a chave enviaria a chave da OpenAI para
        # ele, que a recusaria depois de já tê-la recebido.
        if api_key is None and self.base_url == OPENAI_BASE_URL:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(f"Informe api_key para {self.base_url}.")
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        # O laço do agente executa uma ferramenta por passo, e este parâmetro
        # pede o mesmo ao provedor, em vez de descartar as chamadas extras.
        self.parallel_tool_calls = parallel_tool_calls
        self.tools: list[dict] = []
        self.last_usage: dict = {}
        self.usage: list[dict] = []

    def complete(self, messages: list[dict], response_format: dict | None = None, **kwargs) -> dict:
        """Envia a conversa e devolve a mensagem crua da API, registrando o uso.

        É o único ponto da classe que fala HTTP. Os outros métodos montam o que
        entra aqui e traduzem o que sai.
        """
        payload = {
            "model": self.model,
            "messages": to_api_messages(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        if self.tools:
            payload["tools"] = to_api_tools(self.tools)
            payload["parallel_tool_calls"] = self.parallel_tool_calls
        if response_format is not None:
            payload["response_format"] = response_format

        started = time.perf_counter()
        data = post(f"{self.base_url}/chat/completions", payload, self.api_key)
        usage = data.get("usage") or {}
        self.last_usage = {
            "tokens_in": usage.get("prompt_tokens", 0),
            "tokens_out": usage.get("completion_tokens", 0),
            "seconds": round(time.perf_counter() - started, 3),
            "model": self.model,
        }
        self.usage.append(self.last_usage)
        return data["choices"][0]["message"]

    def generate(self, prompt: str, **kwargs) -> str:
        """Gera texto a partir de uma string.

        A API não continua prompt cru: o mais próximo disso é uma conversa de
        uma mensagem só, e é essa a diferença para o LLM local.
        """
        return self.complete([{"role": "user", "content": prompt}], **kwargs).get("content") or ""

    def chat(self, messages: list[dict], **kwargs) -> str | dict:
        """Envia a conversa e devolve a resposta.

        Com ferramentas ligadas por bind_tools, devolve a mensagem do assistente
        em vez do texto, com a chave tool_calls quando o modelo pede uma chamada.
        """
        message = self.complete(messages, **kwargs)
        if not self.tools:
            return message.get("content") or ""
        return from_api_message(message)

    def invoke(self, input: str | list[dict], **kwargs) -> str | dict:
        """Encaminha string para generate e lista de mensagens para chat."""
        if isinstance(input, str):
            return self.generate(input, **kwargs)
        return self.chat(input, **kwargs)

    def bind_tools(self, tools: list) -> "LLMAPI":
        """Devolve uma cópia do modelo com as ferramentas ligadas às chamadas."""
        bound = copy.copy(self)
        bound.tools = [fn.tool_schema for fn in tools]
        return bound

    def generate_structured(
        self,
        prompt_or_messages: str | list[dict],
        schema: type,
        max_tokens: int | None = None,
    ) -> object:
        """Gera uma saída forçada pelo esquema e devolve o objeto validado.

        O mecanismo é outro: no LLM local a gramática restringe a amostragem, e
        aqui o esquema vai no response_format e quem restringe é o provedor.
        """
        messages = (
            [{"role": "user", "content": prompt_or_messages}]
            if isinstance(prompt_or_messages, str)
            else prompt_or_messages
        )
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": to_api_schema(schema),
                "strict": True,
            },
        }
        message = self.complete(
            messages, response_format=response_format, max_tokens=max_tokens or self.max_tokens
        )
        return schema.model_validate_json(message["content"])


def post(url: str, payload: dict, api_key: str) -> dict:
    """Envia o JSON e devolve a resposta, com a biblioteca padrão.

    Sem SDK de propósito: o corpo da requisição é o mesmo que a documentação do
    provedor mostra, e dá para ler o que sai daqui. O User-Agent é obrigatório
    porque provedores atrás de CDN recusam o que o urllib manda por padrão.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "agentkit",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{error.code} {error.reason}: {error.read().decode('utf-8')}") from error


def to_api_tools(schemas: list[dict]) -> list[dict]:
    """Traduz o esquema de ferramenta do curso para o formato de function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": {
                    "type": "object",
                    "properties": schema["parameters"],
                    "required": schema["required"],
                },
            },
        }
        for schema in schemas
    ]


def to_api_messages(messages: list[dict]) -> list[dict]:
    """Traduz as mensagens do curso para o formato da API.

    A API exige que cada mensagem tool aponte para a chamada que a originou. O
    formato do curso não guarda esse identificador, então ele é criado aqui, por
    posição, e só precisa ser consistente dentro desta requisição.
    """
    api_messages = []
    pending: list[str] = []
    for message in messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                identifier = f"call_{len(pending) + len(calls)}"
                calls.append(
                    {
                        "id": identifier,
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call.get("arguments", {})),
                        },
                    }
                )
            pending.extend(call["id"] for call in calls)
            api_messages.append(
                {"role": "assistant", "content": message.get("content") or "", "tool_calls": calls}
            )
        elif message["role"] == "tool":
            if not pending:
                raise ValueError("mensagem tool sem chamada correspondente no histórico")
            api_messages.append(
                {"role": "tool", "tool_call_id": pending.pop(0), "content": message["content"]}
            )
        else:
            api_messages.append({"role": message["role"], "content": message["content"]})
    return api_messages


def from_api_message(message: dict) -> dict:
    """Traduz a mensagem do assistente devolvida pela API para o formato do curso."""
    calls = message.get("tool_calls") or []
    assistant = {"role": "assistant", "content": message.get("content") or ""}
    if not calls:
        return assistant
    assistant["tool_calls"] = [
        {
            "name": call["function"]["name"],
            "arguments": json.loads(call["function"]["arguments"] or "{}"),
        }
        for call in calls
    ]
    return assistant


def to_api_schema(schema: type) -> dict:
    """Devolve o esquema JSON do Pydantic no formato que a saída estruturada exige.

    O modo estrito do provedor pede que todo objeto proíba campos extras e
    declare todos os seus campos como obrigatórios, coisas que o Pydantic não
    escreve sozinho.
    """
    def fix(node):
        if isinstance(node, list):
            return [fix(item) for item in node]
        if not isinstance(node, dict):
            return node
        node = {key: fix(value) for key, value in node.items()}
        if node.get("type") == "object":
            node["additionalProperties"] = False
            node["required"] = list(node.get("properties", {}))
        return node

    return fix(schema.model_json_schema())


class Embeddings:
    """Codificador de textos em vetores, com pooling médio pela máscara de atenção."""

    def __init__(self, model: str, device: str | None = None) -> None:
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.weights = AutoModel.from_pretrained(model).to(self.device)
        self.weights.eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Devolve uma matriz com um vetor por texto.

        O pooling é a média dos estados escondidos ponderada pela máscara, para
        que o preenchimento não entre na conta. Não usamos sentence-transformers
        justamente para que essa etapa fique visível.
        """
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.no_grad():
            output = self.weights(**batch)
        mask = batch["attention_mask"].unsqueeze(-1).float()
        summed = (output.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return (summed / counts).cpu().numpy()

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Interface de embeddings do LangChain: um vetor por texto, em lotes.

        Com estes dois métodos a classe pode ser passada a um vector store do
        LangChain, como o Chroma, no lugar dos embeddings de um provedor.
        """
        vectors = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(self.embed(texts[start:start + batch_size]).tolist())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Interface de embeddings do LangChain: o vetor de um único texto."""
        return self.embed([text])[0].tolist()
