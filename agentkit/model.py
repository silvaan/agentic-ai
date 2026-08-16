"""Modelo de linguagem e embeddings.

As duas classes do pacote estão aqui, e existem porque carregam modelo e
tokenizador. O resto do agentkit é função.
"""

from __future__ import annotations

import time

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

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Aplica o template de conversa às mensagens e gera a resposta."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate(prompt, **kwargs)

    def invoke(self, input: str | list[dict], **kwargs) -> str:
        """Encaminha string para generate e lista de mensagens para chat.

        É o método usado no restante do pacote, para que o agente e a memória
        não precisem saber qual das duas formas o usuário escolheu.
        """
        if isinstance(input, str):
            return self.generate(input, **kwargs)
        return self.chat(input, **kwargs)

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
                "Instale as dependências com: pip install -e \".[structured]\""
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
