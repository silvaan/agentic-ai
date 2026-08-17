# IA Agêntica

Tópicos Especiais em Inteligência Computacional A · 2026.2

Disciplina de graduação sobre construção de agentes de IA, organizada em três
unidades. Na Unidade I o agente é implementado à mão, em Python, sobre um modelo
aberto executado localmente, para que nenhuma abstração apareça como caixa preta
antes de ter sido construída. A Unidade II reescreve o que foi feito usando
LangChain e LangGraph, com foco em confiabilidade. A Unidade III trata de
sistemas multiagente.

A ordem das aulas é de dependência, não de exposição. Ferramentas vêm antes de
planejamento porque formam o menor laço fechado que já muda o comportamento
observável do agente, e estado vem antes de planejamento porque replanejar exige
lembrar o que já foi tentado.

## Cronograma

**Unidade I — Anatomia de um Agente.** Python sobre modelo aberto local, sem
frameworks de agentes.

1. Panorama e loop mínimo de agente
2. Modelos de linguagem e tokenização
3. Geração e amostragem
4. Prompt engineering e contexto
5. Saída estruturada e validação
6. Ferramentas, function calling e composição de chamadas
7. Estado e memória
8. Planejamento e reflexão
9. Harness e engenharia de laço

**Unidade II — Agentes em Produção.** LangChain e LangGraph.

1. LangChain e LangGraph
2. Workflows com LangGraph
3. Agentes com LangGraph
4. RAG: chunking, indexação, hierárquico, pai e filho, graph RAG
5. RAG agêntico: roteamento, reescrita e correção
6. Integração de ferramentas e Model Context Protocol
7. Gestão de contexto em agentes de longa duração
8. Human in the loop
9. Guardrails e segurança
10. Observabilidade, avaliação e custo

**Unidade III — Sistemas Multiagente.**

1. Fundamentos e arquiteturas de coordenação
2. Supervisor na prática
3. Handoffs e swarm
4. CrewAI e abstrações de papel
5. Protocolo A2A e interoperabilidade
6. Debate, negociação e consenso
7. Simulação social e fronteiras
8. Avaliação e modos de falha de sistemas multiagente

## Conteúdo do repositório

Os notebooks das aulas ficam na raiz, numerados na ordem dos encontros. O
componente do dia é escrito do zero no notebook e, a partir do encontro
seguinte, passa a vir pronto da biblioteca.

O pacote `agentkit` é a biblioteca de referência da disciplina, construída ao
longo da Unidade I:

```text
agentkit/
├── model.py    modelo e embeddings com Hugging Face
├── tools.py    formato e execução de ferramentas
├── memory.py   memória com listas e dicionários
└── agent.py    laço do agente
```

O único backend é um modelo local do Hugging Face carregado com `transformers`.

## Uso

```python
from agentkit import LLM, run_agent, tool

@tool
def soma(a: int, b: int) -> int:
    """Add two integer numbers."""
    return a + b

llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", temperature=0.0)
resultado = run_agent(llm, "Use the soma tool to add 20 and 22.", tools=[soma])
print(resultado["answer"])
```
