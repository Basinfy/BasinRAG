# Diretrizes de Contribuição - BasinRAG

Obrigado pelo seu interesse em contribuir para o **BasinRAG**! Nosso objetivo é construir uma solução de código aberto de excelência técnica, com arquitetura robusta e documentação rigorosa.

## Configuração do Ambiente de Desenvolvimento

Para começar a desenvolver, faça um fork e clone o repositório, em seguida, configure seu ambiente virtual e instale as dependências:

```bash
git clone https://github.com/SEU_USUARIO/BasinRAG.git
cd BasinRAG
python -m venv venv
source venv/bin/activate  # ou no Windows: venv\Scripts\activate
pip install -e ".[dev,api,ollama]"
```

## Padrões de Código e Linting

Mantemos altos padrões de qualidade de código. Todo o código deve ser formatado e validado usando a ferramenta **Ruff**:

- Para verificar inconsistências de linting:
  ```bash
  ruff check .
  ```
- Para aplicar a formatação padrão em todo o repositório:
  ```bash
  ruff format .
  ```

## Verificação de Tipos Estática

O BasinRAG faz uso rigoroso de anotações de tipo (`typing`) para garantir consistência. Utilizamos o **Mypy**:

```bash
mypy basinrag
```

Nenhum aviso ou erro do Mypy deve ser introduzido em novos Pull Requests.

## Execução de Testes

Os testes são essenciais para garantir que a topologia e as lógicas de recuperação não sejam corrompidas. Utilizamos **Pytest**:

```bash
# Para rodar toda a suíte de testes com relatório de cobertura
pytest tests/ -v --cov=basinrag
```

Garantimos que a cobertura de testes continue crescendo. Por favor, adicione testes unitários para qualquer nova funcionalidade.

## Convenção de Commits Semânticos

Seguimos a convenção de [Conventional Commits](https://www.conventionalcommits.org/). Os prefixos aceitos são:

- `feat:` Novas funcionalidades ou capacidades.
- `fix:` Correção de bugs.
- `docs:` Alterações e melhorias na documentação.
- `perf:` Alterações focadas em melhoria de performance (tempo, uso de memória).
- `test:` Adição ou refatoração de testes.
- `refactor:` Alterações no código que não adicionam funcionalidades nem corrigem bugs, mas melhoram a estrutura.

## Processo de Pull Request

1. **Branching**: Crie um branch a partir da branch `main` com um nome descritivo (ex: `feat/improve-rrf-fusion` ou `fix/tokenizer-leak`).
2. **Implementação**: Escreva seu código mantendo a formatação (Ruff) e as anotações de tipo (Mypy).
3. **Testes**: Assegure-se de que a cobertura de testes não diminuiu (`pytest tests/`).
4. **Checklist de PR**: 
   - A descrição do PR explica claramente o problema resolvido ou a feature adicionada.
   - Foram adicionados testes suficientes.
   - A documentação (Docstrings e manuais) foi atualizada de acordo.
   - Os commits respeitam as convenções estabelecidas.

## Código de Conduta e Comunicação

Espera-se que toda comunicação, seja em *Issues*, *Pull Requests* ou discussões técnicas, mantenha o máximo de respeito e profissionalismo. Focamos em:
- **Crítica Construtiva**: Ataque o problema, não a pessoa.
- **Rigor Técnico**: Baseie sugestões de arquitetura em evidências e benchmarks.
- **Inclusão**: Seja paciente com novos contribuidores e ofereça mentoria nas revisões de código.

Agradecemos sua colaboração e estamos ansiosos para revisar suas contribuições!
