## Descrição da Alteração

Descreva brevemente o propósito desse Pull Request. Qual problema está sendo resolvido e qual a abordagem técnica adotada? 
Se este PR resolve alguma Issue específica, inclua o link (ex: `Resolves #12`).

## Tipo de Alteração

Marque com um `x` as opções aplicáveis:

- [ ] 🐛 Bug fix (correção não impactante que resolve um bug)
- [ ] ✨ Feature (nova funcionalidade não quebra a API atual)
- [ ] 💥 Breaking change (correção ou feature que altera a API e pode quebrar compatibilidade)
- [ ] 📝 Documentação (melhorias no README, docstrings ou diretório `docs/`)
- [ ] 🏎️ Performance (melhorias de eficiência, redução de latência/memória)
- [ ] 🧪 Testes (adição ou correção de testes)
- [ ] ♻️ Refatoração (alteração estrutural do código sem alteração de comportamento)

## Checklist de Qualidade

Antes de solicitar a revisão, verifique os pontos abaixo:

- [ ] Executei localmente `ruff check .` e `ruff format .`.
- [ ] Executei a verificação de tipos com `mypy basinrag` e não há erros.
- [ ] Adicionei ou atualizei os testes (Pytest) referentes a essa alteração.
- [ ] Todos os testes locais passaram (`pytest tests/ -v`).
- [ ] Minhas alterações não geram avisos de *deprecation* desnecessários.
- [ ] Atualizei a documentação correspondente, se necessário (incluindo assinaturas de funções e TypeHints).

## Notas para o Revisor (Opcional)

Adicione comentários sobre pontos críticos da implementação, decisões de design, trade-offs ou arquivos que exigem atenção especial.
