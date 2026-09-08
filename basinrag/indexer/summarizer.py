import asyncio
import re
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from ..core.topology import BasinTopologyEngine
from ..core.llm import UniversalLLM
from ..logging_config import setup_logging

logger = setup_logging()


class BasinSummarySchema(BaseModel):
    title: str = Field(default="", description="Título curto e descritivo da bacia temática")
    themes: List[str] = Field(default_factory=list, description="Lista de tópicos-chave")
    entities: List[str] = Field(default_factory=list, description="Entidades nomeadas mencionadas")
    summary: str = Field(default="", description="Resumo conciso de 2-3 frases")


class BasinCritiqueSchema(BaseModel):
    critique: str = Field(default="", description="Análise crítica sobre precisão e ausência de alucinações")
    score: int = Field(default=5, ge=1, le=10, description="Nota de qualidade de 1 a 10")


def extract_json_payload(raw_text: str) -> Optional[dict]:
    """Extrator de JSON resiliente em 3 estágios para LLMs locais."""
    if not raw_text:
        return None
    try:
        return json.loads(raw_text.strip())
    except Exception:
        pass
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    first_brace = raw_text.find('{')
    last_brace = raw_text.rfind('}')
    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(raw_text[first_brace:last_brace + 1])
        except Exception:
            pass
    return None


class BasinSummarizer:
    """Gerador de resumos L3 Agentic com loop de Refinamento e Crítica."""
    
    def __init__(
        self,
        engine: BasinTopologyEngine,
        provider: str = "ollama",
        model_name: str = "qwen2.5",
    ):
        self.engine = engine
        self.provider = provider
        self.model_name = model_name
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = UniversalLLM(provider=self.provider, model_name=self.model_name)
        return self._llm
        
    def _extract_score(self, critique: str) -> int:
        """Extrai o score (1-10) da crítica com suporte a JSON e regex."""
        payload = extract_json_payload(critique)
        if payload and isinstance(payload, dict) and "score" in payload:
            try:
                return min(10, max(1, int(payload["score"])))
            except Exception:
                pass
        matches = re.findall(r'SCORE:\s*(\d+)', critique, re.IGNORECASE)
        if matches:
            return min(10, max(1, int(matches[-1])))
        matches = re.findall(r'score.*?(\d+)', critique, re.IGNORECASE)
        if matches:
            return min(10, max(1, int(matches[-1])))
        return 5


    async def _draft(self, texts: str) -> str:
        prompt = (
            "Você é um analista especialista em extração de informações.\n"
            "IMPORTANTE: Você DEVE gerar o resumo estritamente no MESMO IDIOMA predominante dos textos fornecidos.\n"
            "Analise os seguintes textos de uma comunidade de documentos e crie um resumo estruturado no EXATO formato:\n\n"
            "TÍTULO: [título curto no idioma dos textos]\n"
            "TEMAS: [tema1, tema2, ...]\n"
            "ENTIDADES: [entidade1, entidade2, ...]\n"
            "RESUMO: [2-3 frases detalhando o conteúdo exclusivo desta comunidade no mesmo idioma dos textos]\n\n"
            f"TEXTOS BRUTOS:\n{texts}"
        )
        return await self.llm.generate("Siga o formato exigido rigorosamente e preserve o idioma original dos textos.", prompt)

    async def _critique(self, texts: str, draft: str) -> tuple[int, str]:
        prompt = (
            "Avalie o DRAFT do resumo com base nos TEXTOS BRUTOS.\n"
            "O resumo captura todos os temas importantes? Ele foca no que é único e evita superficialidades?\n"
            "Há alucinações (fatos inventados)?\n\n"
            "Forneça sua crítica e no final inclua uma linha exata no formato 'SCORE: X' onde X é de 1 a 10.\n\n"
            f"TEXTOS BRUTOS:\n{texts}\n\n"
            f"DRAFT ATUAL:\n{draft}"
        )
        critique = await self.llm.generate("Você é um crítico rigoroso.", prompt)
        score = self._extract_score(critique)
        return score, critique

    async def _refine(self, texts: str, draft: str, critique: str) -> str:
        prompt = (
            "Melhore o DRAFT do resumo usando o FEEDBACK DO CRÍTICO.\n"
            "Corrija os problemas apontados, preserve rigorosamente o IDIOMA dos textos de origem, e MANTENHA EXATAMENTE o formato estruturado:\n"
            "TÍTULO: ...\nTEMAS: ...\nENTIDADES: ...\nRESUMO: ...\n\n"
            f"TEXTOS BRUTOS:\n{texts}\n\n"
            f"DRAFT ANTERIOR:\n{draft}\n\n"
            f"FEEDBACK DO CRÍTICO:\n{critique}"
        )
        return await self.llm.generate("Aja como um editor final de altíssima qualidade mantendo o idioma dos textos.", prompt)

    async def summarize_basin(self, basin_id: str, verbose: bool = False) -> str:
        """Loop Agentic para resumir uma bacia: Draft -> (Critique -> Refine)."""
        basin = self.engine.basins.get(basin_id)
        if not basin:
            return ""
        
        # Coleta textos dos nós (max 10)
        texts_list = []
        for node_id, data in sorted(
            basin.rho_tree.nodes(data=True),
            key=lambda x: x[1].get('hops', 999)
        ):
            text = data.get('text', '')
            if text:
                texts_list.append(text[:500])
            if len(texts_list) >= 10:
                break
        
        if not texts_list:
            return ""
            
        combined_texts = "\n---\n".join(texts_list)
        
        # 1. Draft
        draft = await self._draft(combined_texts)
        
        # Loop de Refinamento (max 2 iterações)
        for i in range(2):
            score, critique = await self._critique(combined_texts, draft)
            if verbose:
                print(f"    [Bacia {basin_id[:4]} | Iter {i+1}] Critica Score: {score}/10", flush=True)
            
            if score >= 8:
                break # Está bom o suficiente
                
            # 3. Refine
            draft = await self._refine(combined_texts, draft, critique)
            
        return draft.strip()
    
    async def summarize_all(self, concurrency: int = 3):
        """Orquestra a sumarização de todas as bacias paralelamente."""
        logger.info(f"🤖 Gerando Resumos L3 Agentic ({len(self.engine.basins)} bacias)...")
        semaphore = asyncio.Semaphore(concurrency)
        
        async def _process(basin_id):
            async with semaphore:
                try:
                    summary = await self.summarize_basin(basin_id)
                    if basin_id in self.engine.basins:
                        basin = self.engine.basins[basin_id]
                        if basin.rho_tree.has_node(basin_id):
                            basin.rho_tree.nodes[basin_id]['l3_summary'] = summary
                            basin.rho_tree.nodes[basin_id]['l3_source'] = 'llm'
                    # Extract just the title for printing
                    title_match = re.search(r'TÍTULO:\s*(.*?)\n', summary, re.IGNORECASE)
                    title = title_match.group(1)[:40] if title_match else "Sem Título"
                    logger.info(f"  ✅ L3 {basin_id[:4]} concluído: {title}")
                except Exception:
                    logger.exception(f"  ⚠️ Falha no L3 da Bacia {basin_id[:4]}")
        
        tasks = [_process(bid) for bid in self.engine.basins]
        await asyncio.gather(*tasks)
        logger.info("✅ Todos os resumos L3 Agentic gerados e anexados ao grafo!")

    async def summarize_missing_background(self, persistence=None, interval: float = 2.0, verbose: bool = False):
        """
        Background Daemon: gera resumos L3 apenas para bacias que ainda não têm.
        Roda silenciosamente enquanto o usuário interage com o chat.
        Salva o grafo incrementalmente após cada bacia concluída.
        """
        missing = [
            bid for bid, basin in self.engine.basins.items()
            if basin.rho_tree.has_node(bid)
            and basin.rho_tree.nodes[bid].get("l3_source") != "llm"
        ]
        
        if not missing:
            return
            
        total = len(missing)
        if verbose:
            logger.info(f"🔄 Background: {total} bacias sem resumo L3. Gerando em standby...")
            
        for i, basin_id in enumerate(missing):
            try:
                summary = await self.summarize_basin(basin_id, verbose=verbose)
                if basin_id in self.engine.basins:
                    basin = self.engine.basins[basin_id]
                    if basin.rho_tree.has_node(basin_id):
                        basin.rho_tree.nodes[basin_id]['l3_summary'] = summary
                        basin.rho_tree.nodes[basin_id]['l3_source'] = 'llm'
                
                if verbose:
                    title_match = re.search(r'TÍTULO:\s*(.*?)\n', summary, re.IGNORECASE)
                    title = title_match.group(1)[:40] if title_match else "OK"
                    logger.info(f"  🧠 [{i+1}/{total}] L3 gerado: {title}")
                
                # Salva incrementalmente para não perder progresso
                if persistence and (i + 1) % 5 == 0:
                    persistence.save_topology(self.engine)
                    if verbose:
                        logger.info(f"  💾 Progresso salvo ({i+1}/{total})")
                    
            except Exception:
                if verbose:
                    logger.exception(f"  ⚠️ Background L3 {basin_id[:4]} falhou")
            
            # Respira entre bacias para não travar o chat do usuário
            await asyncio.sleep(interval)
        
        # Salva final
        if persistence:
            persistence.save_topology(self.engine)
            if verbose:
                logger.info(f"✅ Background completo! {total} resumos L3 gerados e salvos.")
