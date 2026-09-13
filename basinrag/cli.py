import argparse
import sys
import asyncio
import os
from basinrag import __version__
from basinrag.factory import BasinRAG

try:
    reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
    if reconfigure_stdout:
        reconfigure_stdout(encoding="utf-8")
    reconfigure_stdin = getattr(sys.stdin, "reconfigure", None)
    if reconfigure_stdin:
        reconfigure_stdin(encoding="utf-8")
except Exception:
    pass

def main():
    parser = argparse.ArgumentParser(description="BasinRAG CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingerir documentos (merge incremental)")
    ingest_parser.add_argument("path", type=str, help="Caminho do arquivo ou diretório")
    ingest_parser.add_argument("--storage-dir", default=None, help="Destino do índice (padrão: .basinrag-v3)")

    reindex_parser = subparsers.add_parser("reindex", help="Reconstruir o indice do zero")
    reindex_parser.add_argument("path", type=str, help="Caminho do arquivo ou diretório")
    reindex_parser.add_argument("--storage-dir", default=None, help="Destino novo/vazio do snapshot v3 (padrão: .basinrag-v3)")

    sync_parser = subparsers.add_parser(
        "sync", help="Sincronizar fontes: substituir arquivos alterados e remover os ausentes"
    )
    sync_parser.add_argument("path", type=str, help="Diretório ou arquivo a sincronizar")
    sync_parser.add_argument("--storage-dir", default=None, help="Raiz do snapshot v3")
    
    # Query
    query_parser = subparsers.add_parser("query", help="Fazer uma busca topológica")
    query_parser.add_argument("question", type=str, help="Sua pergunta")
    query_parser.add_argument("--type", type=str, default="auto", choices=["auto", "local", "global", "hybrid"], help="Tipo de busca")
    query_parser.add_argument("--top-k", dest="top_k", type=int, default=5, help="Numero de passagens")
    query_parser.add_argument("--storage-dir", default=None, help="Raiz do snapshot v3")

    # Chat
    chat_parser = subparsers.add_parser("chat", help="Iniciar chat interativo")
    chat_parser.add_argument("--storage-dir", default=None, help="Raiz do snapshot v3")
    chat_parser.add_argument("--enable-background-l3", action="store_true", help="Ativar sumarização L3 em background")
    chat_parser.add_argument("--allow-remote-l3-egress", action="store_true", help="Permitir envio de trechos ao provedor LLM remoto para L3")

    # Serve
    serve_parser = subparsers.add_parser("serve", help="Iniciar servidor API")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host do servidor")
    serve_parser.add_argument("--port", type=int, default=8000, help="Porta do servidor")
    serve_parser.add_argument("--storage-dir", default=None, help="Raiz do snapshot v3")
    serve_parser.add_argument("--enable-background-l3", action="store_true", help="Ativar sumarização L3 em background")
    serve_parser.add_argument("--allow-remote-l3-egress", action="store_true", help="Permitir envio de trechos ao provedor LLM remoto para L3")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "serve":
        import uvicorn
        from basinrag.api.server import require_api_key_for_public_bind
        try:
            require_api_key_for_public_bind(args.host)
        except RuntimeError as exc:
            print(exc)
            sys.exit(2)
        if args.storage_dir:
            os.environ["BASINRAG_STORAGE_DIR"] = args.storage_dir
        if args.enable_background_l3:
            os.environ["BASINRAG_ENABLE_BACKGROUND_L3"] = "true"
        if args.allow_remote_l3_egress:
            os.environ["BASINRAG_ALLOW_REMOTE_L3_EGRESS"] = "true"
        uvicorn.run(
            "basinrag.api.server:app", host=args.host, port=args.port, reload=False,
            ws_max_size=8192,
        )
        return

    # Comandos que precisam do RAG carregado
    overrides = {}
    if getattr(args, "storage_dir", None):
        overrides["storage_dir"] = args.storage_dir
    for key in ("enable_background_l3", "allow_remote_l3_egress"):
        if getattr(args, key, False):
            overrides[key] = True
    rag = BasinRAG.create(load_existing_index=args.command != "reindex", **overrides)

    if args.command == "ingest":
        print(f"Ingerindo {args.path}...")
        n = rag.ingest(args.path)
        if n == 0:
            print(f"❌ Nenhum nó gerado a partir de {args.path}. Verifique o caminho e os arquivos.")
            sys.exit(1)
        print(f"✅ Ingestão completa. {n} nós criados.")

    elif args.command == "reindex":
        print(f"Reindexando a partir de {args.path}...")
        n = rag.reindex(args.path)
        if n == 0:
            print(f"❌ Nenhum nó gerado a partir de {args.path}. Verifique o caminho e os arquivos.")
            sys.exit(1)
        print(f"✅ Reindexação completa. {n} nós criados.")

    elif args.command == "sync":
        print(f"Sincronizando {args.path}...")
        n = rag.sync(args.path)
        print(f"✅ Sincronização concluída. {n} nós ingeridos no snapshot atualizado.")

    elif args.command == "query":
        if not rag._loaded:
            print("Nenhum indice carregado. Rode: basinrag ingest <pasta>")
            sys.exit(1)
        docs = rag.query(args.question, search_type=args.type, top_k=args.top_k)
        if not docs:
            print("Nenhum trecho recuperado.")
            return
        for i, d in enumerate(docs):
            text = d.page_content if hasattr(d, "page_content") else str(d)
            if not text.strip():
                continue
            print(f"\n--- Resultado {i+1} ---")
            print(text)
            
    elif args.command == "chat":
        if not rag._loaded:
            print("Nenhum indice carregado. Rode: basinrag ingest <pasta>")
            sys.exit(1)
        print("🤖 Chat BasinRAG Iniciado! (Digite 'sair' para encerrar)")
        
        async def chat_loop():
            # Lança o daemon de L3 em background (não bloqueia)
            bg_task = None
            if rag.config.enable_background_l3 and (
                rag.config.provider != "openai" or rag.config.allow_remote_l3_egress
            ):
                bg_task = asyncio.create_task(rag.start_background_summarizer())
            
            while True:
                try:
                    q = await asyncio.to_thread(input, "\nVocê: ")
                    if q.lower() in ['sair', 'exit', 'quit']:
                        if bg_task:
                            bg_task.cancel()
                        break
                    print("BasinRAG: ", end="", flush=True)
                    
                    async for token in rag.chat(q):
                        print(token, end="", flush=True)
                    print()
                except KeyboardInterrupt:
                    if bg_task:
                        bg_task.cancel()
                    break
            
            # Salva o progresso ao sair
            rag.persistence.save_topology(rag.engine)
        
        asyncio.run(chat_loop())

if __name__ == "__main__":
    main()
