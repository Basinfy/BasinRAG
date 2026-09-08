import argparse
import sys
import asyncio
from basinrag.factory import BasinRAG

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def main():
    parser = argparse.ArgumentParser(description="BasinRAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingerir documentos")
    ingest_parser.add_argument("path", type=str, help="Caminho do arquivo ou diretório")
    
    # Query
    query_parser = subparsers.add_parser("query", help="Fazer uma busca topológica")
    query_parser.add_argument("question", type=str, help="Sua pergunta")
    query_parser.add_argument("--type", type=str, default="auto", choices=["auto", "local", "global", "hybrid"], help="Tipo de busca")
    query_parser.add_argument("--top-k", dest="top_k", type=int, default=5, help="Numero de passagens")

    # Chat
    chat_parser = subparsers.add_parser("chat", help="Iniciar chat interativo")

    # Serve
    serve_parser = subparsers.add_parser("serve", help="Iniciar servidor API")
    serve_parser.add_argument("--port", type=int, default=8000, help="Porta do servidor")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "serve":
        import uvicorn
        uvicorn.run("basinrag.api.server:app", host="0.0.0.0", port=args.port, reload=False)
        return

    # Comandos que precisam do RAG carregado
    rag = BasinRAG.create()

    if args.command == "ingest":
        print(f"Ingerindo {args.path}...")
        n = rag.ingest(args.path)
        if n == 0:
            print(f"❌ Nenhum nó gerado a partir de {args.path}. Verifique o caminho e os arquivos.")
            sys.exit(1)
        print(f"✅ Ingestão completa. {n} nós criados.")

        
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
            # LanÃ§a o daemon de L3 em background (nÃ£o bloqueia)
            bg_task = asyncio.create_task(rag.start_background_summarizer())
            
            while True:
                try:
                    q = await asyncio.to_thread(input, "\nVocê: ")
                    if q.lower() in ['sair', 'exit', 'quit']:
                        bg_task.cancel()
                        break
                    print("BasinRAG: ", end="", flush=True)
                    
                    async for token in rag.chat(q):
                        print(token, end="", flush=True)
                    print()
                except KeyboardInterrupt:
                    bg_task.cancel()
                    break
            
            # Salva o progresso ao sair
            rag.persistence.save_topology(rag.engine)
        
        asyncio.run(chat_loop())

if __name__ == "__main__":
    main()
