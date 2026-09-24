"""
LocalLLM Desktop Application
Main entry point for the application
"""
## 
## venv\Scripts\activate
##
import sys
import os
import logging
from pathlib import Path
import argparse
from dotenv import load_dotenv

# Adicionar diretório raiz ao path para imports
sys.path.insert(0, str(Path(__file__).parent))

from gui import LocalLLMApp
from utils import setup_logging, check_system_requirements


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='LocalLLM - Execute LLMs localmente no seu computador'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Caminho para o arquivo do modelo GGML/GGUF'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.env',
        help='Caminho para arquivo de configuração (padrão: config.env)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Habilitar modo debug com logs detalhados'
    )
    return parser.parse_args()


def check_dependencies():
    """Verifica se todas as dependências estão instaladas."""
    try:
        import psutil  # noqa: F401
    except ImportError:
        print("\nERRO: dependências faltando. Execute: pip install -r requirements.txt\n")
        sys.exit(1)

    try:
        import tkinter
    except ImportError:
        print("\n" + "="*70)
        print("ERRO: tkinter não está disponível!")
        print("="*70)
        print("\nNo Linux, instale com:")
        print("  sudo apt-get install python3-tk")
        print("\nNo Windows/Mac, tkinter já vem com Python.")
        print("="*70 + "\n")
        sys.exit(1)


def main():
    """Função principal da aplicação."""
    # Parse argumentos
    args = parse_arguments()
    
    # Configurar logging
    log_level = 'DEBUG' if args.debug else None
    setup_logging(log_level=log_level)
    
    logging.info("="*50)
    logging.info("Iniciando LocalLLM Desktop Application")
    logging.info("="*50)
    
    # Verificar dependências
    check_dependencies()

    # Carregar configurações do arquivo .env
    config_file = args.config
    if os.path.exists(config_file):
        load_dotenv(config_file)
        logging.info(f"Configurações carregadas de: {config_file}")
    else:
        logging.warning(f"Arquivo de configuração não encontrado: {config_file}")
        logging.info("Usando configurações padrão")

    server_path = os.getenv('LLAMA_SERVER_PATH', '')
    if not server_path or not os.path.exists(server_path):
        logging.warning(
            f"llama-server não encontrado ('{server_path}'). Baixe o llama.cpp em "
            "https://github.com/ggml-org/llama.cpp/releases e configure LLAMA_SERVER_PATH."
        )
    
    # Verificar requisitos do sistema
    requirements_met, warnings = check_system_requirements()
    if warnings:
        for warning in warnings:
            logging.warning(warning)
    
    # Obter model_path da linha de comando ou variável de ambiente
    model_path = args.model or os.getenv('MODEL_PATH', '')
    
    # Criar diretórios necessários
    Path('cache').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    Path('exports').mkdir(exist_ok=True)
    
    # Iniciar aplicação GUI
    try:
        app = LocalLLMApp(initial_model_path=model_path)
        logging.info("Interface gráfica inicializada")
        app.run()
    except KeyboardInterrupt:
        logging.info("Aplicação interrompida pelo usuário")
    except Exception as e:
        logging.error(f"Erro fatal: {e}", exc_info=True)
        raise
    finally:
        logging.info("Aplicação encerrada")


if __name__ == "__main__":
    main()