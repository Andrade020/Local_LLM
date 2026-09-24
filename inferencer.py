"""
Wrapper de inferência sobre o llama-server (llama.cpp)
Sobe o servidor como subprocesso e conversa via API de chat com streaming
"""

import json
import logging
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from model_manager import ModelManager


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


class LLMInferencer:
    """
    Executa o modelo com o llama-server.exe do llama.cpp.

    Usar o binário oficial (em vez do llama-cpp-python) dá suporte imediato a
    arquiteturas novas (ex.: Qwen3.6 MoE), CUDA em GPUs antigas (Pascal) e
    offload de experts MoE para a CPU (--n-cpu-moe).
    """

    def __init__(self, model_manager: ModelManager):
        """
        Inicializa o inferencer.

        Args:
            model_manager: Instância do ModelManager com modelo validado
        """
        self.model_manager = model_manager
        self.process: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self.n_ctx = 0
        self.is_loaded = False
        self.last_timings: Dict = {}
        self._log_file = None

        logging.info("LLMInferencer inicializado")

    def load_model(
        self,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        n_cpu_moe: Optional[int] = None,
        timeout: int = 600,
    ):
        """
        Sobe o llama-server com o modelo e espera ele ficar pronto.

        Valores não informados vêm do config.env (N_CTX, N_THREADS,
        N_GPU_LAYERS, N_CPU_MOE).

        Args:
            n_ctx: Tamanho do contexto (tokens)
            n_threads: Número de threads da CPU
            n_gpu_layers: Camadas na GPU (99 = todas)
            n_cpu_moe: Camadas cujos experts MoE ficam na CPU (0 = nenhuma)
            timeout: Segundos máximos esperando o servidor subir
        """
        server = Path(os.getenv('LLAMA_SERVER_PATH', 'llama-server.exe'))
        if not server.exists():
            raise FileNotFoundError(
                f"llama-server não encontrado em: {server}\n"
                f"Configure LLAMA_SERVER_PATH no config.env."
            )

        self.n_ctx = n_ctx or _env_int('N_CTX', 8192)
        n_threads = n_threads or _env_int('N_THREADS', os.cpu_count() or 4)
        n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else _env_int('N_GPU_LAYERS', 0)
        n_cpu_moe = n_cpu_moe if n_cpu_moe is not None else _env_int('N_CPU_MOE', 0)
        self.port = self._free_port()

        cmd = [
            str(server),
            '-m', str(self.model_manager.model_path),
            '--host', '127.0.0.1',
            '--port', str(self.port),
            '-c', str(self.n_ctx),
            '-t', str(n_threads),
            '-ngl', str(n_gpu_layers),
            '-fa', 'on',
            '-np', '1',
            '--no-webui',
        ]
        if n_cpu_moe > 0:
            cmd += ['--n-cpu-moe', str(n_cpu_moe)]
        cmd += os.getenv('LLAMA_SERVER_EXTRA_ARGS', '').split()

        Path('logs').mkdir(exist_ok=True)
        self._log_file = open(Path('logs') / 'llama-server.log', 'w', encoding='utf-8')
        logging.info(f"Iniciando llama-server: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )

        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.process.poll() is not None:
                self._cleanup()
                raise RuntimeError(
                    "O llama-server encerrou durante o carregamento.\n"
                    f"Últimas linhas do log:\n{self._log_tail()}"
                )
            if self._healthy():
                self.is_loaded = True
                logging.info(f"Modelo carregado em {time.time() - t0:.1f}s (porta {self.port})")
                return
            time.sleep(1)

        self.unload_model()
        raise TimeoutError(f"O llama-server não ficou pronto em {timeout}s")

    def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.6,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        repeat_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
        enable_thinking: bool = True,
        should_stop=lambda: False,
    ) -> Iterator[Dict[str, str]]:
        """
        Gera a resposta do chat com streaming.

        Args:
            messages: Histórico no formato [{'role': 'user'|'assistant', 'content': ...}]
            temperature: Controla aleatoriedade
            top_p: Nucleus sampling
            max_tokens: Máximo de tokens a gerar (inclui o raciocínio)
            repeat_penalty: Penalidade para repetição
            stop: Lista de sequências que param a geração
            enable_thinking: Liga o modo de raciocínio (modelos Qwen3.x)
            should_stop: Função checada a cada token; True interrompe a geração

        Yields:
            {'type': 'reasoning' | 'content', 'text': trecho}
        """
        if not self.is_loaded:
            raise RuntimeError("Modelo não carregado. Chame load_model() primeiro.")

        payload = {
            'messages': messages,
            'stream': True,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'top_k': 20,
            'repeat_penalty': repeat_penalty,
            'chat_template_kwargs': {'enable_thinking': enable_thinking},
            'timings_per_token': False,
        }
        if stop:
            payload['stop'] = stop

        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
        )
        self.last_timings = {}

        # Fechar a conexão (saindo do with) faz o servidor cancelar a geração
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                if should_stop():
                    break
                line = raw.decode('utf-8').strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    break
                chunk = json.loads(data)
                if chunk.get('timings'):
                    self.last_timings = chunk['timings']
                for choice in chunk.get('choices', []):
                    delta = choice.get('delta') or {}
                    if delta.get('reasoning_content'):
                        yield {'type': 'reasoning', 'text': delta['reasoning_content']}
                    if delta.get('content'):
                        yield {'type': 'content', 'text': delta['content']}

    def unload_model(self):
        """Encerra o llama-server e libera a memória."""
        if self.process is not None:
            logging.info("Encerrando llama-server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self._cleanup()
        logging.info("Modelo descarregado")

    def get_context_size(self) -> int:
        """Retorna o tamanho do contexto do modelo."""
        return self.n_ctx if self.is_loaded else 0

    def _cleanup(self):
        self.process = None
        self.is_loaded = False
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def _healthy(self) -> bool:
        try:
            url = f"http://127.0.0.1:{self.port}/health"
            with urllib.request.urlopen(url, timeout=2) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            return s.getsockname()[1]

    @staticmethod
    def _log_tail(n: int = 15) -> str:
        try:
            lines = (Path('logs') / 'llama-server.log').read_text(
                encoding='utf-8', errors='replace').splitlines()
            return '\n'.join(lines[-n:])
        except OSError:
            return '(log indisponível)'
