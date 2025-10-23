"""
Wrapper de inferência para llama-cpp-python
Gerencia geração de texto com streaming
"""

import logging
from typing import Iterator, Optional, List
from llama_cpp import Llama

from model_manager import ModelManager


class LLMInferencer:
    """Wrapper para inferência com llama-cpp-python."""
    
    def __init__(self, model_manager: ModelManager):
        """
        Inicializa o inferencer.
        
        Args:
            model_manager: Instância do ModelManager com modelo validado
        """
        self.model_manager = model_manager
        self.llm: Optional[Llama] = None
        self.is_loaded = False
        
        logging.info("LLMInferencer inicializado")
    
    def load_model(
        self,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        n_batch: int = 512,
        use_mlock: bool = False,
        verbose: bool = False
    ):
        """
        Carrega o modelo na memória.
        
        Args:
            n_ctx: Tamanho do contexto (tokens)
            n_threads: Número de threads (None = auto)
            n_batch: Tamanho do batch para processamento
            use_mlock: Usar mlock para evitar swap (requer privilégios)
            verbose: Modo verbose do llama.cpp
        """
        model_path = str(self.model_manager.model_path)
        
        logging.info(f"Carregando modelo: {model_path}")
        logging.info(f"Parâmetros: n_ctx={n_ctx}, n_threads={n_threads}, n_batch={n_batch}")
        
        try:
            # Criar instância do Llama
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_batch=n_batch,
                use_mlock=use_mlock,
                verbose=verbose,
                # Otimizações para CPU
                n_gpu_layers=0,  # Não usar GPU
                f16_kv=True,  # Usar FP16 para cache K/V (economiza RAM)
            )
            
            self.is_loaded = True
            logging.info("Modelo carregado com sucesso!")
            
        except Exception as e:
            self.is_loaded = False
            logging.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Gera texto de forma não-streaming.
        
        Args:
            prompt: Texto de entrada
            temperature: Controla aleatoriedade (0.0 = determinístico, 2.0 = muito aleatório)
            top_p: Nucleus sampling (0.0-1.0)
            max_tokens: Máximo de tokens a gerar
            repeat_penalty: Penalidade para repetição (>1.0 = menos repetição)
            stop: Lista de sequências que param a geração
            
        Returns:
            Texto gerado
        """
        if not self.is_loaded:
            raise RuntimeError("Modelo não carregado. Chame load_model() primeiro.")
        
        logging.debug(f"Gerando resposta para prompt: {prompt[:50]}...")
        
        try:
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=stop or [],
                echo=False  # Não incluir prompt na saída
            )
            
            # Extrair texto da resposta
            text = output['choices'][0]['text']
            
            logging.debug(f"Resposta gerada: {len(text)} caracteres")
            return text
            
        except Exception as e:
            logging.error(f"Erro na geração: {e}")
            raise
    
    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None
    ) -> Iterator[str]:
        """
        Gera texto com streaming (token por token).
        
        Args:
            prompt: Texto de entrada
            temperature: Controla aleatoriedade
            top_p: Nucleus sampling
            max_tokens: Máximo de tokens a gerar
            repeat_penalty: Penalidade para repetição
            stop: Lista de sequências que param a geração
            
        Yields:
            Tokens individuais conforme são gerados
        """
        if not self.is_loaded:
            raise RuntimeError("Modelo não carregado. Chame load_model() primeiro.")
        
        logging.debug(f"Gerando resposta (streaming) para prompt: {prompt[:50]}...")
        
        try:
            # Criar gerador com streaming
            stream = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=stop or [],
                echo=False,
                stream=True  # Habilitar streaming
            )
            
            # Iterar sobre tokens
            for output in stream:
                # Extrair texto do token
                token = output['choices'][0]['text']
                yield token
                
        except Exception as e:
            logging.error(f"Erro na geração (streaming): {e}")
            raise
    
    def unload_model(self):
        """Descarrega o modelo da memória."""
        if self.llm is not None:
            logging.info("Descarregando modelo...")
            del self.llm
            self.llm = None
            self.is_loaded = False
            logging.info("Modelo descarregado")
    
    def get_context_size(self) -> int:
        """Retorna o tamanho do contexto do modelo."""
        if not self.is_loaded:
            return 0
        return self.llm.n_ctx()
    
    def reset_context(self):
        """Reseta o contexto do modelo (limpa histórico)."""
        if self.is_loaded and self.llm is not None:
            self.llm.reset()
            logging.info("Contexto resetado")