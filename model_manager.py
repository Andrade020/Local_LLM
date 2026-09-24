"""
Gerenciamento de modelos GGML/GGUF
Validação, verificação de memória e carregamento
"""

import os
import re
import logging
import subprocess
from pathlib import Path
from typing import Tuple, Dict
import psutil


class ModelManager:
    """Gerencia carregamento e validação de modelos LLM."""
    
    def __init__(self, model_path: str):
        """
        Inicializa o gerenciador de modelos.
        
        Args:
            model_path: Caminho para o arquivo do modelo
        """
        self.model_path = Path(model_path)
        self.model_info: Dict = {}
        
        logging.info(f"ModelManager inicializado com: {model_path}")
    
    def validate_model(self) -> Tuple[bool, str]:
        """
        Valida se o arquivo do modelo existe e é válido.
        
        Returns:
            Tupla (is_valid, message)
        """
        # Verificar se arquivo existe
        if not self.model_path.exists():
            return False, f"Arquivo não encontrado: {self.model_path}"
        
        # Verificar se é arquivo (não diretório)
        if not self.model_path.is_file():
            return False, f"Caminho não é um arquivo: {self.model_path}"
        
        # Verificar extensão
        valid_extensions = ['.gguf', '.bin', '.ggml']
        if self.model_path.suffix.lower() not in valid_extensions:
            logging.warning(
                f"Extensão não reconhecida: {self.model_path.suffix}. "
                f"Esperado: {', '.join(valid_extensions)}"
            )
        
        # Verificar tamanho mínimo (pelo menos 10 MB)
        size_mb = self.model_path.stat().st_size / (1024 * 1024)
        if size_mb < 10:
            return False, f"Arquivo muito pequeno ({size_mb:.1f} MB). Modelo inválido?"
        
        # Coletar informações do modelo
        self.model_info = {
            'name': self.model_path.name,
            'path': str(self.model_path.absolute()),
            'size_bytes': self.model_path.stat().st_size,
            'size_mb': size_mb,
            'extension': self.model_path.suffix,
        }
        
        # Estimar uso de RAM (heurística)
        # O llama.cpp mapeia o GGUF direto na memória (mmap): o uso é ~o tamanho
        # do arquivo + contexto/buffers. A parte que vai para a GPU sai da RAM.
        self.model_info['estimated_ram_mb'] = size_mb * 1.1
        
        logging.info(f"Modelo validado: {self.model_info}")
        return True, "Modelo válido"
    
    def check_memory_requirements(self) -> Tuple[bool, str]:
        """
        Verifica se o sistema tem memória suficiente.
        
        Returns:
            Tupla (has_enough_memory, message)
        """
        # Obter memória RAM disponível
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)
        total_mb = memory.total / (1024 * 1024)
        
        estimated_ram = self.model_info.get('estimated_ram_mb', 0)
        
        logging.info(
            f"Memória: {available_mb:.0f} MB disponível / "
            f"{total_mb:.0f} MB total"
        )
        logging.info(f"Modelo requer ~{estimated_ram:.0f} MB")
        
        # Com mmap o Windows libera cache sob demanda, então o limite real é a RAM
        # total (menos ~1 GB do sistema), somada ao que couber na GPU.
        limit_mb = total_mb + self._gpu_memory_mb() - 1024

        if estimated_ram > limit_mb:
            message = (
                f"⚠️ AVISO: o modelo provavelmente não cabe na memória!\n\n"
                f"RAM total + VRAM: {limit_mb + 1024:.0f} MB\n"
                f"Modelo: ~{estimated_ram:.0f} MB\n\n"
                f"O sistema pode ficar muito lento ou travar."
            )
            return False, message

        if estimated_ram > available_mb:
            message = (
                f"ℹ️ Memória livre agora: {available_mb:.0f} MB; o modelo usa ~{estimated_ram:.0f} MB.\n"
                f"Feche programas pesados (navegador, jogos) para manter a velocidade."
            )
            return True, message

        return True, f"Memória suficiente: {available_mb:.0f} MB disponível"

    @staticmethod
    def _gpu_memory_mb() -> float:
        """VRAM total da GPU NVIDIA (0 se não houver ou se não der para consultar)."""
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
            return float(out.stdout.split()[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return 0.0
    
    def get_model_info(self) -> Dict:
        """Retorna informações do modelo."""
        return self.model_info.copy()
    
    def get_quantization_info(self) -> str:
        """
        Detecta o tipo de quantização do modelo pelo nome.
        
        Returns:
            String descrevendo a quantização
        """
        name_lower = self.model_path.name.lower()

        match = re.search(r'(iq\d_[a-z0-9]+|q\d_k_[a-z]+|q\d_k|q\d_\d|mxfp4|bf16|f16|f32)', name_lower)
        if not match:
            return "Quantização: Desconhecida (verifique o nome do arquivo)"

        quant = match.group(1)
        if quant in ('bf16', 'f16', 'f32'):
            description = 'sem quantização (preciso, pesado e lento)'
        elif quant == 'mxfp4':
            description = '4-bit em ponto flutuante'
        else:
            bits = int(re.search(r'\d', quant).group())
            description = f'~{bits}-bit' + (' (i-quant: melhor qualidade por bit)' if quant.startswith('iq') else '')
        return f"Quantização: {quant.upper()} - {description}"
    
    @staticmethod
    def suggest_model_size(available_ram_mb: float) -> str:
        """
        Sugere tamanho de modelo baseado na RAM disponível.
        
        Args:
            available_ram_mb: RAM disponível em MB
            
        Returns:
            String com recomendação
        """
        if available_ram_mb < 4096:
            return "< 4 GB RAM: Modelos muito pequenos (1B-3B quantizados)"
        elif available_ram_mb < 8192:
            return "4-8 GB RAM: Modelos pequenos (3B-7B quantizados Q4)"
        elif available_ram_mb < 16384:
            return "8-16 GB RAM: Modelos médios (7B quantizados Q5/Q8)"
        elif available_ram_mb < 32768:
            return "16-32 GB RAM: Modelos grandes (13B quantizados, 7B não quantizados)"
        else:
            return "32+ GB RAM: Modelos muito grandes (30B+ quantizados, 13B não quantizados)"