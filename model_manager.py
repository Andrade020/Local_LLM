"""
Gerenciamento de modelos GGML/GGUF
Validação, verificação de memória e carregamento
"""

import os
import logging
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
        # Modelos quantizados usam ~1.2-1.5x seu tamanho em RAM
        # Modelos não quantizados usam ~2-3x
        if 'q4' in self.model_path.name.lower() or 'q5' in self.model_path.name.lower():
            multiplier = 1.3
        elif 'q8' in self.model_path.name.lower():
            multiplier = 1.5
        else:
            multiplier = 2.0
        
        self.model_info['estimated_ram_mb'] = size_mb * multiplier
        
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
        
        # Verificar se há memória suficiente (com margem de segurança)
        safety_margin_mb = 2048  # 2 GB de margem
        required_mb = estimated_ram + safety_margin_mb
        
        if available_mb < required_mb:
            message = (
                f"⚠️ AVISO: Memória pode ser insuficiente!\n\n"
                f"Disponível: {available_mb:.0f} MB\n"
                f"Necessário: ~{required_mb:.0f} MB\n"
                f"(Modelo: {estimated_ram:.0f} MB + Margem: {safety_margin_mb} MB)\n\n"
                f"O sistema pode ficar lento ou travar."
            )
            return False, message
        
        # Avisar se RAM total é baixa
        if total_mb < 16384:  # Menos de 16 GB
            message = (
                f"ℹ️ Sistema com {total_mb:.0f} MB RAM total.\n"
                f"Recomendado: 16 GB ou mais para modelos maiores.\n"
                f"Considere usar modelos menores (7B quantizados)."
            )
            return True, message
        
        return True, f"Memória suficiente: {available_mb:.0f} MB disponível"
    
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
        
        quant_types = {
            'q4_0': '4-bit (menor, mais rápido, menos preciso)',
            'q4_1': '4-bit (balanceado)',
            'q5_0': '5-bit (bom balanço)',
            'q5_1': '5-bit (melhor qualidade)',
            'q8_0': '8-bit (alta qualidade, mais lento)',
            'f16': '16-bit float (muito preciso, muito lento)',
            'f32': '32-bit float (precisão máxima, extremamente lento)',
        }
        
        for quant_key, description in quant_types.items():
            if quant_key in name_lower:
                return f"Quantização: {quant_key.upper()} - {description}"
        
        return "Quantização: Desconhecida (verifique o nome do arquivo)"
    
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