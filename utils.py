"""
Utilitários: logging, caching, hash, verificações de sistema
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import psutil


def setup_logging(
    log_level: Optional[str] = None,
    log_file: str = 'app.log'
):
    """
    Configura o sistema de logging.
    
    Args:
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR)
        log_file: Arquivo para salvar logs
    """
    # Obter nível do ambiente ou usar padrão
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    # Converter string para nível
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configurar formato
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configurar logging
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging configurado: nível={log_level}, arquivo={log_file}")


def hash_prompt(prompt: str, settings: Dict[str, Any]) -> str:
    """
    Cria hash único para prompt + configurações.
    Usado para cache de respostas.
    
    Args:
        prompt: Texto do prompt
        settings: Dicionário de configurações de inferência
        
    Returns:
        Hash SHA256 em hexadecimal
    """
    # Criar string com prompt + settings relevantes
    cache_key_data = {
        'prompt': prompt,
        'temperature': settings.get('temperature', 0.7),
        'top_p': settings.get('top_p', 0.9),
        'max_tokens': settings.get('max_tokens', 512),
    }
    
    # Serializar e criar hash
    key_str = json.dumps(cache_key_data, sort_keys=True)
    hash_obj = hashlib.sha256(key_str.encode('utf-8'))
    
    return hash_obj.hexdigest()


def load_cache(cache_key: str) -> Optional[str]:
    """
    Carrega resposta do cache.
    
    Args:
        cache_key: Hash do prompt + settings
        
    Returns:
        Resposta cacheada ou None se não existir
    """
    cache_dir = Path('cache')
    cache_file = cache_dir / f"{cache_key}.json"
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logging.debug(f"Cache hit: {cache_key[:8]}...")
            return data.get('response')
    except Exception as e:
        logging.warning(f"Erro ao carregar cache: {e}")
        return None


def save_cache(cache_key: str, response: str):
    """
    Salva resposta no cache.
    
    Args:
        cache_key: Hash do prompt + settings
        response: Resposta a ser cacheada
    """
    cache_dir = Path('cache')
    cache_dir.mkdir(exist_ok=True)
    
    cache_file = cache_dir / f"{cache_key}.json"
    
    try:
        data = {
            'cache_key': cache_key,
            'response': response,
            'timestamp': str(Path(cache_file).stat().st_mtime) if cache_file.exists() else None
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logging.debug(f"Cache salvo: {cache_key[:8]}...")
        
    except Exception as e:
        logging.warning(f"Erro ao salvar cache: {e}")


def check_system_requirements() -> Tuple[bool, List[str]]:
    """
    Verifica requisitos mínimos do sistema.
    
    Returns:
        Tupla (requirements_met, list_of_warnings)
    """
    warnings = []
    requirements_met = True
    
    # Verificar RAM
    memory = psutil.virtual_memory()
    total_ram_gb = memory.total / (1024 ** 3)
    
    if total_ram_gb < 8:
        warnings.append(
            f"RAM total: {total_ram_gb:.1f} GB - ABAIXO DO MÍNIMO (8 GB recomendado)"
        )
        requirements_met = False
    elif total_ram_gb < 16:
        warnings.append(
            f"RAM total: {total_ram_gb:.1f} GB - Adequado para modelos pequenos/médios"
        )
    else:
        logging.info(f"RAM total: {total_ram_gb:.1f} GB - Adequado")
    
    # Verificar CPU
    cpu_count = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)
    
    if cpu_count < 4:
        warnings.append(
            f"CPU: {cpu_count} cores - Pode ser lento para modelos maiores"
        )
    else:
        logging.info(f"CPU: {cpu_count} cores ({cpu_count_physical} físicos)")
    
    # Verificar espaço em disco
    disk = psutil.disk_usage('.')
    free_gb = disk.free / (1024 ** 3)
    
    if free_gb < 10:
        warnings.append(
            f"Espaço em disco: {free_gb:.1f} GB - Pouco espaço para modelos e cache"
        )
    
    return requirements_met, warnings


def get_system_info() -> Dict[str, Any]:
    """
    Coleta informações detalhadas do sistema.
    
    Returns:
        Dicionário com informações do sistema
    """
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('.')
    
    return {
        'cpu': {
            'count_logical': psutil.cpu_count(logical=True),
            'count_physical': psutil.cpu_count(logical=False),
            'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None,
        },
        'memory': {
            'total_gb': memory.total / (1024 ** 3),
            'available_gb': memory.available / (1024 ** 3),
            'used_percent': memory.percent,
        },
        'disk': {
            'total_gb': disk.total / (1024 ** 3),
            'free_gb': disk.free / (1024 ** 3),
            'used_percent': disk.percent,
        }
    }


def format_bytes(bytes_size: int) -> str:
    """
    Formata tamanho em bytes para string legível.
    
    Args:
        bytes_size: Tamanho em bytes
        
    Returns:
        String formatada (ex: "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"