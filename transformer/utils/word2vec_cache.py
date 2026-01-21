"""
Word2Vec 模型全局缓存

这个模块实现了 Word2Vec 模型的单例缓存，避免每次创建 TagTransformer 时
都重新加载模型（约需 45 秒）。
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

# 全局缓存
_word2vec_model = None
_word2vec_lock = threading.Lock()
_load_time = None


def get_word2vec_model(model_name: str = "glove-wiki-gigaword-200"):
    """
    获取 Word2Vec 模型，使用全局缓存避免重复加载。
    
    Args:
        model_name: 模型名称，默认为 glove-wiki-gigaword-200
        
    Returns:
        gensim KeyedVectors 对象
    """
    global _word2vec_model, _load_time
    
    # 快速路径：如果已加载，直接返回
    if _word2vec_model is not None:
        logger.info(f"Word2Vec model already cached (loaded in {_load_time:.1f}s), skipping reload")
        return _word2vec_model
    
    # 线程安全的加载
    with _word2vec_lock:
        # 双重检查锁定
        if _word2vec_model is not None:
            return _word2vec_model
        
        logger.info(f"Loading Word2Vec model: {model_name} (this may take ~45 seconds)...")
        start_time = time.time()
        
        import gensim.downloader as api
        _word2vec_model = api.load(model_name)
        
        _load_time = time.time() - start_time
        vocab_size = len(list(_word2vec_model.index_to_key))
        logger.info(f"Word2Vec model loaded in {_load_time:.1f}s, vocab size: {vocab_size}")
        
        return _word2vec_model


def preload_word2vec_model(model_name: str = "glove-wiki-gigaword-200"):
    """
    预加载 Word2Vec 模型（可在程序启动时调用）。
    
    Args:
        model_name: 模型名称
    """
    get_word2vec_model(model_name)


def is_model_cached() -> bool:
    """检查模型是否已缓存"""
    return _word2vec_model is not None


def get_cache_info() -> dict:
    """获取缓存信息"""
    return {
        "is_cached": _word2vec_model is not None,
        "load_time": _load_time,
        "vocab_size": len(list(_word2vec_model.index_to_key)) if _word2vec_model else 0
    }
