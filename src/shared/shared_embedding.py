import torch
from langchain_huggingface import HuggingFaceEmbeddings

_embedding_model = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _embedding_model = HuggingFaceEmbeddings(
        model_name='BAAI/bge-m3',
        model_kwargs={
            'device': device,
            # 'model_kwargs': {'torch_dtype': torch.float16}
            'model_kwargs': {'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32}
        },
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 8}
    )

    try:
        target = (
            getattr(_embedding_model, 'client', None)
            or _embedding_model.__dict__.get('client')
            or _embedding_model.__dict__.get('_client')
        )
        if target:
            target.max_seq_length = 768
    except Exception:
        pass

    print(f"✅ bge-m3 로드 완료 ({device})")
    return _embedding_model
