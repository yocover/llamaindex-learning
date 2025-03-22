from setuptools import setup, find_packages

setup(
    name="llamaindex_learning",
    version="0.1",
    author="Yocover",
    description="LlamaIndex learning project with custom embeddings",
    packages=find_packages(include=['src', 'src.*']),  # 明确指定包含 src 目录
    python_requires=">=3.10",
    install_requires=[
        "llama-index",
        "llama-index-core",
        "requests",
        "aiohttp",
        "pydantic",
    ],
)
