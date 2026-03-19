# FrontMachine 

Este projeto utiliza **MediaPipe** para detecção de marcos da mão e **Scikit-Learn (RandomForest)** para classificação de gestos customizados em tempo real através da webcam.

## Estrutura do Projeto

- `models/`: Contém os arquivos `.tflite`, `.task` e os modelos treinados (`.pkl`).
- `app.py`: O código principal para execução.
- `.venv/`: Ambiente virtual Python gerido pelo **uv**.

## Instalação e Execução

Para instalar as dependências e executar o projeto:

```bash
uv sync
uv run app.py
```
