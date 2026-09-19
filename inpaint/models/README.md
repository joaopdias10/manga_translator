# Modelos de inpainting (etapa 4)

Os pesos NAO ficam no repositorio (sao grandes). Baixe para esta pasta o(s)
que for usar. Todos sao fine-tunados em manga.

| nome no codigo | arquivo | de onde baixar |
|---|---|---|
| `lama` (padrao) | `anime-manga-big-lama.pt` | github.com/Sanster/models -> release **AnimeMangaInpainting** |
| `lama_onnx` | `lama-manga-dynamic.onnx` | huggingface.co/ogkalu/lama-manga-onnx-dynamic |
| `aot` | `aot.onnx` | huggingface.co/ogkalu/aot-inpainting |
| `migan` | `migan_pipeline_v2.onnx` | github.com/Sanster/models -> release **migan** |

- `lama` usa PyTorch (ja instalado pelo ultralytics). Os demais usam ONNX:
  `pip install onnxruntime`.
- Alternativa a colocar o arquivo aqui: aponte a variavel de ambiente
  (`LAMA_MODEL`, `LAMA_ONNX`, `AOT_ONNX`, `MIGAN_ONNX`) para o caminho do peso.
- Sem o peso, o metodo e pulado no comparador e a rota "arte" do pipeline
  cai para o `telea` (OpenCV), sem quebrar.
