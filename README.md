# Manga Translator

Tradutor automático de mangás. Recebe a página escaneada em inglês e devolve a
mesma página com as falas traduzidas para o português, no lugar dos balões
originais.

## Como funciona

O pipeline tem cinco etapas, executadas em sequência pelo `main.py`:

| # | Etapa | O que faz | Ferramenta |
|---|---|---|---|
| 1 | Identificação | Localiza os balões de fala na página | YOLOv8 (modelo próprio) |
| 2 | Extração | Lê o texto de dentro de cada balão | Tesseract OCR |
| 3 | Tradução | Traduz do inglês para o português | Google Tradutor / Gemini |
| 4 | Remoção | Apaga o texto original | Preenchimento branco sobre o balão |
| 5 | Inserção | Escreve a tradução com fonte de mangá | Pillow |

**Etapa 4 — estado atual.** A remoção é feita pintando de branco a área do balão
detectado, por cima do texto original. Funciona bem em balão de fala comum, que
tem fundo branco chapado, e é a abordagem mais simples possível. A limitação
conhecida é que o retângulo cobre tudo o que estiver dentro da caixa detectada,
inclusive a borda do balão e a arte que a caixa tenha englobado. Substituir isso
por inpainting é o próximo passo do projeto.

## Estrutura

```
manga_translator/
├── main.py                 # pipeline completo, ponto de entrada
├── translator/
│   ├── Google_translator.py  # etapa 3 via Google Tradutor, com cache e fallback
│   ├── LLM_translator.py     # etapa 3 via Gemini, com prompt de contexto de mangá
│   └── .env                  # chaves de API (não versionado)
├── lettering/
│   └── lettering.py          # etapa 5: quebra de linha e ajuste do corpo da fonte
├── train/
│   ├── train.py              # treino do detector de balões
│   ├── data.yaml             # configuração do dataset
│   └── runs/                 # pesos gerados pelo treino
├── font/                     # fontes de mangá (KOMIKAX, Anime Ace, CC Wild Words)
└── scr_manga/
    ├── inputs/               # páginas de entrada
    ├── outputs/              # páginas traduzidas
    └── weights_AymanKUMA/    # detector alternativo, de terceiros
```

## Instalação

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**Tesseract.** Precisa ser instalado à parte, pelo instalador do
[UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). O caminho padrão
esperado é `C:\Program Files\Tesseract-OCR`. Se o seu for outro, ajuste a linha
`pytesseract.pytesseract.tesseract_cmd` no `main.py`.

**Chaves de API.** Só são necessárias para a tradução por Gemini. Crie
`translator/.env` com:

```
Google_Api_Key=sua_chave
```

A tradução pelo Google Tradutor não exige chave.

## Uso

Coloque a página em `scr_manga/inputs/` e ajuste a constante `ENTRADA` no topo do
`main.py`. Depois:

```bash
python main.py
```

A página traduzida sai em `scr_manga/outputs/`, junto com uma cópia mostrando as
caixas detectadas pelo YOLO, útil para conferir a etapa 1.

## Detecção de balões

O detector é um YOLOv8m treinado por 100 épocas em um dataset próprio de páginas
de mangá anotadas, com uma única classe (`eng`, balão com texto em inglês). Para
retreinar:

```bash
cd train
python train.py
```

Os pesos ficam em `train/runs/detect/trainN/weights/best.pt` — o `main.py` aponta
para o run em uso. Há também um detector de terceiros em
`scr_manga/weights_AymanKUMA/`, mantido como referência de comparação.

## Tradução

Dois motores, intercambiáveis:

**Google Tradutor** (`Google_translator.py`) é o padrão. Não exige chave, mas o
`deep-translator` acessa o tradutor raspando a página web, o que faz o Google
limitar o IP depois de muitas requisições seguidas. Por isso o módulo mantém um
cache em disco (`cache_traducao.json`), respeita um intervalo mínimo entre
requisições, tenta novamente com espera progressiva e cai para o MyMemory quando
o Google não responde. Nenhum balão problemático derruba a página inteira: no
pior caso o texto original é mantido.

**Gemini** (`LLM_translator.py`) usa um prompt com o contexto da obra e instruções
de adaptação de onomatopeias e expressões de impacto, o que produz tradução mais
natural para mangá do que a tradução literal. Exige chave de API.

## Inserção do texto

A etapa 5 calcula, para cada balão, o maior corpo de fonte que faz o texto caber
na caixa, quebrando as linhas pela largura real em pixels e centralizando o bloco.
A busca é feita por bisseção sobre o corpo da fonte, com as medições de largura
das palavras reaproveitadas entre as tentativas, o que evita remedir e recarregar
a fonte a cada candidato.

Palavras maiores que o balão são divididas em vez de vazarem para fora dele.

## Limitações conhecidas

- A caixa do YOLO é retangular e o balão não é, então o texto pode encostar nas
  bordas em balões muito irregulares.
- O preenchimento branco da etapa 4 apaga o que estiver dentro da caixa, incluindo
  a borda do balão.
- Texto fora de balão (onomatopeia sobre a arte) não é tratado.
- O OCR erra em fontes muito estilizadas, e o erro se propaga para a tradução.

## Licença

MIT. Ver [LICENSE](LICENSE).
