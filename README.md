# Manga Translator

Tradutor automático de mangás. Recebe a página escaneada em inglês ou japonês e devolve a
mesma página com as falas traduzidas para o português, no lugar dos balões
originais.

## Como funciona

O pipeline tem cinco etapas, executadas em sequência pelo `main.py`:

| # | Etapa | O que faz | Ferramenta |
|---|---|---|---|
| 1 | Identificação | Localiza os balões de fala na página | YOLO (modelo próprio) |
| 2 | Extração | Lê o texto de dentro de cada balão | Tesseract (inglês) / Manga-OCR (japonês) |
| 3 | Tradução | Traduz a página inteira para o português, com contexto | GPT (OpenAI) / Google Tradutor |
| 4 | Remoção | Apaga o texto original | Inpainting (preenchimento sólido ou LaMa) |
| 5 | Inserção | Escreve a tradução com fonte de mangá | Pillow |

## Estrutura

```
manga_translator/
├── main.py                 # pipeline completo, ponto de entrada
├── ocr/
│   └── ocr.py                # etapa 2: Tesseract (inglês) / Manga-OCR (japonês)
├── translator/
│   ├── translator.py         # etapa 3 via Google Tradutor, com cache e fallback
│   ├── LLM_translator.py     # etapa 3 via Gemini, com prompt de contexto de mangá
│   └── .env                  # chaves de API (não versionado)
├── inpaint/
│   ├── inpainter.py          # etapa 4: máscara do texto + rotas de preenchimento
│   └── models/               # pesos do LaMa de mangá (baixados à parte, não versionados)
├── lettering/
│   └── lettering.py          # etapa 5: quebra de linha e ajuste do corpo da fonte
├── train/
│   ├── train.py              # treino do detector de balões
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
`pytesseract.pytesseract.tesseract_cmd` no `ocr/ocr.py`.

**Pesos do inpainting.** O LaMa de mangá (`anime-manga-big-lama.pt`, ~200 MB) não
vem no repositório. Baixe para `inpaint/models/` — ver `inpaint/models/README.md`.
Sem ele, a rota de arte da etapa 4 cai automaticamente para o Telea do OpenCV.

**Pesos do Manga-OCR.** O modelo japonês (~450 MB) é baixado do HuggingFace na
primeira execução com texto japonês e fica no cache local.

**Chaves de API.** Só são necessárias para a tradução por LLM. Crie
`translator/.env` com a chave do motor em uso — por padrão o GPT (OpenAI):

```
OpenAI_Api_Key=sua_chave
```

Para usar o Gemini em vez do GPT, troque o modelo em `translator/LLM_translator.py`
e defina `Google_Api_Key` no `.env`. A tradução pelo Google Tradutor não exige
chave.

## Uso

```bash
python main.py                          # página padrão
python main.py scr_manga/inputs/7.png   # outra página
```

A página passada como argumento (ou a padrão) é processada e a versão traduzida
sai em `scr_manga/outputs/`:

- `resultado.jpg` — página com as caixas detectadas pelo YOLO (confere a etapa 1)
- `pagina_limpa.png` — página só com o texto removido (confere a etapa 4)
- `2.jpeg` — página final, traduzida e com o lettering

## Detecção de balões

O detector é um YOLO (ultralytics) treinado em um dataset próprio de páginas de
mangá anotadas, com uma classe por idioma: `eng` (balão em inglês) e `jp` (balão
em japonês). A classe de cada caixa é o que decide o motor de OCR na etapa 2. O
`main.py` aponta para o run em uso (`train/runs/v1/best_lote12.pt`); para
retreinar:

```bash
cd train
python train.py
```

Há também um detector de terceiros em `scr_manga/weights_AymanKUMA/`, mantido como
referência de comparação.

## Extração e idioma

Cada caixa detectada vai para o OCR do seu idioma: **Tesseract** para inglês e
**Manga-OCR** para japonês. O Tesseract não serve para japonês de mangá (texto
vertical, furigana, fontes estilizadas); o Manga-OCR é um modelo
VisionEncoderDecoder treinado em recortes de balão de mangá e lê o bloco inteiro,
vertical ou horizontal. O modelo só é carregado quando aparece texto japonês.

O idioma de cada caixa vem da classe que o YOLO deu a ela (`eng` ou `jp`),
resolvido em `ocr/ocr.py`.

## Tradução

A página é traduzida de uma vez só, não balão a balão: os textos são lidos na
ordem de leitura de mangá (de cima para baixo, da direita para a esquerda) e
enviados juntos, para que o tradutor tenha o contexto da conversa inteira.

Dois motores, intercambiáveis:

**LLM** (`LLM_translator.py`) é o motor de página (`translate_page`). Por padrão
usa o **GPT (OpenAI, `gpt-4o-mini`)**; o Gemini está disponível no mesmo arquivo,
bastando trocar a linha do modelo. Usa um prompt com o contexto da obra e
instruções de adaptação de onomatopeias e expressões de impacto, o que produz
tradução mais natural para mangá do que a literal. Exige chave de API.

**Google Tradutor** (`translator.py`) é a alternativa sem chave. Como o
`deep-translator` acessa o tradutor raspando a página web, o Google limita o IP
depois de muitas requisições seguidas; por isso o módulo mantém um cache em disco
(`cache_traducao.json`), respeita um intervalo mínimo entre requisições, tenta de
novo com espera progressiva e cai para o MyMemory quando o Google não responde.
Nenhum balão problemático derruba a página: no pior caso o texto original é
mantido.

## Remoção do texto (inpainting)

A ideia central é que **a caixa do YOLO não é a máscara**: a caixa diz "aqui tem
texto", a máscara diz "estes pixels são o texto". `inpaint/inpainter.py` gera a
máscara dos glifos (binarização + filtro de componentes ancorados na caixa,
contidos pelo interior do balão para não comer a linha) e então escolhe a rota de
preenchimento por região, no modo `auto`:

- **balão** (fundo uniforme atrás do texto) → preenchimento **sólido** com a cor
  de fundo amostrada. Apaga o interior inteiro do balão de uma vez, instantâneo e
  sem modelo.
- **arte** (desenho ou screentone atrás do texto) → **LaMa**, uma rede neural de
  inpainting com checkpoint afinado em mangá. Se o peso não estiver presente, cai
  para o Telea (`cv2.inpaint`) do OpenCV.

Além do LaMa, o módulo traz os baselines clássicos (`telea`, `ns`) e backends
alternativos em ONNX (`lama_onnx`, `aot`, `migan`), úteis para comparação —
ver `comparar_inpaint.py`.

## Inserção do texto

A etapa 5 calcula, para cada balão, o maior corpo de fonte que faz o texto caber
na caixa, quebrando as linhas pela largura real em pixels e centralizando o bloco.
A busca é feita por bisseção sobre o corpo da fonte, com as medições de largura
das palavras reaproveitadas entre as tentativas, o que evita remedir e recarregar
a fonte a cada candidato.

Palavras maiores que o balão são divididas em vez de vazarem para fora dele, e
quando o texto é longo demais para um balão pequeno o corpo encolhe até caber
(sem transbordar por cima da arte).

## Limitações conhecidas

- A caixa do YOLO é retangular e o balão não é, então o texto pode encostar nas
  bordas em balões muito irregulares.
- O LaMa custa tempo e é sensível em legendas largas de fundo claro; para essas,
  um método clássico costuma ficar mais limpo.
- Quando a caixa do detector não cobre toda a largura de uma legenda, sobra texto
  nítido na borda — é limitação da etapa 1, não do inpaint.
- Texto fora de balão (onomatopeia sobre a arte) não é tratado.
- O OCR erra em fontes muito estilizadas, e o erro se propaga para a tradução.

## Licença

MIT. Ver [LICENSE](LICENSE).
