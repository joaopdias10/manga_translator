"""
Etapa 5 - Inserção (lettering).

Desenha o texto traduzido dentro do balão, escolhendo o maior tamanho de
fonte que ainda cabe na caixa.

Otimizações em relação à versão anterior:
  1. Fontes ficam em cache (lru_cache) - antes cada balão recarregava o
     arquivo .ttf do disco até ~34 vezes.
  2. A largura de cada palavra é medida UMA única vez, num tamanho de
     referência, e depois só escalada proporcionalmente. Medir glifo é a
     parte cara; escalar é multiplicação.
  3. Busca binária pelo tamanho da fonte (~7 tentativas) no lugar da
     varredura linear de 80 até 12 de 2 em 2 (~34 tentativas).
  4. A quebra de linha acumula larguras em vez de refazer ' '.join(...) e
     remedir a linha inteira a cada palavra (era O(n^2) em cada tamanho).
  5. Palavras maiores que o balão são fatiadas em vez de vazarem pra fora.
  6. O desenho usa anchor="ma", então não é preciso remedir cada linha só
     para centralizar.
"""

from functools import lru_cache

from PIL import ImageFont

# Tamanho usado só para medir proporções. Métricas TrueType são lineares
# no corpo da fonte, então medir aqui e escalar é fiel o bastante para a
# busca - e no fim ainda existe uma verificação exata.
_TAMANHO_REF = 100

TAMANHO_MIN = 12
TAMANHO_MAX = 80


# --------------------------------------------------------------------------
# Cache de fontes e medições
# --------------------------------------------------------------------------

@lru_cache(maxsize=64)
def carregar_fonte(caminho_fonte, tamanho):
    """Carrega (e memoriza) a fonte. Chamadas repetidas não tocam o disco."""
    try:
        return ImageFont.truetype(caminho_fonte, size=tamanho)
    except (IOError, OSError):
        try:
            return ImageFont.load_default(size=tamanho)
        except TypeError:  # Pillow antigo
            return ImageFont.load_default()


@lru_cache(maxsize=8192)
def _largura_ref(caminho_fonte, texto):
    """Largura do trecho no tamanho de referência (medida uma vez por palavra)."""
    return carregar_fonte(caminho_fonte, _TAMANHO_REF).getlength(texto)


@lru_cache(maxsize=64)
def _metricas_ref(caminho_fonte):
    """(altura_linha, descida) por unidade de tamanho de fonte."""
    fonte = carregar_fonte(caminho_fonte, _TAMANHO_REF)
    ascent, descent = fonte.getmetrics()
    return (ascent + descent) / _TAMANHO_REF, descent / _TAMANHO_REF


# --------------------------------------------------------------------------
# Quebra de linha
# --------------------------------------------------------------------------

def _fatiar_palavra(palavra, largura_de, max_largura):
    """Parte uma palavra que não cabe na largura do balão (nomes, onomatopeias)."""
    pedacos = []
    atual = ""
    for ch in palavra:
        candidato = atual + ch
        if atual and largura_de(candidato) > max_largura:
            pedacos.append(atual)
            atual = ch
        else:
            atual = candidato
    if atual:
        pedacos.append(atual)
    return pedacos or [palavra]


def _quebrar(palavras, largura_de, largura_espaco, max_largura):
    """
    Monta as linhas acumulando larguras já conhecidas.
    `largura_de` é uma função texto -> largura em pixels.
    """
    linhas = []
    atual = []
    largura_atual = 0.0

    for palavra in palavras:
        largura = largura_de(palavra)

        # Palavra sozinha maior que o balão: fatia e segue com o último pedaço.
        if largura > max_largura:
            if atual:
                linhas.append(" ".join(atual))
                atual, largura_atual = [], 0.0
            pedacos = _fatiar_palavra(palavra, largura_de, max_largura)
            linhas.extend(pedacos[:-1])
            atual = [pedacos[-1]]
            largura_atual = largura_de(pedacos[-1])
            continue

        extra = largura if not atual else largura_espaco + largura
        if largura_atual + extra <= max_largura:
            atual.append(palavra)
            largura_atual += extra
        else:
            linhas.append(" ".join(atual))
            atual = [palavra]
            largura_atual = largura

    if atual:
        linhas.append(" ".join(atual))

    return linhas


def wrap_text(texto, max_width, font):
    """Compatibilidade com a API antiga: quebra usando um objeto de fonte pronto."""
    palavras = texto.split()
    if not palavras:
        return []
    cache = {}

    def largura_de(t):
        valor = cache.get(t)
        if valor is None:
            valor = font.getlength(t)
            cache[t] = valor
        return valor

    return _quebrar(palavras, largura_de, largura_de(" "), max_width)


# --------------------------------------------------------------------------
# Escolha do tamanho da fonte
# --------------------------------------------------------------------------

def _linhas_cabem(linhas, tamanho, altura_linha_unit, descida_unit, altura_box):
    altura_linha = altura_linha_unit * tamanho
    espacamento = descida_unit * tamanho * 0.5
    total = len(linhas) * altura_linha + (len(linhas) - 1) * espacamento
    return total <= altura_box


def best_font(texto, largura_box, altura_box, caminho_fonte,
              tamanho_min=TAMANHO_MIN, tamanho_max=TAMANHO_MAX):
    """
    Maior tamanho de fonte que faz o texto caber em (largura_box, altura_box).

    Retorna (fonte, linhas, altura_linha, espacamento) - mesma assinatura de antes.
    """
    palavras = texto.split()
    altura_linha_unit, descida_unit = _metricas_ref(caminho_fonte)

    if not palavras:
        fonte = carregar_fonte(caminho_fonte, tamanho_min)
        ascent, descent = fonte.getmetrics()
        return fonte, [], ascent + descent, descent * 0.5

    # Medições de referência: cada palavra distinta é medida uma única vez.
    ref = {p: _largura_ref(caminho_fonte, p) for p in set(palavras)}
    espaco_ref = _largura_ref(caminho_fonte, " ")
    soma_ref = sum(ref[p] for p in palavras) + espaco_ref * (len(palavras) - 1)

    # Limites analíticos - cortam boa parte do intervalo de busca de graça.
    # (a) uma única linha já precisa caber na altura
    limite_altura = altura_box / altura_linha_unit
    # (b) estimativa por área: n_linhas ~ (soma * s) / largura  =>  s^2 <= ...
    limite_area = ((altura_box * largura_box * _TAMANHO_REF)
                   / (soma_ref * altura_linha_unit)) ** 0.5 * 1.2
    alto = int(min(tamanho_max, limite_altura, limite_area))
    alto = max(alto, tamanho_min)
    baixo = tamanho_min

    def cabe(tamanho):
        fator = tamanho / _TAMANHO_REF
        linhas = _quebrar(
            palavras,
            lambda t, _r=ref, _c=caminho_fonte: (_r.get(t) or _largura_ref(_c, t)) * fator,
            espaco_ref * fator,
            largura_box,
        )
        return linhas, _linhas_cabem(linhas, tamanho, altura_linha_unit,
                                     descida_unit, altura_box)

    # Busca binária: ~7 avaliações em vez de ~34.
    melhor = None
    while baixo <= alto:
        meio = (baixo + alto) // 2
        linhas, ok = cabe(meio)
        if ok:
            melhor = (meio, linhas)
            baixo = meio + 1
        else:
            alto = meio - 1

    tamanho = melhor[0] if melhor else tamanho_min

    # Verificação exata com a fonte real (a escala de referência tem erro
    # de arredondamento de subpixel). Se estourou, desce de 1 em 1.
    for _ in range(4):
        fonte = carregar_fonte(caminho_fonte, tamanho)
        ascent, descent = fonte.getmetrics()
        altura_linha = ascent + descent
        espacamento = descent * 0.5
        linhas = wrap_text(texto, largura_box, fonte)
        total = len(linhas) * altura_linha + (len(linhas) - 1) * espacamento
        if total <= altura_box or tamanho <= tamanho_min:
            return fonte, linhas, altura_linha, espacamento
        tamanho -= 1

    return fonte, linhas, altura_linha, espacamento


# --------------------------------------------------------------------------
# Desenho
# --------------------------------------------------------------------------

def spell(draw, texto, x1, y1, x2, y2, caminho_fonte="arial.ttf",
          cor=(0, 0, 0), margem_x=0.85, padding_y=10,
          tamanho_min=TAMANHO_MIN, tamanho_max=TAMANHO_MAX):
    """
    Escreve o texto centralizado dentro da caixa (x1, y1) - (x2, y2).
    """
    if not texto or not texto.strip():
        return

    largura_util = (x2 - x1) * margem_x
    altura_util = (y2 - y1) - padding_y * 2
    if largura_util <= 0 or altura_util <= 0:
        return

    fonte, linhas, altura_linha, espacamento = best_font(
        texto, largura_util, altura_util, caminho_fonte,
        tamanho_min=tamanho_min, tamanho_max=tamanho_max,
    )
    if not linhas:
        return

    altura_bloco = len(linhas) * altura_linha + (len(linhas) - 1) * espacamento
    centro_x = (x1 + x2) / 2
    y = y1 + ((y2 - y1) - altura_bloco) / 2

    # anchor="ma" = centralizado na horizontal, topo (ascender) na vertical:
    # o Pillow centraliza sozinho, então não remedimos cada linha aqui.
    for linha in linhas:
        draw.text((centro_x, y), linha, font=fonte, fill=cor, anchor="ma")
        y += altura_linha + espacamento
