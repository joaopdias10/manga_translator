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
  7. Quebra de linha balanceada (programação dinâmica, estilo Knuth-Plass)
     no lugar da gulosa: linhas de comprimento parecido, sem a última curta.
"""

from functools import lru_cache

from PIL import ImageFont

# Tamanho usado só para medir proporções. Métricas TrueType são lineares
# no corpo da fonte, então medir aqui e escalar é fiel o bastante para a
# busca - e no fim ainda existe uma verificação exata.
_TAMANHO_REF = 100

TAMANHO_MIN = 12
TAMANHO_MAX = 80

# Piso absoluto: quando o texto nao cabe nem em TAMANHO_MIN, e melhor encolher
# abaixo do minimo "confortavel" do que deixar o bloco vazar por cima da arte
# e da borda do painel. Texto pequeno que cabe no balao > texto legivel que
# invade o quadro vizinho.
_TAMANHO_PISO = 7


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


def _quebrar_trecho(palavras, larguras, largura_espaco, max_largura):
    """
    Quebra balanceada, no estilo Knuth-Plass (programacao dinamica).

    A quebra gulosa enche cada linha ao maximo e deixa a ultima curta
    ("O QUE VOCE ESTA / FAZENDO?"). Aqui cada linha custa (sobra/largura)^3
    e escolhe-se o conjunto de quebras de menor custo total: o cubo pune
    muito uma linha quase vazia e pouco varias linhas levemente curtas, o
    que da blocos de linhas parecidas - o formato que assenta num balao.
    Linha a mais acrescenta uma largura inteira de sobra, entao a DP nao
    usa mais linhas que a gulosa a nao ser em casos degenerados.

    Pressupoe que toda palavra cabe sozinha na largura (o _quebrar fatia
    as que nao cabem antes de chegar aqui).
    """
    n = len(palavras)
    if n == 0:
        return []

    custo = [0.0] + [float("inf")] * n
    inicio = [0] * (n + 1)  # inicio[i] = primeira palavra da linha que termina em i-1

    for i in range(1, n + 1):
        largura = -largura_espaco
        for j in range(i - 1, -1, -1):
            largura += larguras[j] + largura_espaco
            if largura > max_largura and j < i - 1:
                break
            sobra = max(0.0, max_largura - largura) / max_largura
            total = custo[j] + sobra ** 3
            if total < custo[i]:
                custo[i], inicio[i] = total, j

    linhas = []
    i = n
    while i > 0:
        j = inicio[i]
        linhas.append(" ".join(palavras[j:i]))
        i = j
    linhas.reverse()
    return linhas


def _quebrar(palavras, largura_de, largura_espaco, max_largura):
    """
    Monta as linhas a partir das larguras já conhecidas.
    `largura_de` é uma função texto -> largura em pixels.

    Palavra sozinha maior que o balão vira quebra forçada: é fatiada, cada
    pedaço menos o último ocupa uma linha inteira, e o último começa o
    trecho seguinte. Cada trecho entre quebras forçadas é balanceado à parte.
    """
    linhas = []
    trecho = []

    for palavra in palavras:
        if largura_de(palavra) > max_largura:
            linhas.extend(_quebrar_trecho(
                trecho, [largura_de(p) for p in trecho], largura_espaco, max_largura))
            pedacos = _fatiar_palavra(palavra, largura_de, max_largura)
            linhas.extend(pedacos[:-1])
            trecho = [pedacos[-1]]
        else:
            trecho.append(palavra)

    linhas.extend(_quebrar_trecho(
        trecho, [largura_de(p) for p in trecho], largura_espaco, max_largura))
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

    # Verificação exata com a fonte real (a escala de referência tem erro de
    # arredondamento de subpixel). Se estourou, desce de 1 em 1 até caber -
    # inclusive ABAIXO de tamanho_min, até o piso absoluto: texto que vaza por
    # cima do painel de cima e pior que texto miúdo que cabe no balão.
    piso = min(tamanho_min, _TAMANHO_PISO)
    tamanho = max(tamanho, piso)
    for _ in range(tamanho - piso + 1):
        fonte = carregar_fonte(caminho_fonte, tamanho)
        ascent, descent = fonte.getmetrics()
        altura_linha = ascent + descent
        espacamento = descent * 0.5
        linhas = wrap_text(texto, largura_box, fonte)
        total = len(linhas) * altura_linha + (len(linhas) - 1) * espacamento
        if total <= altura_box or tamanho <= piso:
            return fonte, linhas, altura_linha, espacamento
        tamanho -= 1

    return fonte, linhas, altura_linha, espacamento


# --------------------------------------------------------------------------
# Cor do texto
# --------------------------------------------------------------------------

PRETO = (0, 0, 0)
BRANCO = (255, 255, 255)


def cores_para_fundo(fundo_bgr, com_contorno=False):
    """
    (cor, contorno) do texto para o fundo que a etapa 4 deixou.

    Balao preto (ou arte escura) pede texto branco; o resto, preto. Com
    `com_contorno`, devolve tambem a cor oposta para o contorno - usado
    quando o fundo nao e chapado (texto sobre arte), onde a mediana sozinha
    nao garante contraste em todos os pontos.
    """
    if fundo_bgr is None:
        return PRETO, (BRANCO if com_contorno else None)
    b, g, r = fundo_bgr
    escuro = 0.299 * r + 0.587 * g + 0.114 * b < 128  # luminancia (Rec. 601)
    cor, oposta = (BRANCO, PRETO) if escuro else (PRETO, BRANCO)
    return cor, (oposta if com_contorno else None)


# --------------------------------------------------------------------------
# Desenho
# --------------------------------------------------------------------------

def spell(draw, texto, x1, y1, x2, y2, caminho_fonte="arial.ttf",
          cor=PRETO, margem_x=0.85, padding_y=10,
          tamanho_min=TAMANHO_MIN, tamanho_max=TAMANHO_MAX, contorno=None):
    """
    Escreve o texto centralizado dentro da caixa (x1, y1) - (x2, y2).

    contorno - cor do contorno das letras (None = sem contorno). A espessura
               acompanha o corpo da fonte; a folga de margem_x/padding_y
               absorve o que ela acrescenta.
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

    # Se o bloco ainda for mais alto que o balão (texto longo demais mesmo no
    # piso), a centralização jogaria o topo para cima de y1, invadindo a arte
    # e o painel de cima. Prende no topo da caixa: o excesso vaza só para
    # baixo, dentro do próprio balão/painel.
    y = max(y1 + padding_y, y)

    # anchor="ma" = centralizado na horizontal, topo (ascender) na vertical:
    # o Pillow centraliza sozinho, então não remedimos cada linha aqui.
    espessura = 0
    if contorno is not None:
        espessura = max(1, round(getattr(fonte, "size", TAMANHO_MIN) * 0.08))

    for linha in linhas:
        draw.text((centro_x, y), linha, font=fonte, fill=cor, anchor="ma",
                  stroke_width=espessura, stroke_fill=contorno)
        y += altura_linha + espacamento
