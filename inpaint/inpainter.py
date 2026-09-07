"""
Etapa 4 - Remocao do texto original (inpainting).

A ideia central: a caixa do YOLO NAO e a mascara. A caixa diz "aqui tem
texto"; a mascara diz "estes pixels sao o texto". Pintar a caixa inteira
de branco apaga a borda do balao e a arte que a caixa pegou junto.

Tres rotas, da mais barata para a mais cara:
  solido  - preenche com a cor de fundo dominante. Perfeito em balao
            branco chapado, instantaneo, sem modelo.
  telea   - cv2.inpaint (Telea). Baseline classico, bom em mascara fina.
  lama    - rede neural (big-lama). Para texto sobre arte/screentone.

`remover_texto(..., metodo="auto")` decide a rota por regiao: balao
chapado nao precisa de rede neural.
"""

import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 1. Mascara do texto
# ---------------------------------------------------------------------------

def _componentes_alinhados(mascara, caixa_texto, tolerancia=0.45):
    """
    Mantem so os componentes com cara de glifo do bloco de texto.

    Componente ancorado = tem a maior parte da area dentro da caixa do
    detector. A altura mediana dos ancorados vira a referencia; vizinhos so
    entram se a altura for compativel E se estiverem alinhados em linha ou
    coluna com algum ancorado.

    Isso descarta o contorno do balao e as linhas do desenho, que sao
    componentes compridos e esparsos - o filtro por espessura de traco que
    havia antes aqui errava nos dois sentidos.

    Ideia adaptada de comic-translate (_select_text_like_components).
    """
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(mascara, 8)
    if n <= 1:
        return mascara, 0.0

    tx1, ty1, tx2, ty2 = caixa_texto
    ancorados, outros = [], []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx, cy = x + w / 2, y + h / 2
        dentro = (tx1 <= cx <= tx2) and (ty1 <= cy <= ty2)
        (ancorados if dentro else outros).append((i, x, y, w, h))

    if not ancorados:
        return mascara, 0.0

    altura_ref = float(np.median([h for _, _, _, _, h in ancorados]))
    mantidos = {i for i, _, _, _, _ in ancorados}

    # Vizinhos: mesma altura aproximada E alinhados com algum ancorado.
    alt_rec, larg_rec = mascara.shape
    for i, x, y, w, h in outros:
        # Componente cortado pela borda do recorte e arte vizinha, nao texto
        # deste bloco: a linha do balao entra por aqui.
        if x <= 1 or y <= 1 or x + w >= larg_rec - 1 or y + h >= alt_rec - 1:
            continue
        if not (altura_ref * (1 - tolerancia) <= h <= altura_ref * (1 + tolerancia)):
            continue
        cy, cx = y + h / 2, x + w / 2
        alinhado = any(
            abs(cy - (ay + ah / 2)) < altura_ref * 0.6 or
            abs(cx - (ax + aw / 2)) < altura_ref * 0.6
            for _, ax, ay, aw, ah in ancorados
        )
        if alinhado:
            mantidos.add(i)

    saida = np.zeros_like(mascara)
    for i in mantidos:
        saida[rotulos == i] = 255
    return saida, altura_ref


def interior_do_balao(recorte, caixa_texto, tolerancia=14, erosao=3):
    """
    Acha o interior do balao por flood fill em LAB (espaco perceptualmente
    uniforme) a partir do centro da caixa de texto, e erode o resultado para
    NAO encostar na linha do balao.

    Devolve a mascara do interior, ou None quando o preenchimento nao tem
    tamanho plausivel (vazou para a pagina ou nem cobriu o texto).

    Cascata adaptada de FrankYomik (_flood_fill_boundary).
    """
    h, w = recorte.shape[:2]
    x1, y1, x2, y2 = caixa_texto
    cx = int(np.clip((x1 + x2) // 2, 0, w - 1))
    cy = int(np.clip((y1 + y2) // 2, 0, h - 1))

    lab = cv2.cvtColor(recorte, cv2.COLOR_BGR2LAB)
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(lab.copy(), flood, (cx, cy), 0,
                  (tolerancia,) * 3, (tolerancia,) * 3,
                  cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    interior = flood[1:-1, 1:-1]

    # O flood preenche o fundo AO REDOR das letras, deixando cada glifo como
    # um buraco. Sem fechar esses buracos, a erosao seguinte os alarga e come
    # justamente a regiao que precisa ser apagada.
    contornos, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    interior = np.zeros_like(interior)
    cv2.drawContours(interior, contornos, -1, 255, -1)

    area_texto = max(1, (x2 - x1) * (y2 - y1))
    area = int(np.count_nonzero(interior))
    if area < area_texto * 0.5 or area > area_texto * 15:
        return None            # nao cobriu o texto, ou vazou para a pagina

    if erosao > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (erosao * 2 + 1, erosao * 2 + 1))
        interior = cv2.erode(interior, k)
    return interior


def gerar_mascara(recorte, caixa_texto=None, dilatacao=None, iteracoes=2,
                  area_min=4, fracao_area_max=0.35, caixas_ocr=None,
                  retornar_base=False):
    """
    Mascara binaria (0/255) do texto dentro do recorte.

    caixa_texto - (x1,y1,x2,y2) da deteccao, em coordenadas DO RECORTE.
                  Quando informada, ancora o filtro de componentes.
    dilatacao   - None = derivada do corpo do texto (30% da altura mediana
                  dos glifos, minimo 3), como no manga-image-translator.
    iteracoes   - repeticoes da dilatacao. 3 e o valor do comic-translate;
                  dilatacao curta deixa halo de antialiasing ("fantasma"),
                  que foi o defeito da versao anterior em pagina colorida.
    retornar_base - devolve tambem a mascara SEM dilatacao. A dilatada serve
                  para apagar; a crua serve para analisar o entorno do texto
                  (classificar_regiao, cor_de_fundo), porque o anel medido a
                  partir da mascara inflada ja cai fora do balao.
    """
    if recorte.size == 0:
        return np.zeros(recorte.shape[:2], np.uint8)

    cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    cinza = cv2.fastNlMeansDenoising(cinza, None, 5, 7, 21)

    escuro_no_claro = np.median(cinza) > 127
    modo = cv2.THRESH_BINARY_INV if escuro_no_claro else cv2.THRESH_BINARY
    _, binaria = cv2.threshold(cinza, 0, 255, modo + cv2.THRESH_OTSU)

    altura, largura = binaria.shape
    area_total = altura * largura

    # Descarte grosseiro antes de olhar alinhamento.
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(binaria, 8)
    limpa = np.zeros_like(binaria)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < area_min or area > area_total * fracao_area_max:
            continue
        if h > altura * 0.95 or w > largura * 0.95:
            continue
        limpa[rotulos == i] = 255

    if caixa_texto is None:
        caixa_texto = (0, 0, largura, altura)
    mascara, altura_glifo = _componentes_alinhados(limpa, caixa_texto)

    # Fecha buracos internos das letras antes de dilatar.
    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    if caixas_ocr:
        permitido = np.zeros_like(mascara)
        for (cx1, cy1, cx2, cy2) in caixas_ocr:
            cv2.rectangle(permitido, (cx1, cy1), (cx2, cy2), 255, -1)
        mascara = cv2.bitwise_and(mascara, permitido)

    base = mascara.copy()

    if dilatacao is None:
        dilatacao = int(np.clip(round(altura_glifo * 0.30), 3, 9))
    if dilatacao > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (dilatacao, dilatacao))
        mascara = cv2.dilate(mascara, k, iterations=iteracoes)

    # Contencao: dilatacao generosa sem limite come a linha do balao.
    # 1a opcao, o interior do balao; 2a, a caixa do detector com folga.
    limite = interior_do_balao(recorte, caixa_texto)
    if limite is None:
        folga = max(4, dilatacao)
        limite = np.zeros_like(mascara)
        cv2.rectangle(limite,
                      (max(0, caixa_texto[0] - folga), max(0, caixa_texto[1] - folga)),
                      (min(largura, caixa_texto[2] + folga),
                       min(altura, caixa_texto[3] + folga)), 255, -1)
    mascara = cv2.bitwise_and(mascara, limite)

    return (mascara, base) if retornar_base else mascara


def montar_caixas_ocr(dados_tesseract, confianca_min=30):
    """
    Converte pytesseract.image_to_data(..., output_type=DICT) em caixas
    utilizaveis por gerar_mascara(caixas_ocr=...).
    """
    caixas = []
    for i in range(len(dados_tesseract.get("text", []))):
        try:
            conf = float(dados_tesseract["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf < confianca_min or not dados_tesseract["text"][i].strip():
            continue
        x, y = dados_tesseract["left"][i], dados_tesseract["top"][i]
        w, h = dados_tesseract["width"][i], dados_tesseract["height"][i]
        caixas.append((x, y, x + w, y + h))
    return caixas


def _anel(mascara, perto=2, longe=6):
    """
    Faixa que circunda o texto sem encostar nele. Amostrar aqui, e nao no
    resto do recorte, mantem a medicao DENTRO do balao mesmo quando a caixa
    da deteccao e folgada ou o balao e irregular.
    """
    k = lambda it: cv2.dilate((mascara > 0).astype(np.uint8),
                              np.ones((5, 5), np.uint8), iterations=it) > 0
    return k(longe) & ~k(perto)


def classificar_regiao(recorte, mascara, tolerancia=12, cobertura_min=0.94,
                       limite_bordas=0.05):
    """
    'balao' = fundo atras do texto e uniforme -> preencher com a cor resolve.
    'arte'  = ha desenho/screentone atras -> precisa de inpaint de verdade.

    Criterio de uniformidade do comic-translate (>=94% dos pixels a no maximo
    12 niveis da mediana, por canal), medido no anel ao redor das letras.
    Mais robusto que densidade de bordas: nao depende de limiar de Canny e
    nao confunde fundo claro texturizado com fundo chapado.
    """
    selecao = _anel(mascara)
    amostra = recorte[selecao]
    if amostra.shape[0] < 64:
        return "arte"

    # Sinal 1: uniformidade de cor (criterio do comic-translate).
    mediana = np.median(amostra, axis=0)
    perto = np.all(np.abs(amostra.astype(np.float32) - mediana) <= tolerancia, axis=1)
    uniforme = perto.mean() >= cobertura_min

    # Sinal 2: ausencia de bordas. Fundo claro mas com linha fina passa no
    # teste de cor e reprova aqui.
    cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    bordas = cv2.Canny(cv2.GaussianBlur(cinza, (3, 3), 0), 60, 160)
    liso = (bordas[selecao] > 0).mean() < limite_bordas

    # Exige os dois: errar para "arte" custa tempo, errar para "balao"
    # estraga o desenho.
    return "balao" if (uniforme and liso) else "arte"


def cor_de_fundo(recorte, mascara):
    """Cor dominante do fundo, medida fora do halo do texto (nunca branco fixo)."""
    amostra = recorte[_anel(mascara)]
    if amostra.shape[0] < 16:
        amostra = recorte.reshape(-1, recorte.shape[-1])
    return np.median(amostra, axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Rotas de preenchimento
# ---------------------------------------------------------------------------

def inpaint_solido(recorte, mascara, base=None):
    """
    Preenche com a cor de fundo amostrada. Ideal para balao chapado.
    Nunca usa branco puro: pagina escaneada tem balao em 245-250, e branco
    puro cria um retangulo que brilha contra o papel.
    """
    saida = recorte.copy()
    saida[mascara > 0] = cor_de_fundo(recorte, base if base is not None else mascara)
    return saida


def inpaint_telea(recorte, mascara, raio=3):
    """cv2.inpaint classico - baseline de comparacao."""
    return cv2.inpaint(recorte, mascara, raio, cv2.INPAINT_TELEA)


def inpaint_ns(recorte, mascara, raio=3):
    """cv2.inpaint Navier-Stokes - o outro baseline do OpenCV."""
    return cv2.inpaint(recorte, mascara, raio, cv2.INPAINT_NS)


_lama = None


def _carregar_lama():
    """
    Carrega o LaMa uma unica vez.

    IMPORTANTE: o simple-lama-inpainting baixa o big-lama ORIGINAL, treinado
    em fotografia (Places/CelebA). Os projetos de referencia da area usam
    checkpoints fine-tunados em manga:
      manga-image-translator -> dreMaz/AnimeMangaInpainting (lama_large_512px)
      koharu                 -> mayocream/lama-manga, mayocream/aot-inpainting
    Para usar outro modelo TorchScript, aponte a variavel de ambiente
    LAMA_MODEL para o arquivo .pt antes de rodar - o simple-lama a respeita.
    """
    global _lama
    if _lama is None:
        from simple_lama_inpainting import SimpleLama  # import tardio
        _lama = SimpleLama()
    return _lama


def _ajustar_para_modulo(img, modulo=8):
    """LaMa exige lados multiplos de 8."""
    h, w = img.shape[:2]
    nh = h + (modulo - h % modulo) % modulo
    nw = w + (modulo - w % modulo) % modulo
    if (nh, nw) == (h, w):
        return img
    interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp)


def inpaint_lama(recorte, mascara, tamanho=512):
    """
    LaMa. Segue o pre-processamento dos projetos de referencia:
    redimensiona mantendo proporcao ate `tamanho`, ajusta para multiplo de 8,
    zera a area mascarada antes da inferencia e devolve a saida SO onde a
    mascara esta - assim o modelo nao altera de leve o resto do recorte.
    """
    from PIL import Image

    lama = _carregar_lama()
    h, w = recorte.shape[:2]

    escala = tamanho / max(h, w) if max(h, w) > tamanho else 1.0
    if escala != 1.0:
        nw, nh = max(1, int(w * escala)), max(1, int(h * escala))
        peq = cv2.resize(recorte, (nw, nh), interpolation=cv2.INTER_AREA)
        m_peq = cv2.resize(mascara, (nw, nh), interpolation=cv2.INTER_NEAREST)
    else:
        peq, m_peq = recorte, mascara

    peq = _ajustar_para_modulo(peq)
    m_peq = _ajustar_para_modulo(m_peq)
    m_peq = ((m_peq > 127) * 255).astype(np.uint8)

    # Zera o que sera reconstruido (o modelo nao deve ver o texto).
    peq = peq.copy()
    peq[m_peq > 0] = 0

    rgb = Image.fromarray(cv2.cvtColor(peq, cv2.COLOR_BGR2RGB))
    resultado = lama(rgb, Image.fromarray(m_peq))

    saida = cv2.cvtColor(np.array(resultado), cv2.COLOR_RGB2BGR)
    if saida.shape[:2] != (h, w):
        saida = cv2.resize(saida, (w, h), interpolation=cv2.INTER_LINEAR)

    final = recorte.copy()
    final[mascara > 0] = saida[mascara > 0]
    return final


METODOS = {
    "solido": inpaint_solido,
    "telea": inpaint_telea,
    "ns": inpaint_ns,
    "lama": inpaint_lama,
}


# ---------------------------------------------------------------------------
# 3. API do pipeline
# ---------------------------------------------------------------------------

def limpar_regiao(imagem, caixa, metodo="auto", margem=8, dilatacao=None):
    """
    Remove o texto de UMA caixa, direto na imagem (BGR, in-place na copia).
    A margem amplia o recorte: o inpaint precisa de contexto ao redor.

    Devolve (imagem_alterada, informacoes).
    """
    x1, y1, x2, y2 = caixa
    h, w = imagem.shape[:2]
    ax1, ay1 = max(0, x1 - margem), max(0, y1 - margem)
    ax2, ay2 = min(w, x2 + margem), min(h, y2 + margem)

    recorte = imagem[ay1:ay2, ax1:ax2]
    if recorte.size == 0:
        return imagem, {"tipo": "vazio", "metodo": None, "tempo": 0.0}

    # caixa da deteccao em coordenadas do recorte: ancora o filtro de glifos
    caixa_local = (x1 - ax1, y1 - ay1, x2 - ax1, y2 - ay1)
    mascara, base = gerar_mascara(recorte, caixa_texto=caixa_local,
                                  dilatacao=dilatacao, retornar_base=True)
    tipo = classificar_regiao(recorte, base)  # analisa pela mascara crua

    if metodo == "auto":
        escolhido = "solido" if tipo == "balao" else "lama"
    else:
        escolhido = metodo

    inicio = time.perf_counter()
    try:
        if escolhido == "solido":
            # cor amostrada pelo anel da mascara CRUA (dentro do balao)
            limpo = inpaint_solido(recorte, mascara, base=base)
        else:
            limpo = METODOS[escolhido](recorte, mascara)
    except Exception as e:
        # LaMa indisponivel (pacote/modelo ausente): cai para o Telea.
        print(f"    [inpaint] {escolhido} falhou ({type(e).__name__}), usando telea")
        escolhido = "telea"
        limpo = inpaint_telea(recorte, mascara)
    tempo = time.perf_counter() - inicio

    imagem[ay1:ay2, ax1:ax2] = limpo
    cobertura = float((mascara > 0).mean())
    return imagem, {"tipo": tipo, "metodo": escolhido, "tempo": tempo,
                    "cobertura": cobertura}


def remover_texto(imagem, caixas, metodo="auto", margem=8, dilatacao=None,
                  verbose=True):
    """
    Limpa todas as caixas de uma pagina. Devolve (imagem_limpa, relatorio).
    """
    saida = imagem.copy()
    relatorio = []
    for i, caixa in enumerate(caixas):
        saida, info = limpar_regiao(saida, caixa, metodo, margem, dilatacao)
        info["regiao"] = i
        relatorio.append(info)
        if verbose:
            print(f"    regiao {i}: {info['tipo']:<6} -> {info['metodo']:<7} "
                  f"({info['tempo'] * 1000:.0f} ms)")
    return saida, relatorio
