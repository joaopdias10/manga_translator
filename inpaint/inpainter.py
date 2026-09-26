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
        vazia = np.zeros(recorte.shape[:2], np.uint8)
        return (vazia, vazia.copy()) if retornar_base else vazia

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
        # lado impar: MORPH_ELLIPSE de lado par nao tem centro e a mascara
        # "anda" numa direcao fixa a cada iteracao da dilatacao.
        dilatacao = int(np.clip(round(altura_glifo * 0.30), 3, 9)) | 1
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


# ---------------------------------------------------------------------------
# Backends de rede neural (LaMa / AOT-GAN / MI-GAN)
#
# Nenhum peso vem no repositorio; baixe o que for usar para inpaint/models/
# (ou aponte a variavel de ambiente indicada). Todos sao fine-tunados em
# manga - o LaMa generico do simple-lama e treinado em fotografia e nao
# serve para arte 1-bit.
#
#   lama       anime-manga-big-lama.pt   (TorchScript)  -> PADRAO da rota "arte"
#              github.com/Sanster/models  release AnimeMangaInpainting
#   lama_onnx  lama-manga-dynamic.onnx    (ONNX)   HF ogkalu/lama-manga-onnx-dynamic
#   aot        aot.onnx                   (ONNX)   HF ogkalu/aot-inpainting
#   migan      migan_pipeline_v2.onnx     (ONNX)   github.com/Sanster/models release migan
#
# torch ja e dependencia do projeto (ultralytics). Os backends ONNX pedem
# `pip install onnxruntime`.
# ---------------------------------------------------------------------------

import os

_MODELOS_DIR = os.path.join(os.path.dirname(__file__), "models")

# engine: torch|onnx | norm: "0_1" (LaMa) / "-1_1" (AOT) / "migan"
# zera: aplicar img*(1-mask) antes da inferencia | modulo: multiplo exigido
_BACKENDS = {
    "lama":      dict(engine="torch", arquivo="anime-manga-big-lama.pt",
                      env="LAMA_MODEL",  norm="0_1",   zera=False, modulo=8),
    "lama_onnx": dict(engine="onnx",  arquivo="lama-manga-dynamic.onnx",
                      env="LAMA_ONNX",   norm="0_1",   zera=False, modulo=8),
    "aot":       dict(engine="onnx",  arquivo="aot.onnx",
                      env="AOT_ONNX",    norm="-1_1",  zera=True,  modulo=8),
    "migan":     dict(engine="onnx",  arquivo="migan_pipeline_v2.onnx",
                      env="MIGAN_ONNX",  norm="migan", zera=True,  modulo=512,
                      quadrado=True),
}

_modelos = {}  # cache: nome -> (modelo, cfg)


def _obter_backend(nome):
    """Carrega (uma vez) o modelo do backend. Levanta se o arquivo faltar."""
    if nome in _modelos:
        return _modelos[nome]
    cfg = _BACKENDS[nome]
    caminho = os.environ.get(cfg["env"]) or os.path.join(_MODELOS_DIR, cfg["arquivo"])
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"modelo '{nome}' nao encontrado em {caminho}. "
            f"Baixe para inpaint/models/ ou defina a variavel {cfg['env']}.")
    if cfg["engine"] == "torch":
        import torch
        modelo = torch.jit.load(caminho, map_location="cpu").eval()
    else:
        import onnxruntime as ort
        modelo = ort.InferenceSession(caminho, providers=["CPUExecutionProvider"])
    _modelos[nome] = (modelo, cfg)
    return _modelos[nome]


def _ajustar_para_modulo(img, modulo=8):
    """Ajusta os lados para multiplo de `modulo` (exigencia dos modelos)."""
    h, w = img.shape[:2]
    nh = h + (modulo - h % modulo) % modulo
    nw = w + (modulo - w % modulo) % modulo
    if (nh, nw) == (h, w):
        return img
    interp = cv2.INTER_NEAREST if img.ndim == 2 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp)


def _preparar(recorte, mascara, cfg, tamanho):
    """Redimensiona mantendo proporcao e ajusta forma. Devolve (rgb, mask01)."""
    h, w = recorte.shape[:2]
    if cfg.get("quadrado"):
        lado = cfg["modulo"]  # MI-GAN: quadrado fixo (512)
        rec = cv2.resize(recorte, (lado, lado), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(mascara, (lado, lado), interpolation=cv2.INTER_NEAREST)
    else:
        escala = tamanho / max(h, w) if max(h, w) > tamanho else 1.0
        if escala != 1.0:
            nw, nh = max(1, int(w * escala)), max(1, int(h * escala))
            rec = cv2.resize(recorte, (nw, nh), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(mascara, (nw, nh), interpolation=cv2.INTER_NEAREST)
        else:
            rec, msk = recorte, mascara
        rec = _ajustar_para_modulo(rec, cfg["modulo"])
        msk = _ajustar_para_modulo(msk, cfg["modulo"])
    rgb = cv2.cvtColor(rec, cv2.COLOR_BGR2RGB)
    msk = ((msk > 127) * 255).astype(np.uint8)
    return rgb, msk


def _rodar_modelo(nome, recorte, mascara, tamanho=512):
    """
    Passa o recorte por um dos modelos e devolve o BGR reconstruido no
    tamanho original, escrevendo SO onde a mascara esta (o resto do recorte
    fica intocado). Convencoes copiadas do comic-translate.
    """
    modelo, cfg = _obter_backend(nome)
    h, w = recorte.shape[:2]
    rgb, msk = _preparar(recorte, mascara, cfg, tamanho)
    m01 = (msk > 0).astype(np.float32)

    if cfg["norm"] == "migan":
        # pipeline MI-GAN: uint8, mascara 0=apagar / 255=manter
        conhecido = np.where(msk > 120, 0, 255).astype(np.uint8)
        img_nchw = np.transpose(rgb, (2, 0, 1))[None].astype(np.uint8)
        mask_nchw = conhecido[None, None]
        nomes = [i.name for i in modelo.get_inputs()]
        saida = modelo.run(None, {nomes[0]: img_nchw, nomes[1]: mask_nchw})[0]
        out_rgb = np.transpose(saida[0], (1, 2, 0)).astype(np.uint8)

    elif cfg["engine"] == "onnx":
        if cfg["norm"] == "0_1":
            img = rgb.astype(np.float32) / 255.0
        else:  # -1_1
            img = rgb.astype(np.float32) / 127.5 - 1.0
        if cfg["zera"]:
            img = img * (1 - m01[..., None])
        img_nchw = np.transpose(img, (2, 0, 1))[None]
        mask_nchw = m01[None, None]
        nomes = [i.name for i in modelo.get_inputs()]
        saida = modelo.run(None, {nomes[0]: img_nchw, nomes[1]: mask_nchw})[0]
        arr = saida[0].transpose(1, 2, 0)
        if cfg["norm"] == "0_1":
            out_rgb = np.clip(arr * 255, 0, 255).astype(np.uint8)
        else:
            out_rgb = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)

    else:  # torch (LaMa / AOT TorchScript)
        import torch
        if cfg["norm"] == "0_1":
            img_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        else:
            img_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        mask_t = torch.from_numpy(m01).unsqueeze(0).unsqueeze(0)
        if cfg["zera"]:
            img_t = img_t * (1 - mask_t)
        with torch.inference_mode():
            saida = modelo(img_t, mask_t)
        arr = saida[0].float().permute(1, 2, 0).cpu().numpy()
        if cfg["norm"] == "0_1":
            out_rgb = np.clip(arr * 255, 0, 255).astype(np.uint8)
        else:
            out_rgb = np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)

    saida_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    if saida_bgr.shape[:2] != (h, w):
        saida_bgr = cv2.resize(saida_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    final = recorte.copy()
    final[mascara > 0] = saida_bgr[mascara > 0]
    return final


def _fazer_metodo(nome):
    """Cria a funcao inpaint_<nome>(recorte, mascara) para o METODOS."""
    def _m(recorte, mascara):
        return _rodar_modelo(nome, recorte, mascara)
    _m.__name__ = f"inpaint_{nome}"
    return _m


inpaint_lama = _fazer_metodo("lama")            # PADRAO da rota "arte"
inpaint_lama_onnx = _fazer_metodo("lama_onnx")
inpaint_aot = _fazer_metodo("aot")
inpaint_migan = _fazer_metodo("migan")


METODOS = {
    "solido": inpaint_solido,
    "telea": inpaint_telea,
    "ns": inpaint_ns,
    "lama": inpaint_lama,
    "lama_onnx": inpaint_lama_onnx,
    "aot": inpaint_aot,
    "migan": inpaint_migan,
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

    # Sem pixels de texto na mascara: nao ha o que apagar. Sair aqui evita
    # rodar o modelo caro a toa (o classificar_regiao devolveria "arte" ->
    # "lama" e o inpaint nao mudaria um pixel) e deixa explicito no relatorio
    # que a regiao ficou intocada - texto residual atras do lettering.
    if not mascara.any():
        return imagem, {"tipo": "vazio", "metodo": "nenhum", "tempo": 0.0,
                        "cobertura": 0.0}

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
    except (FileNotFoundError, ImportError, OSError, RuntimeError) as e:
        # Modelo/runtime indisponivel (peso ou pacote ausente, erro de
        # inferencia): cai para o Telea. Nao capturamos Exception generico
        # para nao mascarar bugs de logica do _rodar_modelo como "modelo
        # indisponivel".
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
