from PIL import ImageFont

def wrap_text(texto, max_width, font):
    """
    Quebra o texto medindo a largura real de cada palavra em pixels,
    garantindo que a linha ocupe o máximo de espaço sem estourar.
    """
    words = texto.split()
    if not words:
        return []
    
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        # Mede o tamanho da linha em pixels usando a fonte atual
        w = font.getlength(test_line)
        
        if w <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word)
                current_line = []
    
    if current_line:
        lines.append(' '.join(current_line))
        
    return lines

def best_font(texto, largura_box, altura_box, caminho_fonte):
    """
    Calcula dinamicamente o maior tamanho de fonte possível
    que caiba dentro do balão de fala, forçando quebra de linhas reais.
    """
    tamanho_min = 12
    tamanho_max = 80 # Limite ajustado para mangás

    # Começa do maior tamanho possível e vai reduzindo
    for tamanho in range(tamanho_max, tamanho_min, -2):
        try:
            font = ImageFont.truetype(caminho_fonte, size=tamanho)
        except IOError:
            font = ImageFont.load_default()

        # Usa o wrap inteligente para medir com a fonte exata
        linhas = wrap_text(texto, largura_box, font)

        ascent, descent = font.getmetrics()
        altura_linha = ascent + descent
        
        # Altura total considera as linhas e um pequeno espaçamento entre elas
        espacamento = descent * 0.5
        altura_total = (len(linhas) * altura_linha) + ((len(linhas) - 1) * espacamento)

        # Se a altura total couber na altura útil do balão, achamos a fonte!
        if altura_total <= altura_box:
            return font, linhas, altura_linha, espacamento

    # Fallback mínimo caso o texto seja gigantesco
    try:
        font = ImageFont.truetype(caminho_fonte, size=tamanho_min)
    except IOError:
        font = ImageFont.load_default()
        
    return font, wrap_text(texto, largura_box, font), 15, 2


def spell(draw, texto, x1, y1, x2, y2, caminho_fonte="arial.ttf"):
    """
    Recebe as coordenadas do balão, calcula a fonte e desenha o texto centralizado.
    Mantém o padrão original do seu script.
    """
    if not texto.strip():
        return

    # --- CONFIGURAÇÃO DE MARGENS ---
    # Usa 85% da largura para afastar o texto das bordas arredondadas do balão
    largura_util = (x2 - x1) * 0.85 
    padding_y = 10
    altura_util = (y2 - y1) - (padding_y * 2)

    if largura_util <= 0 or altura_util <= 0:
        return

    # Calcula fonte e quebras
    font, linhas, altura_linha, espacamento = best_font(
        texto,
        largura_util,
        altura_util,
        caminho_fonte,
    )

    # Centralização vertical precisa do bloco inteiro de texto
    altura_bloco = (len(linhas) * altura_linha) + ((len(linhas) - 1) * espacamento)
    y_atual = y1 + ( (y2 - y1) - altura_bloco ) / 2 

    # Desenha texto linha por linha
    for linha in linhas:
        largura_linha = font.getlength(linha)
        # Centralização horizontal precisa para cada linha individual
        x_atual = x1 + ( (x2 - x1) - largura_linha ) / 2

        draw.text(
            (x_atual, y_atual),
            linha,
            font=font,
            fill=(0, 0, 0),
        )

        y_atual += altura_linha + espacamento