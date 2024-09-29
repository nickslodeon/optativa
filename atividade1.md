import cv2  # Biblioteca OpenCV para processamento de imagens
import numpy as np  # Biblioteca NumPy para operações matemáticas avançadas
from matplotlib import pyplot as plt  # Biblioteca Matplotlib para plotagem de gráficos e imagens

# Carregar a imagem original em cores
imagem = cv2.imread('D:\\atividade\\gunter.jpg', cv2.IMREAD_COLOR)

# Verificar se a imagem foi carregada corretamente
if imagem is None:
    print("Erro ao carregar a imagem.")
    exit()

# 1. Mudança de brilho e contraste
def ajustar_brilho_contraste(imagem, alpha, beta):
    """
    Ajusta o brilho e o contraste da imagem.
    :param imagem: imagem de entrada
    :param alpha: ganho de contraste (1.0-3.0)
    :param beta: valor de brilho (0-100)
    :return: imagem com brilho e contraste ajustados
    """
    # cv2.convertScaleAbs aplica a transformação: new_image = alpha * image + beta
    nova_imagem = cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)
    return nova_imagem

# 2. Redimensionamento
def redimensionar_imagem(imagem, largura, altura):
    """
    Redimensiona a imagem para as dimensões especificadas.
    :param imagem: imagem de entrada
    :param largura: nova largura desejada
    :param altura: nova altura desejada
    :return: imagem redimensionada
    """
    # cv2.resize redimensiona a imagem usando a interpolação especificada
    nova_imagem = cv2.resize(imagem, (largura, altura), interpolation=cv2.INTER_AREA)
    return nova_imagem

# 3. Rotação
def rotacionar_imagem(imagem, angulo):
    """
    Rotaciona a imagem em torno do seu centro.
    :param imagem: imagem de entrada
    :param angulo: ângulo de rotação em graus
    :return: imagem rotacionada
    """
    # Obtém as dimensões da imagem
    altura, largura = imagem.shape[:2]
    # Calcula o centro da imagem
    centro = (largura / 2, altura / 2)
    # Obtém a matriz de rotação
    matriz_rotacao = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    # Aplica a transformação de rotação
    imagem_rotacionada = cv2.warpAffine(imagem, matriz_rotacao, (largura, altura))
    return imagem_rotacionada

# 4. Corte
def cortar_imagem(imagem, x_inicio, y_inicio, largura, altura):
    """
    Corta uma região retangular da imagem.
    :param imagem: imagem de entrada
    :param x_inicio: coordenada x inicial
    :param y_inicio: coordenada y inicial
    :param largura: largura do corte
    :param altura: altura do corte
    :return: imagem cortada
    """
    # Utiliza slicing do NumPy para cortar a imagem
    imagem_cortada = imagem[y_inicio:y_inicio+altura, x_inicio:x_inicio+largura]
    return imagem_cortada

# 5. Filtragem para suavizar (blur) e realçar (sharpen)
def filtrar_imagem(imagem, tipo):
    """
    Aplica um filtro à imagem para suavizar ou realçar.
    :param imagem: imagem de entrada
    :param tipo: 'suavizar' para aplicar blur, 'realcar' para aplicar sharpening
    :return: imagem filtrada
    """
    if tipo == 'suavizar':
        # Aplica um filtro Gaussiano para suavização
        imagem_filtrada = cv2.GaussianBlur(imagem, (5, 5), 0)
    elif tipo == 'realcar':
        # Define um kernel para realçar a imagem
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        # Aplica o filtro personalizado usando o kernel
        imagem_filtrada = cv2.filter2D(imagem, -1, kernel)
    else:
        # Se o tipo não for reconhecido, retorna a imagem original
        imagem_filtrada = imagem
    return imagem_filtrada

# 6. Segmentação (usando limiarização)
def segmentar_imagem(imagem):
    """
    Segmenta a imagem usando limiarização binária.
    :param imagem: imagem de entrada
    :return: imagem segmentada em tons binários
    """
    # Converte a imagem para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    # Aplica limiarização binária com um limite fixo
    ret, imagem_segmentada = cv2.threshold(imagem_cinza, 127, 255, cv2.THRESH_BINARY)
    return imagem_segmentada

# 7. Equalização de histograma
def equalizar_histograma(imagem):
    """
    Equaliza o histograma da imagem em escala de cinza.
    :param imagem: imagem de entrada
    :return: imagem com histograma equalizado
    """
    # Converte a imagem para escala de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    # Aplica a equalização de histograma
    imagem_equalizada = cv2.equalizeHist(imagem_cinza)
    return imagem_equalizada

# Aplicando as operações definidas acima
# 1. Ajuste de brilho e contraste
imagem_brilho_contraste = ajustar_brilho_contraste(imagem, alpha=1.5, beta=50)

# 2. Redimensionamento para largura=400 e altura=300
imagem_redimensionada = redimensionar_imagem(imagem, largura=400, altura=300)

# 3. Rotação de 45 graus
imagem_rotacionada = rotacionar_imagem(imagem, angulo=45)

# 4. Corte de uma região da imagem
imagem_cortada = cortar_imagem(imagem, x_inicio=50, y_inicio=50, largura=200, altura=200)

# 5. Filtragem para suavizar
imagem_suavizada = filtrar_imagem(imagem, tipo='suavizar')

# 6. Filtragem para realçar
imagem_realcada = filtrar_imagem(imagem, tipo='realcar')

# 7. Segmentação da imagem
imagem_segmentada = segmentar_imagem(imagem)

# 8. Equalização do histograma
imagem_equalizada = equalizar_histograma(imagem)

# Exibindo as imagens resultantes usando Matplotlib
titulos = ['Original', 'Brilho/Contraste', 'Redimensionada', 'Rotacionada', 'Cortada',
           'Suavizada', 'Realçada', 'Segmentada', 'Equalizada']
imagens = [imagem, imagem_brilho_contraste, imagem_redimensionada, imagem_rotacionada,
           imagem_cortada, imagem_suavizada, imagem_realcada, imagem_segmentada, imagem_equalizada]

# Configura o tamanho da figura para melhor visualização
plt.figure(figsize=(12, 8))

for i in range(len(imagens)):
    plt.subplot(3, 3, i+1)
    if len(imagens[i].shape) == 2:
        # Imagem em escala de cinza
        plt.imshow(imagens[i], cmap='gray')
    else:
        # Converter de BGR (OpenCV) para RGB (Matplotlib)
        plt.imshow(cv2.cvtColor(imagens[i], cv2.COLOR_BGR2RGB))
    plt.title(titulos[i])
    plt.axis('off')  # Oculta os eixos para melhor visualização

plt.tight_layout()  # Ajusta o espaçamento entre os subplots
plt.show()  # Exibe a janela com as imagens
