import pygame, sys, os, copy, json, numpy as np
from random import randint, randrange

class Individuo:
    """Representa um indivíduo (ou solução) em um algoritmo evolutivo, com pesos e biases que 
    definem a configuração de uma rede neural, além de um valor de fitness que avalia sua performance."""
    def __init__(self, pesos:list, bias:list):
        """Inicializa o indivíduo com uma lista de pesos e uma lista de biases, 
        e define o valor inicial de fitness como 0."""
        self.pesos = pesos
        self.bias = bias
        self.fitness = 0

class RedeNeural:
    """Representa uma rede neural com camadas de entrada, camadas escondidas e camada de saída, 
    contendo métodos para inicialização, mutação e cálculo da previsão do modelo."""
    def __init__(self, camada_entrada:int, camadas_escondida:list, camada_saida:int, descricao:list):
        """Inicializa a rede neural com as dimensões das camadas de entrada, escondida e saída, 
        e cria uma lista com o número de neurônios em cada camada."""
        self.camada_entrada = camada_entrada
        self.camadas_escondida = camadas_escondida
        self.camada_saida = camada_saida
        self.descricao = descricao
        self.lista_pontos = [0]
        self.lista_neuronios = [self.camada_entrada]

        for camada in self.camadas_escondida:
            self.lista_neuronios.append(camada)
        self.lista_neuronios.append(self.camada_saida)

    def relu(self, x) -> np.ndarray:
        """Aplica a função de ativação ReLU (Rectified Linear Unit), que retorna o valor de entrada 
        se for positivo e 0 se for negativo."""
        return np.maximum(0, x)

    def neuronio(self, inputs:np.ndarray, pesos:list, bias:list) -> np.ndarray:
        """Calcula a saída de um neurônio aplicando a soma ponderada dos inputs, pesos e bias, 
        seguida da função de ativação ReLU."""
        soma_ponderada = np.dot(inputs, pesos) + bias
        return self.relu(soma_ponderada)

    def individuo_random(self) -> Individuo:
        """Gera um indivíduo (modelo de rede neural) com pesos e biases aleatórios para cada camada 
        e retorna o objeto Individuo com esses parâmetros."""
        pesos = []
        bias = []

        # Inicializando a primeira camada escondida
        pesos.append(np.random.randn(self.camada_entrada, self.camadas_escondida[0]))
        bias.append(np.random.randn(self.camadas_escondida[0]))

        # Inicializando as camadas escondidas subsequentes
        for i in range(1, len(self.camadas_escondida)):
            pesos.append(np.random.randn(self.camadas_escondida[i-1], self.camadas_escondida[i]))  # Pesos de uma camada escondida para a próxima
            bias.append(np.random.randn(self.camadas_escondida[i]))  # Viés para cada camada escondida

        # Inicializando a camada de saída
        pesos.append(np.random.randn(self.camadas_escondida[-1], self.camada_saida))  # Pesos da última camada escondida para a camada de saída
        bias.append(np.random.randn(self.camada_saida))  # Viés para a camada de saída

        return Individuo(pesos, bias)

    def mutacao(self, individuo:Individuo, taxa_mutacao:float) -> Individuo:
        """Aplica mutação nos pesos e biases de um indivíduo com uma determinada taxa de mutação, 
        alterando aleatoriamente valores em seus parâmetros."""
        pai = copy.deepcopy(individuo)
        pesos = pai.pesos
        bias = pai.bias

        for camada in range(len(pesos)):
            for neuronio in range(len(pesos[camada])):
                for peso in range(len(pesos[camada][neuronio])):
                    if np.random.rand() < taxa_mutacao:
                        pesos[camada][neuronio][peso] += np.random.randn() * 0.2

            for neuronio in range(len(bias[camada])):
                if np.random.rand() < taxa_mutacao:
                    bias[camada][neuronio] += np.random.randn() * 0.2

        return Individuo(pesos, bias)

    def forward(self, entradas:list, individuo:Individuo) -> list:
        """Realiza a propagação para frente (feedforward) na rede neural, calculando a saída da rede 
        com base nas entradas e parâmetros (pesos e biases) do indivíduo."""
        pesos = individuo.pesos
        bias = individuo.bias
        resultado = []

        x = np.array(entradas)  # A entrada inicial

        # Passando pela primeira camada escondida até a última camada escondida
        for i in range(len(self.camadas_escondida)):
            x = self.neuronio(x, pesos[i], bias[i])  # Passa pela função de ativação
            resultado.append(x)

        # Passando pela camada de saída
        x = self.neuronio(x, pesos[-1], bias[-1])
        resultado.append(x)

        return resultado  # Retorna a saída da rede (previsão do indivíduo)

    def draw(self, surface:pygame.surface.Surface, entradas:list, saidas:list, posicao:tuple, escala_y:int):
        """Desenha a estrutura da rede neural (camadas de neurônios, entradas, saídas) em uma superfície 
        do Pygame, incluindo conexões entre neurônios com base nos valores das entradas e saídas."""
        origem_x, origem_y = posicao
        quadrado_x = 400
        quadrado_y = 300
        len_lista_pontos = len(self.lista_pontos)
        inc_y = int(quadrado_y/10)
        inc_x = int(quadrado_x/10)
        posicao_y = origem_y
        posicao_x = origem_x

        for indice in range(11):
            if indice == 0 or indice == 10:
                pygame.draw.line(surface, PRETO, (origem_x,posicao_y), (origem_x+quadrado_x,posicao_y), 2)
                pygame.draw.line(surface, PRETO, (posicao_x,origem_y), (posicao_x,origem_y+quadrado_y), 2)
            else:
                pygame.draw.line(surface, CINZA, (origem_x+2,posicao_y), (origem_x+quadrado_x,posicao_y), 2)
                pygame.draw.line(surface, CINZA, (posicao_x,origem_y+2), (posicao_x,origem_y+quadrado_y), 2)
            posicao_y += inc_y
            posicao_x += inc_x

        posicao_antiga = (origem_x,origem_y+quadrado_y)
        pygame.draw.circle(surface, VERMELHO, posicao_antiga, 5)

        divisao_grafico = int(quadrado_x/len_lista_pontos)
        ponto_x = origem_x

        for indice in range(len_lista_pontos-1):
            ponto_x += divisao_grafico
            ponto_y = origem_y + quadrado_y - int(self.lista_pontos[indice] / escala_y)
            posicao_atual = (ponto_x,ponto_y)
            pygame.draw.line(surface, AZUL, posicao_antiga, posicao_atual)
            pygame.draw.circle(surface, AZUL, posicao_atual, 3)
            posicao_antiga = (ponto_x,ponto_y)

        ponto_x = origem_x + quadrado_x
        ponto_y = origem_y + quadrado_y - int(self.lista_pontos[-1] / escala_y)
        posicao_atual = (ponto_x,ponto_y)
        pygame.draw.circle(surface, AZUL, posicao_atual, 3)
        pygame.draw.line(surface, AZUL, posicao_antiga, posicao_atual)

        posicao_x = origem_x + 700

        indice_dino = lista_dinos.index(lista_dinos_vivos[-1])
        texto_descricao = exibe_mensagem(f'individuo: {indice_dino + 1}', 20, VERMELHO)
        surface.blit(texto_descricao, (ponto_x + 50, origem_y))

        texto_descricao = exibe_mensagem('cor:', 20, PRETO)
        surface.blit(texto_descricao, (ponto_x + 300, origem_y))

        pygame.draw.rect(surface, lista_dinos[indice_dino].cor_dino, (ponto_x + 350, origem_y, 17, 17))

        origem_y = 60
        posicoes_xy = []
        for indice_camada, camada in enumerate(self.lista_neuronios):
            if indice_camada == 0:
                posicao_y = origem_y
            else:
                posicao_y = origem_y + ((self.lista_neuronios[0] - camada) * 25)

            posicao_camada = []
            for neuronio in range(camada):
                posicao_atual = (posicao_x,posicao_y)
                posicao_camada.append(posicao_atual)
                if indice_camada == 0:
                    texto_descricao = exibe_mensagem(self.descricao[neuronio], 15, PRETO)
                    surface.blit(texto_descricao, (origem_x+420, posicao_y-6))
                    texto_entrada = exibe_mensagem(entradas[neuronio], 15, PRETO)
                    surface.blit(texto_entrada, (origem_x+620, posicao_y-6))

                    red = entradas[neuronio] / 100
                else:
                    red = saidas[indice_camada-1][neuronio] / 100

                    for posicao_antiga in posicoes_xy[indice_camada-1]:
                        if saidas[indice_camada-1][neuronio]:
                            pygame.draw.line(surface, VERMELHO, posicao_antiga, posicao_atual, 1)
                        else:
                            pygame.draw.line(surface, CINZA, posicao_antiga, posicao_atual, 1)

                if red > 1:
                    red = 255
                elif red < 0:
                    red = 0
                else:
                    red *= 255

                pygame.draw.circle(surface, (red,0,0), posicao_atual, 10)

                posicao_y += 50
            posicao_x += 100
            posicoes_xy.append(posicao_camada)

class Dino(pygame.sprite.Sprite):
    """Representa o personagem dinossauro no jogo, com diferentes estados de animação (correndo, agachando, pulando) 
    e lógica de movimentação (pulo, colisão com o chão e gravidade)."""
    def __init__(self, y_inicial:int):
        """Inicializa o Dino com a posição inicial no eixo y, define variáveis relacionadas à movimentação e
        cria a lista de sprites com a cor aleatória."""
        pygame.sprite.Sprite.__init__(self)
        self.individuo = None
        self.morreu = False
        self.passando_obstaculo = False
        self.y_inicial = y_inicial
        self.velocidade_y = 0
        self.index_sprite = 0
        self.sprite_list = []
        self.cor_dino = None

        self.set_cor()

        self.image = self.sprite_list[1]
        self.rect = pygame.Rect(50, 0, 35, 43)
        self.rect.bottom = self.y_inicial

    def set_cor(self):
        """Define uma cor aleatória para o Dino e adiciona as sprites a lista de sprites com base na
        imagem de fundo (sheet_dino), aplicando a cor escolhida."""
        self.cor_dino = (randint(0,200), randint(0,200), randint(0,200))
        sheet = sheet_dino.copy()
        sheet.fill(self.cor_dino, special_flags=pygame.BLEND_RGB_MULT)

        self.sprite_list = []
        for i in range(6):
            img = sheet.subsurface((i * 64,0), (64,64))
            self.sprite_list.append(img)
        
    def run(self):
        """Define a animação de corrida do Dino, alternando entre os sprites de corrida com base
        em um índice de sprite ajustado ao longo do tempo."""
        if self.index_sprite < 2 or self.index_sprite > 3.65:
            self.index_sprite = 2
        # 6 frames com index 2 e 6 frames com index 3
        self.index_sprite += 0.15
        self.image = self.sprite_list[int(self.index_sprite)]
        self.rect.bottom = self.y_inicial

    def crouch(self):
        """Define a animação de agachamento do Dino, ajustando sua altura e alternando entre os sprites de agachamento."""
        if self.index_sprite < 4 or self.index_sprite > 5.65:
            self.index_sprite = 4
        # 6 frames com index 4 e 6 frames com index 5
        self.index_sprite += 0.15
        self.rect.height = 26
        self.image = self.sprite_list[int(self.index_sprite)]
        self.rect.bottom = self.y_inicial

    def jump(self):
        """Simula o pulo do Dino, alterando sua velocidade vertical."""
        self.velocidade_y -= 0.5

    def update(self):
        """Atualiza a posição do Dino na tela, considerando se ele está no chão ou no ar, e aplicando a
        gravidade para controlar a movimentação de pulo e queda."""
        if self.morreu:
            if self.rect.right+50 > 0:
                self.rect.x -= cenario_velocidade
        else:
            if self.rect.bottom == self.y_inicial:
                self.velocidade_y = 0
            else:
                self.velocidade_y += 1
                if self.rect.bottom + self.velocidade_y > self.y_inicial:
                    self.rect.bottom = self.y_inicial
                else:
                    self.rect.y += self.velocidade_y

class Chao(pygame.sprite.Sprite):
    """Representa o chão do jogo, que se move horizontalmente e é reposicionado quando sai da tela."""
    def __init__(self, y_inicial:int):
        """Inicializa o Chão com uma lista de imagens de sprite e define sua posição inicial,
        com base na coordenada y fornecida como parametro."""
        pygame.sprite.Sprite.__init__(self)
        self.sprite_list = []
        for i in range(6):
            img = sheet_chao.subsurface((i * 60,0), (60,12))
            self.sprite_list.append(img)

        self.image = self.sprite_list[randint(0,5)]
        self.rect = self.image.get_rect()
        self.rect.y = y_inicial

    def update(self):
        """Atualiza a posição do Chão, movendo-o horizontalmente e reposicionando-o ao final da tela
        com uma nova imagem quando sai da tela à esquerda."""
        self.rect.x -= cenario_velocidade
        if self.rect.right <= 0:
            self.rect.x += 60 * len_lista_chao
            self.image = self.sprite_list[randint(0,5)]

class Nuvem(pygame.sprite.Sprite):
    """Representa a Nuvem no jogo, que se move horizontalmente na tela e reaparece 
    aleatoriamente ao sair da tela."""
    def __init__(self):
        """Inicializa a Nuvem com a imagem e define sua posição inicial fora da tela, à direita."""
        pygame.sprite.Sprite.__init__(self)
        self.image = sprite_nuvem
        self.rect = self.image.get_rect()
        self.rect.right = 0

    def update(self):
        """Atualiza a posição da Nuvem, movendo-a horizontalmente e reposicionando-a aleatoriamente 
        quando sai da tela à esquerda."""
        self.rect.x -= cenario_velocidade-4
        if self.rect.right <= 0:
            self.rect.x = LARGURA_TELA + randrange(0, LARGURA_TELA-50, 50)
            self.rect.y = randrange(50, 500, 20)

class Cacto(pygame.sprite.Sprite):
    """Representa o obstáculo Cacto no jogo, que possui diferentes variações de sprites e tamanho.
    O Cacto se move horizontalmente na tela."""
    def __init__(self, y_inicial:int):
        pygame.sprite.Sprite.__init__(self)
        """Inicializa o Cacto com a posição y inicial, carrega as imagens de sprite e define a imagem inicial do Cacto."""
        self.y_inicial = y_inicial
        self.sprite_list = []
        for i in range(5):
            img = sheet_cacto.subsurface((i * 73,0), (73,47))
            self.sprite_list.append(img)

        self.rect = pygame.rect.Rect(0,0,0,0)
        self.image = None
        self.set_image()

    def set_image(self) -> pygame.surface.Surface:
        """Escolhe aleatoriamente uma imagem de sprite do Cacto e ajusta o tamanho e a posição do retângulo de colisão."""
        indice_img = randint(0,4)
        match indice_img:
            case 0:
                self.rect.size = (15,33)
            case 1:
                self.rect.size = (32,33)
            case 2:
                self.rect.size = (49,33)
            case 3:
                self.rect.size = (22,47)
            case 4:
                self.rect.size = (73,47)
        self.rect.bottom = self.y_inicial

        self.image = self.sprite_list[indice_img]

    def update(self):
        """Atualiza a posição do Cacto na tela, movendo-o horizontalmente com base na velocidade do cenário."""
        if self.rect.right > 0:
            self.rect.x -= cenario_velocidade

class Pterossauro(pygame.sprite.Sprite):
    """Representa o obstáculo Pterossauro no jogo, que se move horizontalmente na tela e alterna entre 
    duas imagens de sprite para animação."""
    def __init__(self, y_inicial:int):
        """Inicializa o Pterossauro com a posição y inicial, define os sprites e a posição inicial na tela."""
        pygame.sprite.Sprite.__init__(self)
        self.y_inicial = y_inicial
        self.index_sprite = 0
        self.sprite_list = [
            sheet_pterossauro.subsurface((0,0), (42,36)),
            sheet_pterossauro.subsurface((42,0), (42,36))
        ]
        self.image = self.sprite_list[0]
        self.rect = self.image.get_rect()
        self.rect.right = 0

    def update(self):
        """Atualiza a posição do Pterossauro na tela e altera a animação entre os sprites de voo."""
        if self.rect.right > 0:
            self.rect.x -= cenario_velocidade

            if self.index_sprite > 1.65:
                self.index_sprite = 0
            # 6 frames com index 2 e 6 frames com index 3
            self.index_sprite += 0.15
            self.image = self.sprite_list[int(self.index_sprite)]
        
def nome_da_classe(objeto):
    """Retorna o nome da classe do objeto passado como argumento."""
    return objeto.__class__.__name__

def set_posicao_x(obstaculo):
    """Define a posição horizontal do obstáculo com base na posição do último obstáculo visível na tela, 
    adicionando uma distância aleatória entre 400 e 600 pixels."""
    obstaculo.rect.x = lista_obstaculos_tela[-1].rect.x + randint(400,600)

def set_novo_obstaculo():
    """Define e posiciona um novo obstáculo na tela, escolhendo aleatoriamente entre um Pterossauro ou um Cacto,
    ajustando sua posição vertical e adicionando-o à lista de obstáculos visíveis na tela."""
    nome_classes_espera = list(map(nome_da_classe, lista_obstaculos_espera))
    chance_pterossauro = randint(1,5) == 1 # 20%
    
    if chance_pterossauro:
        nome_obstaculo = "Pterossauro"
    else:
        nome_obstaculo = "Cacto"

    if nome_obstaculo in nome_classes_espera:
        if nome_classes_espera[0] == nome_obstaculo:
            obstaculo_espera = lista_obstaculos_espera.pop(0)
        else:
            obstaculo_espera = lista_obstaculos_espera.pop(1)
        lista_obstaculos_espera.append(lista_obstaculos_tela.pop(0))
    else:
        obstaculo_espera = lista_obstaculos_tela.pop(0)

    if nome_da_classe(obstaculo_espera) == "Pterossauro":
        obstaculo_espera.rect.bottom = randrange(obstaculo_espera.y_inicial-60, obstaculo_espera.y_inicial+30, 30)
    else:
        obstaculo_espera.set_image()

    set_posicao_x(obstaculo_espera)
    lista_obstaculos_tela.append(obstaculo_espera)

def mata_dino(dino:Dino):
    """Marca o dinossauro como morto, altera sua imagem e executa o som de morte, 
    ajustando a altura e posição do dinossauro caso ele tenha um tamanho ou posição específica."""
    if som_ativo:
        som_morte.play()

    dino.morreu = True
    dino.image = dino.sprite_list[0]

    if dino.rect.height == 26 or dino.rect.bottom > dino.y_inicial:
        dino.rect.height = 43
        dino.rect.bottom = dino.y_inicial

def exibe_mensagem(msg, tamanho:int, cor:tuple):
    """Exibe uma mensagem formatada na tela com a fonte e cor especificadas"""
    fonte = pygame.font.Font(diretorio_fonte, tamanho)
    texto_formatado = fonte.render(f"{msg}", True, cor)
    return texto_formatado

def carregar_individuo():
    """Carrega os dados do melhor indivíduo salvo no arquivo "save.json"."""
    try:
        with open("save.json", "r") as f:
            dados = json.load(f)
            for indice in range(len(dados["pesos"])):
                dados["pesos"][indice] = np.array(dados["pesos"][indice])
                dados["bias"][indice] = np.array(dados["bias"][indice])
            return dados
    except FileNotFoundError:
        return None

def salvar_individuo(individuo:Individuo, geracao:int):
    """Salva os dados do melhor indivíduo se ele tiver um fitness superior ao atualmente armazenado."""
    melhor_individuo = carregar_individuo()

    if melhor_individuo is None or individuo.fitness > melhor_individuo["fitness"]:
        dados = {
            "geracao": geracao,
            "pesos": individuo.pesos,
            "bias": individuo.bias,
            "fitness": individuo.fitness
        }

        with open("save.json", "w") as f:
            json.dump(dados, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

def resource_path(*paths) -> str:
    """Retorna o caminho correto, dependendo de estar rodando no executável ou no código fonte."""
    if getattr(sys, "frozen", False):
        # Quando estiver no executável (PyInstaller)
        base_path = sys._MEIPASS
    else:
        # Quando estiver no código fonte
        base_path = os.path.join(os.path.dirname(__file__), "assets")

    return os.path.join(base_path, *paths)


if __name__ == "__main__":

    """Configura a rede neural"""
    rede_neural = RedeNeural(
        camada_entrada=6,
        camadas_escondida=[6],
        camada_saida=2,
        descricao=[
            "obstaculo_distacia:",
            "obstaculo_largura:",
            "obstaculo_altura:",
            "obstaculo_comprimento:",
            "cenario_velocidade:",
            "dino_altura:",
        ]
    )

    geracao = 0
    escala = 10
    limite_grafico_y = escala * 300

    """Configura o pygame"""
    pygame.init()

    try:
        pygame.mixer.init()
        som_ativo = True
    except:
        som_ativo = False

    LARGURA_TELA = 1000
    ALTURA_TELA = 600

    PRETO = (0,0,0)
    BRANCO = (255,255,255)
    VERMELHO = (255,0,0)
    CINZA = (220,220,220)
    AZUL = (0,0,255)

    tela = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
    pygame.display.set_caption("Dino I.A.")
    pygame.display.set_icon(pygame.image.load(resource_path("icon.png")))

    relogio = pygame.time.Clock()
    cenario_velocidade = 5

    """Criando um evento personalizado e o dispara a cada 1000 milissegundos (1 segundo)"""
    TIMER_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(TIMER_EVENT, 1000)
    segundos = 0
    minutos = 0

    """Carrega a fonte as imagens e os sons do jogo"""
    diretorio_fonte = resource_path("fonts", "Minecraft.ttf")

    sheet_dino = pygame.image.load(resource_path("images", "dino.png")).convert_alpha()
    sheet_chao = pygame.image.load(resource_path("images", "chao.png")).convert_alpha()
    sprite_nuvem = pygame.image.load(resource_path("images", "nuvem.png")).convert_alpha()
    sheet_cacto = pygame.image.load(resource_path("images", "cacto.png")).convert_alpha()
    sheet_pterossauro = pygame.image.load(resource_path("images", "pterossauro.png")).convert_alpha()

    if som_ativo:
        som_pulo = pygame.mixer.Sound(resource_path("sounds", "jump_sound.wav"))
        som_morte = pygame.mixer.Sound(resource_path("sounds", "death_sound.wav"))
        som_ponto = pygame.mixer.Sound(resource_path("sounds", "score_sound.wav"))

    """Crias todas as sprites do jogo"""
    group_sprites = pygame.sprite.Group()

    melhor_dino = Dino(ALTURA_TELA-15)
    dados_individuo = carregar_individuo()
    
    if dados_individuo:
        melhor_dino.individuo = Individuo(dados_individuo["pesos"], dados_individuo["bias"])
    else:
        melhor_dino.individuo = rede_neural.individuo_random()

    lista_dinos = [melhor_dino]
    group_sprites.add(melhor_dino)

    for _ in range(499):
        dino = Dino(ALTURA_TELA-15)
        dino.individuo = rede_neural.individuo_random()
        lista_dinos.append(dino)
        group_sprites.add(dino)

    lista_dinos_vivos = lista_dinos.copy()
    len_lista_dinos = len(lista_dinos)
    vivos = len_lista_dinos

    lista_chao = []

    for i in range(18):
        chao = Chao(ALTURA_TELA-20)
        chao.rect.x = 60 * i
        chao.image = chao.sprite_list[randint(0,3)]
        lista_chao.append(chao)
        group_sprites.add(chao)

    len_lista_chao = len(lista_chao)

    lista_nuvem = []

    for _ in range(10):
        nuvem = Nuvem()
        lista_nuvem.append(nuvem)
        group_sprites.add(nuvem)

    group_obstaculos = pygame.sprite.Group()

    cacto = Cacto(ALTURA_TELA-10)
    cacto.rect.x = LARGURA_TELA
    group_obstaculos.add(cacto)
    lista_obstaculos_tela = [cacto]

    cacto.rect.size = (73,47)
    cacto.rect.bottom = cacto.y_inicial
    cacto.image = cacto.sprite_list[4]

    for _ in range(3):
        cacto = Cacto(ALTURA_TELA-10)
        set_posicao_x(cacto)
        lista_obstaculos_tela.append(cacto)
        group_obstaculos.add(cacto)

    lista_obstaculos_espera = []

    for _ in range(2):
        pterossauro = Pterossauro(ALTURA_TELA-15)
        lista_obstaculos_espera.append(pterossauro)
        group_obstaculos.add(pterossauro)

    """Loop principal do jogo"""
    while True:
        tela.fill(BRANCO)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                salvar_individuo(lista_dinos_vivos[0].individuo, geracao)
                pygame.quit()
                sys.exit()
            elif event.type == TIMER_EVENT:
                segundos += 1
                if segundos == 60:
                    segundos = 0
                    minutos += 1
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    vivos = 0
                    melhor_dino = lista_dinos_vivos[0]

        indice = 0

        """Esse while percorre todos os dinos vivos"""
        while indice < vivos:
            dino = lista_dinos_vivos[indice]

            """Referencia o obstáculo mais próximo do dino"""
            if lista_obstaculos_tela[0].rect.right > dino.rect.x:   
                obstaculo_frente = lista_obstaculos_tela[0]
            else:
                obstaculo_frente = lista_obstaculos_tela[1]

            entradas = [
                obstaculo_frente.rect.x - dino.rect.right,     # obstaculo_distacia
                obstaculo_frente.rect.right - dino.rect.right, # obstaculo_largura
                ALTURA_TELA - obstaculo_frente.rect.y,         # obstaculo_altura
                ALTURA_TELA - obstaculo_frente.rect.bottom,    # obstaculo_comprimento
                cenario_velocidade,                            # cenario_velocidade
                ALTURA_TELA - dino.rect.y,                     # dino_altura
            ]

            """Calcula a saída da rede neural para o dino"""
            saida = rede_neural.forward(entradas, dino.individuo)

            """Adiciona um ponto ao fitness do dino se ele passar por baixo do pterossauro"""
            if obstaculo_frente.rect.x <= dino.rect.right:
                if ALTURA_TELA - dino.rect.y < ALTURA_TELA - obstaculo_frente.rect.bottom and dino.passando_obstaculo == False:
                    dino.individuo.fitness += 1

                dino.passando_obstaculo = True
            else:
                dino.passando_obstaculo = False

            """Executa a ação com base na saída da rede neural"""
            if saida[-1][0] < saida[-1][1]: # Agachar
                if dino.rect.bottom == dino.y_inicial:
                    dino.crouch()
                else:
                    dino.velocidade_y += 1
            elif saida[-1][0] > saida[-1][1]: # Pular
                dino.rect.height = 43
                dino.image = dino.sprite_list[1]
                if dino.rect.bottom == dino.y_inicial:
                    dino.velocidade_y = -10
                    dino.rect.y -= 10
                    if som_ativo:
                        som_pulo.play()
                else:
                    dino.jump()
            else: # Correr
                if dino.rect.bottom == dino.y_inicial: 
                    dino.rect.height = 43
                    dino.run()

            """Verifica se o dino colidiu com algum obstáculo"""
            colidiu = pygame.sprite.spritecollide(dino, group_obstaculos, False)
            
            if colidiu:
                mata_dino(dino)
                melhor_dino = lista_dinos_vivos.pop(indice)
                vivos -= 1
            else:
                indice += 1

        rede_neural.lista_pontos[-1] += 1

        if rede_neural.lista_pontos[-1] >= limite_grafico_y:
            escala += 10
            limite_grafico_y = escala * 300

        """Taxa de aumento de velocidade do cenario"""
        if rede_neural.lista_pontos[-1] % 250 == 0:
            if som_ativo:
                som_ponto.play()
            if cenario_velocidade < 15:
                cenario_velocidade += 1

        if lista_obstaculos_tela[0].rect.right <= 0:
            set_novo_obstaculo()

        """Renicia o jogo"""
        if vivos == 0:
            
            """config da rede neural"""
            for dino in lista_dinos:
                if dino.individuo.fitness > melhor_dino.individuo.fitness:
                    melhor_dino = dino

            salvar_individuo(melhor_dino.individuo, geracao)
            melhor_dino.individuo.fitness = 0
            
            for dino in lista_dinos:
                dino.set_cor()
                dino.morreu = False
                dino.rect.x = 50
                if dino != melhor_dino:
                    dino.individuo = rede_neural.mutacao(melhor_dino.individuo, 0.2)

            rede_neural.lista_pontos.append(0)

            geracao += 1

            lista_dinos_vivos = lista_dinos.copy()
            vivos = len_lista_dinos

            """config do jogo"""
            cenario_velocidade = 5

            for _ in range(len(lista_obstaculos_tela)):
                lista_obstaculos_espera.append(lista_obstaculos_tela.pop(0))
            
            for indice_tela in range(4):
                indice_espera = 0
                while nome_da_classe(lista_obstaculos_espera[indice_espera]) != "Cacto":
                    indice_espera += 1

                lista_obstaculos_espera[indice_espera].set_image()
                if indice_tela == 0:
                    lista_obstaculos_espera[indice_espera].rect.x = LARGURA_TELA
                else:
                    set_posicao_x(lista_obstaculos_espera[indice_espera])

                lista_obstaculos_tela.append(lista_obstaculos_espera.pop(indice_espera))

            for pterossauro in lista_obstaculos_espera:
                pterossauro.rect.right = 0

            for indice, chao in enumerate(lista_chao):
                chao.image = chao.sprite_list[randint(0,3)]
                chao.rect.x = 60 * indice

            for nuvem in lista_nuvem:
                nuvem.rect.right = 0

            cacto = lista_obstaculos_tela[0]
            cacto.rect.size = (73,47)
            cacto.rect.bottom = cacto.y_inicial
            cacto.image = cacto.sprite_list[4]

        """Desenha as mensagens na tela"""
        texto_pontos = exibe_mensagem(f"pontos: {rede_neural.lista_pontos[-1]}", 30, AZUL)
        tela.blit(texto_pontos, (130,320))

        texto_tempo = exibe_mensagem(f'tempo: {minutos}:{segundos}', 20, PRETO)
        tela.blit(texto_tempo, (450,350))

        texto_geracao = exibe_mensagem(f"geracao: {geracao}", 20, PRETO)
        tela.blit(texto_geracao, (650,350))
        
        texto_vivos = exibe_mensagem(f"vivos: {vivos}", 20, PRETO)
        tela.blit(texto_vivos, (850,350))

        texto_fps = exibe_mensagem(f"Fps: {relogio.get_fps():.2f}", 15, PRETO)
        tela.blit(texto_fps, (920,20))

        """Atualiza e as sprites na tela"""
        group_sprites.update()
        group_obstaculos.update()

        """Desenha as sprites na tela"""
        group_sprites.draw(tela)
        group_obstaculos.draw(tela)
        rede_neural.draw(tela, entradas, saida, (10,10), escala)

        """Atualiza a tela e o relógio do jogo"""
        pygame.display.flip()
        relogio.tick(60)
