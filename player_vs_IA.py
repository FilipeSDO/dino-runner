import pygame, sys, os, json, numpy as np
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

class Dino(pygame.sprite.Sprite):
    """Representa o personagem dinossauro no jogo, com diferentes estados de animação (correndo, agachando, pulando) 
    e lógica de movimentação (pulo, colisão com o chão e gravidade)."""
    def __init__(self, y_inicial:int, cor_dino:tuple):
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
        self.set_cor(cor_dino)

        self.image = self.sprite_list[1]
        self.rect = pygame.Rect(50, 0, 35, 43)
        self.rect.bottom = self.y_inicial

    def set_cor(self, cor:tuple):
        """Define uma cor aleatória para o Dino e adiciona as sprites a lista de sprites com base na
        imagem de fundo (sheet_dino), aplicando a cor escolhida."""
        sheet = sheet_dino.copy()
        sheet.fill(cor, special_flags=pygame.BLEND_RGB_MULT)

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
        
def nome_da_classe(objeto) -> str:
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

def exibe_mensagem(msg, tamanho:int, cor:tuple) -> pygame.surface.Surface:
    """Exibe uma mensagem formatada na tela com a fonte e cor especificadas"""
    fonte = pygame.font.Font(diretorio_fonte, tamanho)
    texto_formatado = fonte.render(f"{msg}", True, cor)
    return texto_formatado

def carrega_json() -> dict:
    """Carrega os dados da rede neural salvo no arquivo "save.json"."""
    try:
        with open("save.json", "r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
            for indice in range(len(dados["individuo"]["pesos"])):
                dados["individuo"]["pesos"][indice] = np.array(dados["individuo"]["pesos"][indice])
                dados["individuo"]["bias"][indice] = np.array(dados["individuo"]["bias"][indice])
            return dados
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def resource_path(*paths) -> str:
    """Retorna o caminho correto, dependendo de estar rodando no executável ou no código fonte."""
    if getattr(sys, "frozen", False):
        # Quando estiver no executável (PyInstaller)
        base_path = sys._MEIPASS
    else:
        # Quando estiver no código fonte
        base_path = os.path.join(os.path.dirname(__file__), "assets")

    return os.path.join(base_path, *paths)

def alterna_cor(dino:Dino):
    """Altera a cor do Dino para uma nova cor aleatória."""
    for indice in range(3):
        if dino_cor[indice] <= 5:
            dino_inc[indice] = randint(0,2)
        if dino_cor[indice] >= 250:
            dino_inc[indice] = -randint(0,2)

        dino_cor[indice] += dino_inc[indice]
            
    dino.set_cor(tuple(dino_cor))

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
    CINZA = (200,200,200)
    AZUL = (0,0,255)

    tela = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
    pygame.display.set_caption("Dino I.A.")
    pygame.display.set_icon(pygame.image.load(resource_path("icon.png")))

    relogio = pygame.time.Clock()
    cenario_velocidade = 5
    renicia = False
    start = False
    pontos = 0

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

    """Configura o dinossauro da rede neural"""

    dino_ia = Dino(ALTURA_TELA-15, CINZA)
    
    dados = carrega_json()
    
    if dados:
        # tem erro aki
        dino_ia.individuo = Individuo(dados["individuo"]["pesos"], dados["individuo"]["bias"])
    else:
        dino_ia.individuo = Individuo(
            pesos = [
                np.array([
                    [
                        7.196304852649284,
                        -0.8739284556547353,
                        -1.6975048223995874,
                        -2.98242856498219,
                        -2.7021646940160915,
                        1.2781313844327562
                    ],
                    [
                        4.742541953883711,
                        -11.869064400397106,
                        -7.790752627277446,
                        -5.344319388794939,
                        6.899810621104953,
                        -1.1759477190178842
                    ],
                    [
                        1.1224235946486858,
                        -0.40061956985182307,
                        -4.26790057337712,
                        -12.442097770867159,
                        0.9291181289042172,
                        7.280128774885846
                    ],
                    [
                        -12.66623272904336,
                        1.7052058907530259,
                        -1.0943501925153436,
                        6.509224593982106,
                        0.22764843152623804,
                        -13.632405470325589
                    ],
                    [
                        -5.162484955395137,
                        -8.58024069920135,
                        10.624031043448081,
                        -12.83127759695358,
                        -8.497248449476011,
                        5.405933049404479
                    ],
                    [
                        -12.842842267697566,
                        -0.6266692725507854,
                        2.267377048400517,
                        11.992436112326144,
                        6.7753798039642765,
                        1.1226732468251372
                    ]
                ]),
                np.array([
                    [
                        -13.228347415617975,
                        -4.873485555819144
                    ],
                    [
                        -2.4595792782331607,
                        22.226765890496594
                    ],
                    [
                        22.958096390683938,
                        -0.7936401059029947
                    ],
                    [
                        4.816804502679519,
                        -1.9298940656587684
                    ],
                    [
                        0.887515880142751,
                        4.1905217063949145
                    ],
                    [
                        6.855975892895226,
                        -13.933524981600545
                    ]
                ])
            ],
            bias = [
                np.array([
                    -2.04535008540506,
                    5.471609138429885,
                    0.7913569598490003,
                    11.443279839241377,
                    8.238800402782305,
                    -6.3876627774243495
                ]),
                np.array([
                    -0.1773017800941552,
                    -10.152596785011804
                ])
            ]
        )
        
    """Configura o dinossauro do jogador"""

    dino_player = Dino(ALTURA_TELA-15, AZUL)
    dino_player.rect.x = 150

    dino_cor = [randint(0,200),randint(0,200),randint(0,200)]
    dino_inc = [randint(0,2), -randint(0,2), randint(0,2)]

    """Crias todas as sprites do jogo"""
    group_sprites = pygame.sprite.Group()

    group_sprites.add(dino_ia)
    group_sprites.add(dino_player)

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
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    renicia = True
                elif event.key in [pygame.K_UP, pygame.K_SPACE]:
                    start = True
                    if dino_player.rect.bottom == dino_player.y_inicial:
                        dino_player.velocidade_y = -10
                        dino_player.rect.y -= 10
                        if som_ativo:
                            som_pulo.play()

        if start:
            if not dino_ia.morreu:
                """Referencia o obstáculo mais próximo do dino"""
                if lista_obstaculos_tela[0].rect.right > dino_ia.rect.x:   
                    obstaculo_frente = lista_obstaculos_tela[0]
                else:
                    obstaculo_frente = lista_obstaculos_tela[1]

                entradas = [
                    obstaculo_frente.rect.x - dino_ia.rect.right,     # obstaculo_distacia
                    obstaculo_frente.rect.right - dino_ia.rect.right, # obstaculo_largura
                    ALTURA_TELA - obstaculo_frente.rect.y,            # obstaculo_altura
                    ALTURA_TELA - obstaculo_frente.rect.bottom,       # obstaculo_comprimento
                    cenario_velocidade,                               # cenario_velocidade
                    ALTURA_TELA - dino_ia.rect.y,                     # dino_altura
                ]

                """Calcula a saída da rede neural para o dino"""
                saida = rede_neural.forward(entradas, dino_ia.individuo)

                """Executa a ação com base na saída da rede neural"""
                if saida[-1][0] < saida[-1][1]: # Agachar
                    if dino_ia.rect.bottom == dino_ia.y_inicial:
                        dino_ia.crouch()
                    else:
                        dino_ia.velocidade_y += 1
                elif saida[-1][0] > saida[-1][1]: # Pular
                    dino_ia.rect.height = 43
                    dino_ia.image = dino_ia.sprite_list[1]
                    if dino_ia.rect.bottom == dino_ia.y_inicial:
                        dino_ia.velocidade_y = -10
                        dino_ia.rect.y -= 10
                        if som_ativo:
                            som_pulo.play()
                    else:
                        dino_ia.jump()
                else: # Correr
                    if dino_ia.rect.bottom == dino_ia.y_inicial: 
                        dino_ia.rect.height = 43
                        dino_ia.run()

                """Verifica se o dino colidiu com algum obstáculo"""
                colidiu = pygame.sprite.spritecollide(dino_ia, group_obstaculos, False)
                
                if colidiu:
                    ultimo_dino = mata_dino(dino_ia)

            if not dino_player.morreu:
                if pygame.key.get_pressed()[pygame.K_DOWN]:
                    if dino_player.rect.bottom == dino_player.y_inicial:
                        dino_player.crouch()
                    else:
                        dino_player.velocidade_y += 1
                else:
                    dino_player.rect.height = 43

                    if dino_player.rect.bottom < dino_player.y_inicial:
                        dino_player.image = dino_player.sprite_list[1]
                        if pygame.key.get_pressed()[pygame.K_UP] or pygame.key.get_pressed()[pygame.K_SPACE]:
                            dino_player.jump()
                    else:
                        dino_player.run()

                colisoes = pygame.sprite.spritecollide(dino_player, group_obstaculos, False)

                if colisoes:
                    mata_dino(dino_player)

            if not dino_player.morreu:
                alterna_cor(dino_player)

            pontos += 1

            """Taxa de aumento de velocidade do cenario"""
            if pontos % 250 == 0:
                if som_ativo:
                    som_ponto.play()
                if cenario_velocidade < 15:
                    cenario_velocidade += 1

            if lista_obstaculos_tela[0].rect.right <= 0:
                set_novo_obstaculo()

            """Atualiza e as sprites na tela"""
            group_sprites.update()
            group_obstaculos.update()
        
        if dino_ia.morreu and dino_player.morreu:
            renicia = True

        """Renicia o jogo"""
        if renicia:
            
            """config do player e da IA"""
            dino_player.morreu = False
            dino_player.rect.x = 150
            dino_player.rect.height = 43
            dino_player.image = dino_player.sprite_list[1]
            dino_player.rect.bottom = dino_player.y_inicial

            dino_ia.morreu = False
            dino_ia.rect.x = 50
            dino_ia.rect.height = 43
            dino_ia.image = dino_ia.sprite_list[1]
            dino_ia.rect.bottom = dino_ia.y_inicial

            """config do jogo"""
            cenario_velocidade = 5
            renicia = False
            start = False
            pontos = 0

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

        """Desenha as mensagens na tela"""
        texto_fps = exibe_mensagem(f"Fps: {relogio.get_fps():.2f}", 30, PRETO)
        tela.blit(texto_fps, (50,20))

        texto_pontos = exibe_mensagem(f"velocidade: {cenario_velocidade}", 30, PRETO)
        tela.blit(texto_pontos, (350,20))

        texto_pontos = exibe_mensagem(f"pontos: {pontos}", 30, AZUL)
        tela.blit(texto_pontos, (750,20))

        """Desenha as sprites na tela"""
        group_sprites.draw(tela)
        group_obstaculos.draw(tela)

        """Atualiza a tela e o relógio do jogo"""
        pygame.display.flip()
        relogio.tick(60)
