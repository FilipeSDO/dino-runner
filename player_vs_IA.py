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
    
    dados_individuo = carregar_individuo()
    
    if dados_individuo:
        dino_ia.individuo = Individuo(dados_individuo["pesos"], dados_individuo["bias"])
    else:
        dino_ia.individuo = Individuo(
            pesos=[
                np.array([
                    [0.8914157803126906, -0.8372861885659463, 0.002370710344286102, -0.5247555778637559, 0.6197739561417591, 0.8103433063209424],
                    [-0.09302993529974754, -1.7165233765706402, -0.16321148230592541, 0.775417728220612, -0.773347697187472, 0.4616250420005482],
                    [0.3046987000636443, 1.0131424557147488, 1.5292279854939053, -1.5333913674211817, -1.0170446051027942, -0.8584164788833523],
                    [1.7347305883511983, -1.3498604702446155, -0.062019864181227596, 0.20697375631097395, -0.05680626295857924, 0.3645002779661349],
                    [-1.014067648018407, -0.8502935179982883, 0.9867444624350931, 0.30099178229584483, 0.2639810988085307, -1.1656285034284857],
                    [-0.6467407579732929, 1.0590785492332255, -0.4677278688444082, 1.9167003092198094, -1.1363605330496769, -0.10579264946496478]
                ]),
                np.array([
                    [-1.824211509535813, 0.5883695548403097],
                    [0.34480149300350926, 0.2138354807627937],
                    [1.0770411117876877, 1.5215396586089982],
                    [0.21830919094370566, -1.5032079109864864],
                    [-2.5292353181239675, 0.8195705712551891],
                    [-0.3377294301142835, -0.8926588687678352]
                ])
            ],
            bias=[
                np.array([-0.815920734964958, 1.8829256074977516, 0.041424555823322634, 0.8175603399086147, 0.9200972465661936, 0.9161014905060323]),
                np.array([0.3154739804682615, 0.7084499550076195])
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
