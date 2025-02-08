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
        self.sheet = None
        self.sprite_list = []

        self.set_cor()

        self.image = self.sprite_list[1]
        self.rect = pygame.Rect(50, 0, 35, 43)
        self.rect.bottom = self.y_inicial

    def set_cor(self):
        """Define uma cor aleatória para o Dino e adiciona as sprites a lista de sprites com base na
        imagem de fundo (sheet_dino), aplicando a cor escolhida."""
        cor_aleatoria = (randint(0,200), randint(0,200), randint(0,200))
        self.sheet = sheet_dino.copy()
        self.sheet.fill(cor_aleatoria, special_flags=pygame.BLEND_RGB_MULT)

        self.sprite_list = []
        for i in range(6):
            img = self.sheet.subsurface((i * 64,0), (64,64))
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

    cenario_velocidade = 5
    renicia = False
    start = False
    pontos = 0

    tela = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
    pygame.display.set_caption("Dino I.A.")

    relogio = pygame.time.Clock()

    """Carrega as imagens, sons e a fonte do jogo"""
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

    dino_ia = Dino(ALTURA_TELA-15)
    
    dados_individuo = carregar_individuo()
    if dados_individuo:
        pesos = dados_individuo["pesos"]
        bias = dados_individuo["bias"]
        dino_ia.individuo = Individuo(pesos, bias)
    else:
        dino_ia.individuo = Individuo(
            pesos=[[[0.5074315671662251, 0.2633615061417331, 0.5397681365465103, 0.06415556404818265, -0.6534473940271255, -0.26599780732981676], [-0.987905336909293, -0.8763441874234379, 0.0921063500767919, -0.6545921364974719, -0.704535104341474, 0.029202318914437007], [0.9902098157359024, -0.7033346417336367, -0.7064584729202517, -0.7514695676836608, -1.443915005180561, 0.576507756963927], [2.736543265421725, -0.729936757014663, 0.6794409625015634, -0.4125664230637276, 0.8288531942121911, -0.5823492406763886], [-0.3110466142433259, 0.0834032169761393, -0.5767010604645484, 1.6043497057075633, 0.06772790995150739, -0.895447130358725], [-2.159273431340217, -1.1168767684310668, 0.43293582859638907, 0.2983402145000841, -0.798556346900806, 0.7987576215310531], [0.6487397163307907, -1.3332697741434116, 0.7379537924573264, -0.4581321594603124, 1.2876080826821683, -0.033792274321550166]], [[-1.3726606816236013, 0.7269974725689791], [-1.9432376306625583, -0.04488050749893501], [-2.376056207797437, 0.6840570529769193], [1.4492983128457986, -0.28032689461371285], [-0.8233738901342478, -1.492281147813817], [1.165873101025012, -0.978512247365692]]],
            bias=[[0.9216243707333223, -0.3160915201540932, -0.1397999527776848, 0.8378775913877831, 0.45335539691564253, -1.1026840966209435], [0.5647587321068178, -1.0402358174384603]]
        )
        
    """Configura o dinossauro do jogador"""

    dino_player = Dino(ALTURA_TELA-15)
    dino_player.rect.x = 150

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
            if event.type == pygame.KEYDOWN:
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
                    ALTURA_TELA - obstaculo_frente.rect.y,         # obstaculo_altura
                    ALTURA_TELA - obstaculo_frente.rect.bottom,    # obstaculo_comprimento
                    cenario_velocidade,                            # cenario_velocidade
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

            pontos += 1

            """Taxa de aumento de velocidade do cenario"""
            if pontos % 500 == 0:
                if som_ativo:
                    som_ponto.play()
                if cenario_velocidade < 20:
                    cenario_velocidade += 1

            if lista_obstaculos_tela[0].rect.right <= 0:
                set_novo_obstaculo()

            """Atualiza e as sprites na tela"""
            group_sprites.update()
            group_obstaculos.update()
            
        if renicia: # Renicia
            
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
        texto_fps = exibe_mensagem(f"Fps: {relogio.get_fps():.2f}", 20, PRETO)
        tela.blit(texto_fps, (20,20))

        texto_pontos = exibe_mensagem(f"pontos: {pontos}", 50, AZUL)
        tela.blit(texto_pontos, (600,50))

        """Desenha as sprites na tela"""
        group_sprites.draw(tela)
        group_obstaculos.draw(tela)

        """Atualiza a tela e o relógio do jogo"""
        pygame.display.flip()
        relogio.tick(60)
