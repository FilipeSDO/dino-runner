# 🦖 Dino IA - Jogo com Rede Neural  

Dino IA é um jogo baseado no clássico Dino Runner do Chrome, mas com um diferencial: uma Inteligência Artificial (IA) treinada para jogar sozinha!  

![Preview do Jogo](assets/preview.gif)

## 🤖 Sobre a IA  

A IA foi treinada usando uma rede neural para reconhecer padrões e aprender a jogar com base na pontuação e na posição dos obstáculos.

## 🚀 Funcionalidades  
- 🎮 Modo Jogador vs. IA (`player_vs_IA.py`)  
- 🤖 Modo IA Aprendendo a Jogar (`dino_IA.py`)  
- 🧠 Rede Neural para aprendizado  
- 🎨 Gráficos e sons personalizados  

## 📂 Estrutura do Repositório  

```
- assets/            # Recursos visuais e sonoros
  ├── fonts/         # Fontes usadas no jogo
  ├── images/        # Imagens do jogo
  ├── sounds/        # Efeitos sonoros e música
  ├── icon.png       # Ícone do jogo
  └── preview.gif    # GIF de demonstração do jogo

- executaveis.rar    # Arquivos pré-compilados do jogo
- dino_IA.py         # Código da IA jogando sozinha
- dino_IA.spec       # Configuração do PyInstaller para gerar executável
- player_vs_IA.py    # Código para modo jogador vs IA
- player_vs_IA.spec  # Configuração do PyInstaller para modo jogador vs IA
- requirements.txt   # Dependências do projeto
- README.md          # Documentação do projeto
```

## 🛠️ Como Rodar  

### 🔧 Pré-requisitos  
Antes de rodar o jogo, instale as dependências:  

```bash
pip install -r requirements.txt
```

### 🎢 Jogando  

Para jogar contra a IA:  

```bash
python player_vs_IA.py
```

Para ver a IA treinando:  

```bash
python dino_IA.py
```

## 📦 Como Gerar o Executável  

Se quiser criar um executável, use:  

```bash
pyinstaller dino_IA.spec
pyinstaller player_vs_IA.spec
```

Os arquivos `.spec` já estão configurados para facilitar o processo.  

## 🐝 Licença  

Este projeto é open-source e está sob a licença [MIT](LICENSE).  

## ✨ Créditos  

Desenvolvido por Filipe Silva com Python, Pygame, Numpy e técnicas de IA.  

---
Se tiver sugestões, abra uma issue ou faça um fork e contribua! 🚀  

