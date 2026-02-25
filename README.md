# OAK-D SR - Sistema de Medição por Profundidade

> Sistema de visão computacional com câmera OAK-D Short Range para medição de espessura em tempo real, desenvolvido pela **Embeddo**.

---

## Sobre o Projeto

Este projeto utiliza a câmera **Luxonis OAK-D SR** (Short Range) com a biblioteca **DepthAI** para realizar medições de profundidade e espessura em aplicações industriais, como monitoramento de espuma e inspeção de cola em vidro.  
O sistema captura dados de disparidade estéreo, processa as informações de profundidade e exibe resultados em tempo real via OpenCV, permitindo medições precisas na faixa de curto alcance (20 cm a 1 m).

---

## Especificações da Câmera OAK-D SR

| Parâmetro                 | Especificação                |
|---------------------------|------------------------------|
| Sensor                    | OV9782 (par estéreo + cor)   |
| Resolução                 | 1 MP (1280 x 800)            |
| FPS Máximo                | 120 fps (800p)               |
| Campo de Visão (DFOV)     | 89.5°                        |
| HFOV / VFOV               | 80° / 55°                    |
| Foco                      | Fixo: 20 cm - ∞              |
| F-number                  | 2.0 ± 5%                     |
| Focal Efetiva             | 2.35 mm                      |
| Tamanho do Pixel          | 3 µm x 3 µm                  |
| Obturador                 | Global Shutter               |
| Processamento             | 4 TOPS (1.4 TOPS para IA)    |
| Codificação               | H.264, H.265, MJPEG          |
| IMU                       | BMI270 (6 eixos)             |
| Conexão                   | USB-C (USB 2.0 / USB 3.0)    |
| Consumo                   | ~5 W                         |
| Dimensões                 | 56 x 36 x 25.5 mm            |
| Peso                      | 72 g                         |

---

## Requisitos

### Hardware

- Câmera Luxonis OAK-D SR  
- Cabo USB-C 3.0 (recomendado para estabilidade)  
- PC com porta USB 3.0

### Software

- Python 3.10+  
- Ambiente virtual (venv)

### Dependências

```bash
depthai>=3.2.1
opencv-python
numpy
pandas

Instalação
1. Clonar o
bash
git clone https://github.com/Cordeir0t/OAK.git
cd OAK

2. Criar ambiente virtual
Windows:

bash
python -m venv .venv
.\.venv\Scripts\activate
Linux/Ubuntu:

bash
python3 -m venv .venv
source .venv/bin/activate

3. instalar posses
bash
pip install depthai opencv-python numpy pandas

Uso
Executar o sistema de medição
bash
python oak.py

controles
Tecla	Ação
SPACE	Definir ROI (região de interesse)
S	Salvar medição / captura de tela
R	Redefinir
T	Profundidade Alternativa / Disparidade
Q	Sair
Estrutura do Projeto
bash
oak-d-sr-medicao/
├── oak.py                  # Script principal de medição
├── medicoes/               # Logs e CSVs de medições (gerado automaticamente)
├── jlr_espuma_logs/        # Logs específicos de monitoramento de espuma
├── requirements.txt        # Dependências Python
└── README.md               # Este arquivo
Profundidade do PipelineAI
O sistema utiliza o pipeline do DepthAI v3 com os seguintes nós:

Câmera (MonoCamera) – Captura estéreo (CAM_B / CAM_C) a 400p

StereoDepth – Processamento de profundidade estéreo

Predefinição: SHORT_RANGE(otimizado para 20 cm – 1 m)

Filtro mediano:KERNEL_7x7

Limiar de confiança: 180

Verificação LR

XLinkOut – Transmissão de frames (disparidade + profundidade) para o host

texto
[MonoCamera LEFT] ──┐
                     ├── [StereoDepth] ──── [XLinkOut: disparity]
[MonoCamera RIGHT] ─┘                  └── [XLinkOut: depth]
Dicas de Uso
Iluminação: Use iluminação superior e direta sobre o objeto.

Distância: Mantenha a câmera entre 20 cm e 50 cm do objeto para melhor precisão.

Textura: Superfícies sem textura (vidro, espuma lisa) podem gerar poucos pixels válidos — aplique pó fino ou spray fosco se necessário.

Cabo USB: Utilize sempre cabo USB 3.0 para evitar frames perdidos.

Posicionamento: Câmera perpendicular ao objeto, sem vibração.

x de Problemas
Problema	…
Imagem toda preta/azul	Aproximar objeto (20–50 cm), verificar iluminação
Poucos pixels válidos	Adicionar textura na superfície, ajustar confiança
AttributeError: 'CostMatching'	Atualizar DepthAI ou remover APIs obsoletas
CAM_A not found	Usar CAM_B/ CAM_C(ou LEFT/ RIGHT)
Queda de frames / atraso	Trocar cabo para USB 3.0, reduzir resolução
ModuleNotFoundError: depthai	Ativar venv e rodarpip install depthai
s
Documentação Luxonis OAK-D SR

SDK Python DepthAI

Notas de lançamento DepthAI v3

Exemplos DepthAI no GitHub


