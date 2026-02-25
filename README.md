````md
# OAK-D SR - Sistema de Medição por Profundidade

> Sistema de visão computacional utilizando câmera Luxonis OAK-D Short Range para medição de espessura em tempo real.

## Sobre o Projeto

Este projeto utiliza a câmera OAK-D SR com a biblioteca DepthAI para realizar medições de profundidade em superfícies como:

- Espuma  
- Cola em vidro  
- Materiais translúcidos  

A captura é realizada via visão estéreo, com processamento de mapa de profundidade e visualização em tempo real com OpenCV.

Range ideal de operação: 30 cm — 1 m.

---

## Especificações OAK-D SR

| Parâmetro | Especificação |
|---|---|
| Sensor      | OV9782(estéreo + cor)|
| Resolução   | 1 MP (1280x800)      |
| FPS Máx     | 120 fps (800p)       |
| DFOV        | 89.5°                |
| Baseline    | 20 mm                |
| Foco        | 20 cm — ∞            |
| Abertura    | F# 2.0 ±5%           |
| Focal       | 2.35 mm              |
| Pixel Size  | 3 µm                 |
| Shutter     | Global               |
| Processador | 4 TOPS (1.4 AI)      |
| Dimensões   | 56 × 36 × 25.5 mm    |
| Peso        | 72 g                 |



## Requisitos

### Hardware
- OAK-D SR  
- Cabo USB-C 3.0  
- PC com USB 3.0  

### Software
- Python 3.10+  
- venv  



## Dependências

```bash
pip install depthai opencv-python numpy pandas
````



## Instalação

```bash
git clone https://github.com/Cordeir0t/camera_oak_sr_d.git
cd camera_oak_sr_d

python -m venv .venv

# Windows
.\.venv\Scripts\activate

pip install -r requirements.txt
```

---

## Execução

```bash
python oak.py
```

ou

```bash
python oakd_sr_inspect_glass.py
```

Controles:

* SPACE = Selecionar ROI
* S = Salvar medição
* R = Reset
* T = Toggle Depth View
* Q = Sair

---

## Estrutura do Projeto

```
├── oak.py
├── oakd_sr_inspect_glass.py
├── img/
├── medicoes/
├── requirements.txt
└── README.md
```

---

## Pipeline DepthAI

```
LEFT Camera  ──┐
               ├── StereoDepth ── Disparity / Depth Map ── XLinkOut
RIGHT Camera ──┘
```

Configuração Stereo:

* MonoCamera LEFT / RIGHT (400p)
* StereoDepth:

  * Modo: SHORT_RANGE
  * Median Filter: KERNEL_7x7
  * Confidence Threshold: 180

---

## Dicas de Uso

Distância ideal: 30 — 50 cm
Superfícies lisas devem receber spray fosco para melhorar textura
Utilizar cabo USB 3.0 para evitar perda de frames

### Requerimentos 

| Componente | Versão Atual       | Comando para Verificar |
| ---------- | ------------------ | ---------------------- |
| Python     | 3.11.9             | python --version       |
| DepthAI    | 2.24.0.0           | pip show depthai       |
| OpenCV     | 4.x (cv2)          | pip show opencv-python |
| NumPy      |  1.26.4            | pip show numpy         |
| Ambiente   | oak-venv-224       | pip list               |
---

## Troubleshooting

| Problema               | Solução                                |
| ---------------------- | -------------------------------------- |
| Tela preta             | Aproximar objeto e melhorar iluminação |
| Poucos pixels de depth | Aumentar textura da superfície         |
| AttributeError         | Atualizar DepthAI                      |
| Frames dropados        | Verificar cabo USB 3.0                 |

```bash
pip install --upgrade depthai
```

---

## Links Úteis

[https://docs.luxonis.com](https://docs.luxonis.com)
[https://docs.luxonis.com/hardware/products/OAK-D-SR](https://docs.luxonis.com/hardware/products/OAK-D-SR)

## Autor

Projeto desenvolvido para aplicações industriais de visão computacional.

```
```


