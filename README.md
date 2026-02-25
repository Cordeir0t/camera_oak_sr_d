```md
# OAK-D SR — Sistema Industrial de Medição por Profundidade

Sistema de visão computacional para medição de espessura e análise dimensional em tempo real utilizando a câmera **Luxonis OAK-D Short Range** e a biblioteca **:contentReference[oaicite:1]{index=1}**.

---

## 1. Visão Geral

Este projeto implementa um pipeline estéreo para geração de mapa de profundidade (Depth Map) e cálculo de espessura em superfícies com diferentes propriedades ópticas, incluindo:

- Espumas técnicas  
- Aplicação de cola sobre vidro  
- Materiais translúcidos ou semitransparentes  

A aquisição é realizada por visão estéreo ativa, com processamento embarcado no dispositivo (Myriad X) e visualização em tempo real via **:contentReference[oaicite:2]{index=2}**.

**Faixa operacional recomendada:**  
30 cm — 1,0 m  
**Faixa ideal para medições de alta precisão:**  
30 cm — 50 cm  

---

## 2. Especificações Técnicas — OAK-D SR

| Parâmetro      | Especificação                  |
|---------------|--------------------------------|
| Sensor        | OV9782 (Estéreo + RGB)         |
| Resolução     | 1 MP (1280 × 800)              |
| FPS Máx       | 120 fps (800p)                 |
| DFOV          | 89.5°                          |
| Baseline      | 20 mm                          |
| Foco          | 20 cm — ∞                      |
| Abertura      | F# 2.0 ±5%                     |
| Distância Focal | 2.35 mm                     |
| Pixel Size    | 3 µm                           |
| Shutter       | Global                         |
| Processador   | 4 TOPS (1.4 TOPS AI)           |
| Dimensões     | 56 × 36 × 25.5 mm              |
| Peso          | 72 g                           |

---

## 3. Arquitetura do Sistema

### Pipeline DepthAI

```

LEFT Camera  ──┐
├── StereoDepth ── Disparity / Depth Map ── XLinkOut
RIGHT Camera ──┘

````

### Configuração do Estéreo

- MonoCamera LEFT / RIGHT — 400p  
- StereoDepth:
  - Preset: `SHORT_RANGE`
  - Median Filter: `KERNEL_7x7`
  - Confidence Threshold: 180  
  - Depth alignment configurável  

---

## 4. Requisitos

### Hardware

- OAK-D SR  
- Cabo USB-C 3.0 (recomendado alta qualidade)  
- PC com USB 3.0 nativo  

### Software

- Python 3.11  
- Ambiente virtual (venv)

---

## 5. Dependências

```bash
pip install depthai opencv-python numpy pandas
````

---

## 6. Instalação

```bash
git clone https://github.com/Cordeir0t/camera_oak_sr_d.git
cd camera_oak_sr_d

python -m venv .venv

# Windows
.\.venv\Scripts\activate

pip install -r requirements.txt
```

---

## 7. Execução

```bash
python oak.py
```

ou

```bash
python oakd_sr_inspect_glass.py
```

### Controles

| Tecla | Função                         |
| ----- | ------------------------------ |
| SPACE | Selecionar ROI                 |
| S     | Salvar medição                 |
| R     | Resetar ROI                    |
| T     | Alternar visualização de Depth |
| Q     | Encerrar aplicação             |

---

## 8. Estrutura do Projeto

```
├── oak.py
├── oakd_sr_inspect_glass.py
├── img/
├── medicoes/
├── requirements.txt
└── README.md
```

---

## 9. Diretrizes Operacionais

Para garantir maior estabilidade e repetibilidade metrológica:

* Manter distância constante entre 30–50 cm
* Utilizar iluminação difusa e homogênea
* Aplicar spray fosco em superfícies altamente refletivas
* Evitar luz solar direta
* Utilizar cabo USB 3.0 certificado

---

## 10. Ambiente Validado

| Componente | Versão       | Verificação              |
| ---------- | ------------ | ------------------------ |
| Python     | 3.11.9       | `python --version`       |
| DepthAI    | 2.24.0.0     | `pip show depthai`       |
| OpenCV     | 4.x          | `pip show opencv-python` |
| NumPy      | 1.26.4       | `pip show numpy`         |
| Ambiente   | oak-venv-224 | `pip list`               |

---

## 11. Troubleshooting

| Problema              | Possível Causa          | Solução                |
| --------------------- | ----------------------- | ---------------------- |
| Tela preta            | Objeto fora do range    | Ajustar distância      |
| Poucos pixels válidos | Superfície sem textura  | Aplicar spray fosco    |
| AttributeError        | Versão incompatível SDK | Atualizar DepthAI      |
| Queda de frames       | Gargalo USB             | Verificar cabo USB 3.0 |

Atualização do SDK:

```bash
pip install --upgrade depthai
```

---

## 12. Documentação Oficial

* [https://docs.luxonis.com](https://docs.luxonis.com)
* [https://docs.luxonis.com/hardware/products/OAK-D-SR](https://docs.luxonis.com/hardware/products/OAK-D-SR)

---

## 13. Aplicações

* Inspeção industrial de espessura
* Controle de qualidade em linha
* Medição de aplicação de adesivo
* Análise de superfícies translúcidas

---

## Autor

Talita Cordeiro Teixeira
Projeto desenvolvido para aplicações industriais de visão computacional com foco em inspeção dimensional em tempo real.

```

Se desejar, posso adaptar para um padrão mais corporativo (ex: formato para portfólio técnico, apresentação para banca de TCC ou documentação estilo ISO/industrial).
```
