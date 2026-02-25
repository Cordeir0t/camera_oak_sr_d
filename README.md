# OAK-D SR — Sistema de Inspeção Industrial com Visão Estéreo

Sistema de inspeção industrial desenvolvido com a câmera **OAK-D SR**, utilizando o SDK **DepthAI**, voltado para aplicações de controle de qualidade baseadas em análise de profundidade em tempo real.

O projeto implementa processamento embarcado de visão estéreo para medição dimensional, detecção de irregularidades superficiais e validação automática de critérios técnicos de produção.

Repositório:
[https://github.com/Cordeir0t/camera_oak_sr_d](https://github.com/Cordeir0t/camera_oak_sr_d)

---

## 1. Visão Geral

A solução utiliza visão estéreo ativa para gerar mapas de disparidade e extrair métricas quantitativas de inspeção. O sistema foi projetado com foco em:

* Estabilidade temporal da profundidade
* Robustez contra ruído e falhas de textura
* Processamento em tempo real
* Estrutura modular para expansão industrial

Principais aplicações implementadas:

1. Inspeção de faixas de cola em vidro
2. Inspeção de defeitos superficiais em rodas automotivas

---

## 2. Arquitetura Técnica

### Pipeline de Visão

* MonoCamera esquerda — 400p @ 30 FPS
* MonoCamera direita — 400p @ 30 FPS
* StereoDepth (Preset: HIGH_DENSITY)
* Pós-processamento configurado para ambiente industrial:

  * Left-Right Check
  * Extended Disparity
  * Median Filter 7x7
  * Temporal Filter com persistência VALID_2_IN_LAST_4
  * Spatial Filter com fechamento de lacunas
  * Threshold de faixa de profundidade

Fluxo de processamento:

Mono L + Mono R
→ StereoDepth
→ Disparidade
→ Normalização por percentis
→ Suavização e filtros espaciais
→ Extração de características
→ Métricas quantitativas
→ Interface com HUD técnico

---

## 3. Projeto 1 — Inspeção de Faixas de Cola em Vidro

Arquivo principal:
oakd_sr_inspect_glass_v4.py

### Objetivo

Validar automaticamente a aplicação de faixas de cola em vidro com base em critérios quantitativos:

* Percentual de cobertura
* Número de faixas detectadas
* Espessura real das faixas
* Orientação geométrica
* Score final ponderado

### Metodologia

1. Realce da imagem mono

   * CLAHE (Equalização adaptativa)
   * Detecção de bordas via Canny
   * Filtro passa-alta para reforço estrutural

2. Extração das regiões candidatas

   * Dilatação horizontal
   * Operações morfológicas de limpeza
   * Filtro por área relativa

3. Medição de espessura real

   * Projeção vertical da máscara binária
   * Bounding box da maior faixa
   * Espessura final definida pelo menor valor entre:

     * Altura da bounding box
     * Espessura projetada

### Critérios Técnicos

Cobertura aceitável:
0,5% a 3,0% da área total

Número de faixas:
1 a 6

Espessura válida:
8 a 25 pixels

### Score de Qualidade

Peso dos critérios:

* Cobertura: 50%
* Número de faixas: 30%
* Espessura: 20%

Classificação:

* Score ≥ 90: OK
* 70 ≤ Score < 90: AVISO
* Score < 70: FAIL

O sistema gera feedback visual e métrico em tempo real, permitindo integração futura com CLP ou sistema supervisório.

---

## 4. Projeto 2 — Inspeção de Defeitos em Roda Automotiva

Arquivo principal:
oakd_sr_inspect_rodas.py

### Objetivo

Detectar irregularidades superficiais em rodas automotivas por meio de variação local de profundidade.

### Estratégia de Detecção

1. Normalização por percentis
   Ajusta o contraste apenas aos valores válidos da cena.

2. Suavização preservando bordas

   * Bilateral Filter
   * Fechamento morfológico

3. Extração de gradiente de profundidade

   * Sobel X e Sobel Y
   * Magnitude do gradiente
   * CLAHE aplicado ao gradiente

4. Threshold adaptativo
   Isola regiões com variação abrupta de profundidade, associadas a possíveis defeitos.

5. Modo BLEND
   Sobreposição do mapa de defeitos ao mapa de profundidade para inspeção técnica detalhada.

---

## 5. Cálculo de Distância

A distância é estimada a partir da mediana da disparidade em uma ROI central:

dist_cm ≈ (baseline × focal) / disparidade

Implementação aproximada no código:

dist_cm ≈ (460 × 2) / disparidade

O valor é limitado ao intervalo operacional de 0 a 999 cm.

---

## 6. Tecnologias Utilizadas

* Python 3.11
* OpenCV
* NumPy
* DepthAI SDK
* OAK-D SR

Instalação:

pip install depthai opencv-python numpy

---

## 7. Aplicações Industriais Potenciais

* Controle de qualidade em linha de produção
* Validação de aplicação de adesivos industriais
* Inspeção dimensional baseada em profundidade
* Detecção de irregularidades superficiais
* Monitoramento automatizado de conformidade

---

## 8. Diferenciais Técnicos

* Processamento em tempo real
* Métrica quantitativa automatizada
* Filtros temporais para estabilidade industrial
* Redução de ruído por filtros espaciais e percentis
* Estrutura modular para expansão
* Arquitetura preparada para integração com sistemas industriais

---

## 9. Autoria

Talita Cordeiro


---


