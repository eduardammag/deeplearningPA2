# Auditoria do PA2

Data: 29/09/2026. Referência: as cinco páginas de `PA2.pdf` presentes neste repositório.

## Atualização após as correções solicitadas

Os cinco pontos da auditoria foram tratados. A matriz e os achados abaixo desta
atualização preservam o registro histórico do estado anterior.

| Ponto | Correção | Evidência atual |
|---|---|---|
| Treino e avaliação | Protocolo compartilhado de três épocas; avaliação sem treino nem cópia de pesos; comando separado de ablação. | `training/config.py`, `experiments/ablation.py`, `results/evaluation.json`; teste que proíbe treino durante avaliação. |
| Correção da galeria | Antes/depois do mesmo objeto, sequência e quadros do caso 1, com rastreamento do ID original, ausências e IoU. | `results/correction_case.json` e `.png`; vida máxima 3 perde o ID 7 e vida máxima 16 o mantém no evento 173. Análise pós-hoc explicitada, sem seleção de modelo no teste. |
| Entrada do notebook | Infere DPM/FRCNN/SDP; suporta estrutura MOT, pastas JPG/JPEG/PNG/BMP e arquivos de vídeo; expõe FPS e fonte. | `data/video.py`, `pipelines/inference.py`, notebook e testes de DPM, pasta genérica e vídeo. |
| mAP | Reporta média por quadro e agregação por sequência separadamente, com AP individual e convenção para quadros sem GT. | `mAP_frame_mean`, `mAP_sequence`, `mAP_per_frame` nos JSONs; gráficos e tabelas com ambas. |
| Rastreabilidade | Manifesto recursivo do snapshot atual e registro de IA atualizado. | `evaluation/provenance.py`, `results/manifest.json`, `AI_LOG.md`. |

Verificação: 25 testes passaram; a avaliação das quatro sequências, a correção,
o estresse e os gráficos foram reexecutados. O SHA-256 do checkpoint principal
foi preservado. A comparação de detectores reutilizou as detecções existentes;
não houve retreino da ablação. O caso corrigido ilustra um ganho local, enquanto
a correção continua piorando o IDF1 agregado da validação. Isso permanece um
resultado negativo válido e documentado.

Limite de ambiente: o nbclient não conseguiu iniciar o kernel devido às permissões
Windows do arquivo de conexão. As células do notebook são verificadas em Python;
essa execução não certifica o frontend Jupyter. Tampouco certifica a apresentação
oral ou o entendimento da dupla.

## Registro histórico anterior às correções

**Conclusão da auditoria inicial: as seis partes possuem implementação e artefatos, mas não é possível certificar conformidade literal e completa.** Havia limitações na interface de inferência, na ligação entre diagnóstico e correção, e na reprodução do modelo pelo fluxo documentado. A expressão “mAP por quadro” precisava ser esclarecida. Os resultados negativos do modelo, por si só, não significam descumprimento do enunciado.

Esta auditoria verifica o estado local, incluindo a refatoração ainda não commitada. Não altera implementação, pesos ou resultados oficiais. Os ensaios adicionais estão em `.cache/`.

## Matriz de requisitos

“Atende” indica evidência no código e/ou nos artefatos inspecionados, não certificação de uma apresentação ainda não realizada.

| Referência no PDF | Requisito | Constatação | Situação |
|---|---|---|---|
| Dados, p. 1 | MOT17 com identidades, caixas e visibilidade | Leitor próprio em `src/mot_pa2/data/mot17.py`; conserva `visibility`. | Atende |
| Dados, p. 1 | Split por sequência; teste nunca visto; justificativa | Treino 02/04, validação 05, teste 09/10/11/13. Teste automatizado impede variantes da mesma cena em splits diferentes. README justifica a escolha. | Atende; explicação oral não verificável |
| Engenharia, p. 2 | Sem tracker ou associação MOT prontos | Associação, nascimento, atualização e morte estão implementados no projeto. SciPy é usado como solucionador de atribuição, biblioteca expressamente permitida. | Atende no código inspecionado |
| Engenharia, p. 2 | IDF1 e ID switches próprios; sem bibliotecas de métricas MOT | Implementação em `evaluation/metrics.py`; atribuição global para IDF1 e eventos para switches/fragmentações. | Atende |
| Engenharia, p. 2 | NMS próprio, sem `torchvision.ops.nms` | `core/geometry.py` implementa NMS; adaptador substitui a função usada pelo detector. Inferência de um quadro funcionou bloqueando a verificação de operadores do NMS original. | Atende no caminho executado |
| Engenharia, p. 2 | Modelo temporal recorrente, não Kalman | GRU/LSTM/RNN com estado por track; não há Kalman no modelo final. | Atende |
| Engenharia, p. 2 | Autoria e citação de ideias externas | Não identifiquei implementação pronta de MOT incorporada nos módulos lidos. A origem histórica das ideias e o entendimento da dupla não podem ser comprovados por inspeção. | Não certificável integralmente |
| Parte 0.1, p. 2 | Vídeos 128×128, 30–60 quadros, 5–15 elipses, tamanhos, ruído e contraste | Padrão 128×128 e 45 quadros; experimentos com 5, 8, 12 e 15 objetos; parâmetros expostos. | Atende |
| Parte 0.1, p. 2 | Profundidade real e desaparecimento por N quadros | Desenho em ordem de profundidade e mapa de rótulos por pixel; figura, JSON e teste de oclusão de oito quadros. | Atende |
| Parte 0.2, p. 2 | Descarte, ruído nas caixas e falsos positivos | `corrupt_detections` implementa os três tipos de corrupção. | Atende |
| Parte 0.3, p. 2 | Três casos manuais de métricas | Testes de identidade perfeita, troca de duas identidades e divisão de uma track; verificam valores exatos. | Atende; testes passaram |
| Parte 0.4, p. 2 | Baseline quase perfeita e gráfico de quebra | Cenário fácil com IDF1=1; cenário mais difícil com IDF1≈0,2888; figura em `results/synthetic/degradation.png`. | Atende |
| Parte 1.1, p. 2 | Detecções públicas e detector torchvision COCO person | FRCNN público e Faster R-CNN MobileNet V3 COCO; comparação na sequência 09; cache com todos os 525 quadros. | Atende |
| Parte 1.1, p. 2 | Escolher e justificar fonte padrão | FRCNN público fixado e justificado no README; detector sem fine-tuning. | Atende |
| Parte 1.2–4, p. 3 | Associação ingênua, limiar, nascimento/morte, métricas e documentação | IoU 0,3, matching guloso, IDs novos e morte na quarta ausência; IDF1 global, switches, fragmentações e erro de contagem. | Atende |
| Parte 1.5, p. 3 | Gráfico de descolamento, dois painéis e sequências ordenadas | `results/decoupling.png` tem os dois painéis e ordenação por densidade. O mAP é agregado na sequência. | Atende à estrutura; ressalva sobre “por quadro” |
| Parte 2, p. 3 | Escolher uma trilha e congelar a fonte | Trilha A com detecções públicas FRCNN fixas. | Atende |
| Parte 2, p. 3 | Estado por track, previsão da próxima caixa, associação e rollout | `TemporalTracker` mantém estados e caixas previstas; realimenta previsão quando falta observação. | Atende |
| Parte 2, p. 3 | Perda sobre trajetórias GT | Smooth-L1 sobre representação normalizada da caixa, com trajetórias de treino separadas. | Atende |
| Parte 2, p. 3 | Comparação com baseline nas mesmas sequências/métricas | `results/comparison.json` contém as quatro sequências e todas as métricas para ambos. | Atende |
| Parte 2, p. 3 | Responder sobre fronteiras entre janelas, sem implementar | README explica a preservação de estado, caixas, idade e contador de IDs. | Texto disponível; apresentação não verificável |
| Parte 3, pp. 3–4 | Um eixo, três seeds, média e desvio | Eixo 1: 3 células × 4 janelas × 3 seeds = 36 execuções, sem combinações faltantes. | Atende |
| Parte 3, p. 3 | Orçamento aproximadamente igual e T=4,8,16,32 | RNN 13.903, LSTM 14.116 e GRU 14.109 parâmetros; BPTT com detach entre blocos. | Atende |
| Parte 3, p. 3 | Interpretar quebra da RNN e gradiente | Notas informam corretamente que não houve quebra clara no IDF1; curvas mostram atenuação. | Atende como resultado negativo documentado |
| Parte 4, p. 4 | Três falhas com GT, predição e representação intermediária | Três figuras com GT, IDs previstos e caixas previstas pela recorrência. | Atende |
| Parte 4, p. 4 | Diagnóstico de cada falha | JSON contém visibilidade, lacuna, janela e gradiente, mas repete um diagnóstico e usa medidas agregadas. | Parcial quanto à força da explicação causal |
| Parte 4.1, p. 4 | Norma de dL/dh por atraso; comparar células na mesma janela | Curvas de 32 atrasos com checkpoints RNN/LSTM/GRU T=16. LSTM mede h, como informado. | Atende |
| Parte 4.2, p. 4 | Horizonte empírico e comparação com oclusões do dataset | 168 ensaios em 24 trajetórias; comparação com 43 intervalos de oclusão da sequência 09. | Atende, com alcance limitado |
| Parte 4, p. 4 | Correção baseada num diagnóstico, antes/depois e interpretação | Vida máxima 3→16 avaliada na sequência 05; falhas ilustradas são da 09. Falta demonstrar o efeito nos casos escolhidos. | Evidência parcial; ver achado 2 |
| Parte 5, p. 4 | Um teste de estresse sem retreino | Três intensidades de descarte, ruído e falsos positivos; mesmo modelo; mAP e IDF1 juntos. | Atende |
| Entregáveis, p. 4 | Repositório, README, metrics.py, AI_LOG, notebook e checkpoint | Todos existem; checkpoint principal consta no Git; `metrics.py` da raiz reexporta a implementação própria do pacote. | Presentes, com ressalvas abaixo |
| Entregáveis, p. 4 | Um comando treina e um avalia | Os comandos existem, mas o fluxo de avaliação também treina/retoma ablação e substitui o checkpoint. | Operacionalmente inconsistente para avaliar o treino anterior |
| Entregáveis, p. 4 | Notebook aceita sequência qualquer e roda sem retreino | Sem GT e sem retreino funciona; entrada exige estrutura MOT e o detector público padrão é FRCNN. | Parcial numa leitura ampla de “qualquer” |
| Entregáveis, p. 4 | Artefatos da apresentação reproduzíveis | Todos os artefatos exigidos pelo manifesto existem; verificações parciais reproduziram métricas salvas. Manifesto não corresponde mais ao código refatorado. | Reprodutibilidade completa não certificada |
| Política de IA, p. 5 | AI_LOG e entendimento da dupla | Registro existente cobre intervenções de 18/19 de setembro; não cobre a refatoração recente. Entendimento exige apresentação. | Registro presente, atualização pendente |

## Achados e ações necessárias

### 1. Fluxo treino → avaliação não preserva necessariamente o modelo treinado

Evidência: `training/trainer.py:39` usa cinco épocas por padrão; `experiments/analysis.py:21` fixa três. `analysis.py:265` executa/retoma a ablação e `analysis.py:269` copia `checkpoints/ablation/gru_T16_s0.pt` sobre `checkpoints/gru.pt`.

Consequência: executar `python train.py` e depois o comando de experimentos não significa avaliar o checkpoint que acabou de ser treinado. Numa instalação limpa, a avaliação também dispara 36 treinos. O PDF não proíbe expressamente um comando combinado, mas isso fragiliza a entrega “um comando que treina, um comando que avalia” e a identificação do modelo final.

Ação: disponibilizar avaliação que apenas carrega o checkpoint escolhido, separar a reprodução completa e alinhar os protocolos de treino. O cache de ablação também deveria validar a existência dos checkpoints: hoje basta existir o JSON para pular o treino (`analysis.py:97`).

### 2. A correção não fecha a investigação dos três casos mostrados

Evidência: `failure_gallery` recebe a sequência 09; `correction_suite` usa `VALIDATION[0]`, sequência 05. Os três diagnósticos repetem a hipótese de duração de track. `best_forecast_iou` é o máximo entre todas as tracks (`analysis.py:251`), não necessariamente a previsão da identidade que se perdeu; o fator de gradiente é a média das mesmas 16 trajetórias para os três casos (`analysis.py:228–230`).

Reexecutei as tracks: os IDs antigos 7, 12 e 22 realmente morrem nos quadros 167, 174 e 206. Portanto, há evidência para a hipótese de morte da track; não seria correto dizer que os diagnósticos são inventados. Porém, a terceira falha tem zero quadros prévios de baixa visibilidade, de modo que “oclusão longa” não explica uniformemente os três casos.

A correção é real e seus resultados foram reproduzidos: IDF1 0,5432447→0,5081733, queda de aproximadamente 3,51 pontos percentuais. IDFP cresce de 1.353 para 2.985; fragmentações caem de 88 para 76. O PDF permite uma correção que não funciona. A lacuna é demonstrar o que ela fez à falha escolhida e explicar com esses dados por que prolongar a vida aumentou previsões falsas.

Ação: escolher um caso e mostrar seus IDs, observações, previsões e morte antes/depois. Para manter a separação experimental, pode-se selecionar uma falha da validação para fundamentar a correção; eventual ilustração no teste deve ser explicitamente pós-hoc, sem nova escolha de configuração.

### 3. “Sequência qualquer” tem restrições não expostas no notebook

Evidência: `pipelines/inference.py:12` assume FRCNN; o notebook não expõe nem passa `detector`. `data/mot17.py:12` exige `seqinfo.ini`; `data/mot17.py:47` procura apenas `img1/*.jpg`.

Teste concreto: usar uma pasta `MOT17-09-DPM` com os defaults produz `ValueError: Detector does not match the sequence directory`. O usuário precisa editar a chamada, não apenas o caminho. Vídeos MP4 ou pastas genéricas de imagens não são aceitos. Se “qualquer sequência” significar exclusivamente sequências MOT compatíveis, a exigência de `seqinfo.ini` é razoável; não classifico a ausência de suporte a MP4, isoladamente, como infração inequívoca.

O aspecto “sem retreinar” funciona: gerei e decodifiquei um vídeo de três quadros a partir de uma pasta de teste sem `gt/`, usando o checkpoint existente.

Ação: expor/inferir o detector, documentar o contrato de entrada e, se a leitura exigida for mais ampla, aceitar metadados explícitos ou vídeo.

### 4. mAP agregado não é média de AP calculado quadro a quadro

Evidência: `evaluation/metrics.py:75` ordena conjuntamente as detecções de todos os quadros; `analysis.py:65` usa esse resultado no gráfico. Na sequência 09, o valor salvo é 0,4753647; a média dos APs calculados separadamente nos 525 quadros é 0,4971653.

A Parte 1.5 diz “mAP por quadro”. Isso pode significar uma métrica de detecção, em contraste com uma métrica de trajetórias, sem exigir a média aritmética por imagem. Por isso este é um ponto de interpretação, não um erro matemático comprovado do mAP implementado.

Ação: para cobrir a leitura literal, produzir também a média por quadro e identificar claramente as duas agregações. Não trocar silenciosamente uma pela outra nas tabelas existentes.

### 5. Rastreabilidade após a refatoração

Auditei os 563 arquivos referenciados por `results/manifest.json`: 19 caminhos de código não existem mais e três hashes mudaram (`metrics.py`, `train.py`, `pyproject.toml`). Os demais arquivos referenciados conferem. Isso é consequência da refatoração anterior, não evidência de números fabricados. O manifesto não é um entregável nominal obrigatório, mas não deve ser apresentado como assinatura do código atual.

`AI_LOG.md` também descreve a organização anterior e não registra a refatoração recente. Há caracteres `?` no lugar de acentos no notebook. Além disso, `src/mot_pa2/` aparece como não rastreado no Git: uma entrega por push/clone precisa incluir os arquivos novos.

Ação: atualizar o registro de IA, corrigir os textos do notebook e gerar um novo manifesto após a validação, preservando a identificação do snapshot histórico se desejado. Incluir a refatoração no commit da entrega.

## Limitações científicas que não são requisitos ausentes

- O modelo final mata a track após quatro ausências. Nos ensaios empíricos, as 24 trajetórias tiveram horizonte máximo de três quadros. Não há reidentificação de uma track morta: não se pode prometer recuperar a mesma identidade após dezenas de quadros de oclusão. O PDF pede investigar essas falhas, não exige uma taxa mínima de sucesso.
- O ganho temporal é pequeno: no IDF1, aproximadamente −0,02 ponto percentual na 09, +0,29 na 10, +0,22 na 11 e +0,34 na 13. Os resultados não sustentam uma alegação de grande melhoria.
- Não há colapso claro da RNN nos resultados da ablação. Isso deve ser explicado como resultado observado, não substituído por uma conclusão esperada dos slides.
- O horizonte empírico usa trajetórias isoladas e é limitado pela política de morte. Não mede sozinho reidentificação em multidões nem a capacidade intrínseca de memória da rede.
- O protocolo de avaliação não reproduz integralmente as regras de regiões ignoradas do MOTChallenge nem todo o COCO. O README informa isso; o PDF não exige equivalência ao servidor oficial.
- Escolher a trilha A, o eixo 1 e o estresse de detector dispensa implementar a trilha B, os demais eixos, queda de FPS, MOTA, fine-tuning e costura efetiva de janelas. Não há relatório escrito obrigatório.

## Verificações realizadas e alcance

- Leitura das cinco páginas do PDF e inspeção dos módulos, testes, notebook, README e resultados.
- `python -m pytest -q`: **16 testes passaram**.
- Verificação das 36 combinações de ablação, das médias/desvios e dos metadados dos 36 checkpoints locais; todas as execuções registram três épocas no JSON. O histórico menor no checkpoint principal representa o momento da melhor validação, não prova treino incompleto.
- Reexecução da baseline e do temporal em todos os 525 quadros da sequência 09: métricas idênticas às salvas.
- Reexecução da correção na sequência 05, com vida máxima 3 e 16: métricas idênticas às salvas.
- Reconstrução das três falhas para verificar a morte dos IDs antigos.
- Inferência sem GT e sem retreino, com vídeo de três quadros decodificado com sucesso.
- Vídeo existente `outputs/inference.mp4` aberto: 525 quadros, 30 FPS. O JSON registra 55 identidades, consistente com o rastreamento reproduzido.
- Detector torchvision executado em um quadro com o caminho do NMS original bloqueado; as cinco detecções coincidiram com o cache.
- Inspeção visual do gráfico de descolamento e da primeira figura da galeria.
- Todos os artefatos declarados obrigatórios no manifesto estão em `results/`; cache do segundo detector cobre quadros 1–525.
- Não retreinei as 36 configurações, não repeti a inferência torchvision nos 525 quadros, não reproduzi do zero as sequências 10/11/13 nem o pipeline completo. Não executei o notebook inteiro em um kernel novo; testei diretamente sua função de inferência. Download em ambiente limpo, instalação em outra máquina, domínio da dupla e apresentação oral não foram certificados.
