# PA2 — Identidade ao longo do tempo

Implementação própria de associação, gestão de tracks, recorrência, NMS e métricas.
Trilha A (movimento), ablação pelo eixo 1 (célula e janela BPTT) e estresse por
qualidade do detector. Não usa trackers prontos nem bibliotecas de métricas MOT.

## Instalação

Na raiz do repositório, com Python 3.10 ou superior:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[full,test]"
python -m pytest -q
```

O pacote importável se chama `mot_pa2`; seus arquivos ficam em `src/mot_pa2/`.
A instalação descobre automaticamente seus subpacotes. Após atualizar uma
instalação anterior, execute novamente `python -m pip install -e ".[full,test]"`.
A execução verificada nesta máquina usa CPU. Os parâmetros ficam em Python,
sem uma coleção de argumentos e subparsers.

## Organização do código

```text
src/mot_pa2/
├── core/           # Geometria e estruturas Detection/Track
├── data/           # MOT17, dados sintéticos, downloads e splits
├── models/         # Modelo recorrente e adaptador do detector
├── tracking/       # Associação e ciclo de vida dos tracks
├── training/       # Preparação de trajetórias e treino
├── evaluation/     # Métricas, avaliação, agregação e escrita de JSON
├── visualization/  # Desenho de quadros e anotações por identidade
├── experiments/    # Ablação, estresse e comparação de detectores
├── pipelines/      # Inferência e reprodução dos artefatos
└── cli.py          # Execução do experimento sintético
```

Os módulos usam imports explícitos a partir de `mot_pa2`. As estruturas
compartilhadas ficam em `core`; os experimentos e pipelines coordenam os demais
componentes. A visualização não depende da inferência, e a escrita de resultados
não depende dos trackers. Os testes ficam em `tests/`.

Exemplos de imports:

```python
from mot_pa2.data.synthetic import SyntheticConfig, generate_video
from mot_pa2.evaluation.metrics import evaluate_tracking
from mot_pa2.pipelines.inference import infer_video
from mot_pa2.tracking.trackers import TemporalTracker
```

Os antigos imports diretos, como `mot_pa2.temporal`, passam a usar os subpacotes
correspondentes (`mot_pa2.models.temporal`). `train.py` e `metrics.py` na raiz
continuam disponíveis como pontos de entrada de compatibilidade; as implementações
ficam dentro do pacote.

## Dados e separação

Use o [arquivo oficial MOT17](https://motchallenge.net/data/MOT17/).
Extraia MOT17Labels.zip em `data_MOT17Labels/`, contendo
`train/MOT17-02-FRCNN/{gt,det,seqinfo.ini}` e as demais sequências.
As imagens da sequência 09 podem ser obtidas sem baixar todos os 5,5 GB:

```powershell
python -m mot_pa2.data.download
```

O downloader acessa somente as faixas do ZIP oficial que contêm imagens da
sequência 09. Requer um servidor com suporte a HTTP Range.

| Uso | Sequências inteiras |
|---|---|
| Treino | 02, 04 |
| Validação e ablação | 05 |
| Teste reservado | 09, 10, 11, 13 |

Nunca dividimos quadros da mesma sequência entre conjuntos. 02/04 oferecem
cenas de praça e rua noturna com ponto de vista elevado; 05 valida generalização
para câmera móvel e outra resolução; o teste cobre densidades, iluminação e
movimento variados. As variantes DPM/FRCNN/SDP da mesma cena **não** são
sequências independentes para fins de split.

FRCNN público é a fonte fixa do modelo temporal: dispensa treino de detector e
permite comparar com Faster R-CNN COCO do torchvision na sequência 09.
Essa escolha foi fixada antes de avaliar os testes, não escolhida pelo melhor
resultado. Não há ajuste fino de nenhum detector.

## Treinar e avaliar

Um comando para treinar GRU em trajetórias verdadeiras, com checkpoint escolhido
pela perda de validação:

```powershell
python train.py
```

Um comando para avaliar o checkpoint existente, sem treino ou substituição de pesos:

```powershell
python -m mot_pa2.experiments.analysis
```

Esse comando grava em `outputs/evaluation.json` o caminho e o SHA-256 do
checkpoint avaliado. Se os checkpoints da ablação estiverem disponíveis,
também compara suas curvas de gradiente; caso contrário, mede a curva do modelo
final. Para gerar a ablação completa, execute explicitamente:

```powershell
python -m mot_pa2.experiments.ablation
```

Os splits e a localização dos dados estão em `src/mot_pa2/data/splits.py`;
o protocolo compartilhado fica em `src/mot_pa2/training/config.py`.
O treino principal e cada execução da ablação usam três épocas.
A configuração principal é GRU, T=16, seed=0,
fixada previamente. A ablação usa três seeds por célula e T em {4,8,16,32},
com aproximadamente 14 mil parâmetros por célula. As tabelas registram a
contagem exata. BPTT carrega o estado entre blocos e corta seu grafo a cada T;
todos os modelos recebem as mesmas trajetórias e épocas. O treino usa teacher
forcing e clipping de norma 1. O estresse usa o checkpoint congelado.

Resultados de ablação existentes são retomados quando o checkpoint existe e
os metadados de célula, janela, seed, splits e número de épocas conferem.
Isso não detecta toda mudança possível no algoritmo. Para mudar o protocolo,
use uma nova pasta de resultados/checkpoints ou remova conscientemente os
resultados antigos antes da execução.

A segunda fonte de detecções roda sem retreinar:

```powershell
python -m mot_pa2.experiments.detector_comparison
```

Esse comando processa **todos** os quadros da sequência 09, mantém cache
retomável e reporta mAP e métricas de tracking para as duas fontes. O download
inicial dos pesos oficiais do torchvision requer internet. O NMS interno do
detector é substituído pela implementação deste projeto.
Para viabilizar a avaliação em CPU, Faster R-CNN MobileNet V3 COCO usa lado menor 320 e lado
maior no máximo 640, com limiar de score 0,5. Essa resolução faz parte da
configuração fixa dessa fonte e é registrada no resultado.

Para reproduzir tudo, incluindo download das imagens, treino principal, ablação,
avaliação, vídeo e cópia dos artefatos da entrega (as anotações devem estar extraídas):

```powershell
python -m mot_pa2.pipelines.reproduce
```

Esse é o comando que retreina o modelo principal. Para apenas copiar resultados
já gerados e atualizar o manifesto, sem treinar:

```powershell
python -c "from mot_pa2.pipelines.reproduce import publish; publish()"
```

## Regras de associação e métricas

A baseline usa matching guloso por IoU, limiar 0,3. O Hungarian opcional usa
somente o solucionador de atribuição do scipy; construção da matriz, limiar
e gestão de tracks são próprios. Cada observação não associada cria um ID.
Tracks sobrevivem a três quadros sem observação e morrem no quarto. Enquanto
vivas, suas caixas são emitidas e avaliadas, inclusive sob oclusão: isso pode
aumentar falsos positivos. O relógio inclui quadros sem detecções.

O modelo temporal recebe centro/largura/altura normalizados pela imagem,
mantém estado oculto por track e prevê um incremento na caixa para o próximo
quadro. A associação usa a caixa prevista. Sem observação, a previsão realimenta
a recorrência. A perda é smooth-L1 com beta=0,01. O estado morre junto da track.

IDF1 usa IoU >= 0,5 e atribuição **global ótima** entre identidades na sequência
inteira. Não lê `gt_id` nas predições. IDSW compara IDs espacialmente associados
a cada identidade verdadeira; a última associação persiste através de lacunas.
Fragmentação é retomada após quadro anotado não associado, depois de ao menos
uma associação anterior. Ausência do próprio GT não inicia uma fragmentação.
Erro de contagem é IDs previstos menos IDs verdadeiros (com sinal).

mAP usa uma classe, 101 pontos de recall e média nos limiares
IoU 0,50:0,05:0,95. Reportamos explicitamente duas agregações:

- `mAP_frame_mean`: média aritmética do AP calculado em cada quadro, com peso
  igual por quadro. `mAP_per_frame` preserva os valores individuais. Quadros sem
  GT recebem AP=0, inclusive quando não há detecções, e entram na média.
- `mAP_sequence`: AP com detecções agrupadas e ordenadas por score na sequência
  inteira. A chave legada `mAP` conserva esse significado para compatibilidade.

Os gráficos e tabelas mostram ambas; elas não são intercambiáveis.
Não é a implementação completa
do protocolo COCO. Avaliamos pedestres com conf>0 e classe=1 do MOT17;
não aplicamos todas as regras de regiões ignoradas do servidor MOTChallenge.
Os números são do protocolo explícito deste PA, não uma submissão ao ranking.

## Artefatos das partes 0–5

| Requisito | Arquivo gerado em outputs/ |
|---|---|
| Elipse totalmente oculta por N quadros e reaparecimento | synthetic/occlusion.png e occlusion.json |
| Piso fácil e quebra da baseline | synthetic/results.json e degradation.png |
| Gráfico de descolamento em dois painéis | decoupling.png |
| Baseline versus temporal nas mesmas sequências | comparison.json |
| Duas fontes de detecção na mesma sequência | detector_comparison.json |
| 3 células × 4 janelas × 3 seeds, média ± desvio amostral | ablation.json e ablation.png |
| Norma dL/dh por atraso, RNN/LSTM/GRU | memory/gradient.png e results.json |
| Sobrevivência sob oclusão versus durações do dataset | memory/empirical.png |
| Três falhas com GT, previsão e caixa intermediária | failures/case_1.png até case_3.png e diagnoses.json |
| Correção de vida máxima de 3 para 16, antes/depois | correction.json e correction.png |
| Mesma falha da galeria antes/depois, IDs, IoU e ausências | correction_case.json e correction_case.png |
| Identificação do checkpoint efetivamente avaliado | evaluation.json |
| Três intensidades de degradação, sem retreino, mAP e IDF1 | stress.json e stress.png |
| Vídeo e contagem sem ground truth nem treino | inference.mp4 e inference.json |

O horizonte analítico usa gradiente da perda final em relação ao estado oculto
de cada passo, sobre trajetórias reais. Para LSTM mede h, não a derivada total
em relação ao par (h,c). O horizonte empírico usa oclusões injetadas em trajetórias
reais isoladas: mede a combinação de previsão e regra de sobrevivência, sem
confundir esse resultado com reidentificação em multidões. A comparação do
dataset considera intervalos internos com visibility<0,2 ou anotação ausente,
delimitados por observações visíveis; a definição fica registrada no JSON.

A correção é uma hipótese predefinida de prolongar a vida das tracks e é medida
na validação. Ganho negativo também é resultado: indica que prolongar a vida
sozinho não resolve o erro de movimento/associação. Não se altera a configuração
principal após olhar os testes.

Além da avaliação agregada na sequência 05, `correction_case.*` mostra o mesmo
objeto e os mesmos quadros do primeiro caso da galeria na sequência 09, com vida
máxima 3 e 16. É uma ilustração pós-hoc da hipótese já fixada, não uma nova seleção
de modelo no teste. O JSON acompanha o ID original, o ID associado ao GT, a
sobrevivência, as ausências e a IoU quadro a quadro. Melhorar esse caso isolado
não significa melhorar o resultado agregado.

## Inferência e apresentação

Abra `inferencia.ipynb` no ambiente instalado e edite `sequence_root`.
O notebook não usa GT, carrega `checkpoints/gru.pt`, exporta vídeo com cores
determinísticas por ID e mostra a contagem de objetos únicos. Aceita:

- Pasta MOT com `seqinfo.ini` e `img1/`: `source="auto"` usa `det/det.txt` quando
  disponível e infere DPM/FRCNN/SDP pelo nome da pasta. `detector` permite override
  explícito e rejeita uma variante incompatível.
- Pasta de imagens JPG/JPEG/PNG/BMP: informe `fps`, por exemplo 30. Os arquivos
  são ordenados naturalmente (2 antes de 10); todos devem ter a mesma resolução.
- Arquivo de vídeo: lê FPS e resolução do arquivo e processa quadro a quadro.

Sem detecções públicas, `auto` usa torchvision. Também é possível selecionar
`source="public"` ou `source="torchvision"` explicitamente. A primeira execução
do detector pode baixar seus pesos COCO; isso não é treino.

Em vídeos longos, reiniciar o tracker em cada janela perde os estados, a idade,
as caixas previstas e a numeração de IDs, fragmentando identidades na fronteira.
A trilha A permite costurar janelas mantendo a mesma instância do tracker
ou serializando esses elementos entre blocos. Isso não garante recuperar uma
identidade depois de sua track morrer: o modelo usa geometria, sem memória de
aparência para reidentificação.

O checkpoint principal é pequeno e fica incluído na entrega. `results/`
recebe uma cópia das tabelas e figuras finais e um manifesto com hashes dos
dados, código, testes, documentação, checkpoints e artefatos. O manifesto descreve
o snapshot atual; não atribui retroativamente os treinos antigos ao código novo.
Os vídeos grandes ficam em `outputs/` e são
reproduzíveis pelo notebook. Não há relatório obrigatório: use as figuras e
diagnósticos como suporte à apresentação, sem afirmar melhoria onde os números
não a mostram.
