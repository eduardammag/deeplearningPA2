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

O pacote importável se chama `mot_pa2`; seus arquivos ficam em `src/`.
A instalação faz esse mapeamento. Não é preciso renomear a pasta.
A execução verificada nesta máquina usa CPU. Os parâmetros ficam em Python,
sem uma coleção de argumentos e subparsers.

## Dados e separação

Use o [arquivo oficial MOT17](https://motchallenge.net/data/MOT17/).
Extraia MOT17Labels.zip em `data_MOT17Labels/`, contendo
`train/MOT17-02-FRCNN/{gt,det,seqinfo.ini}` e as demais sequências.
As imagens da sequência 09 podem ser obtidas sem baixar todos os 5,5 GB:

```powershell
python -m mot_pa2.download
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

Um comando para executar os experimentos, incluindo 36 treinos da ablação:

```powershell
python -m mot_pa2.analysis
```

Constantes de treino/split estão em `src/training.py`; o estudo usa três épocas
por execução em `src/analysis.py`. A configuração principal é GRU, T=16, seed=0,
fixada previamente. A ablação usa três seeds por célula e T em {4,8,16,32},
com aproximadamente 14 mil parâmetros por célula. As tabelas registram a
contagem exata. BPTT carrega o estado entre blocos e corta seu grafo a cada T;
todos os modelos recebem as mesmas trajetórias e épocas. O treino usa teacher
forcing e clipping de norma 1. O estresse usa o checkpoint congelado.

Resultados de ablação existentes são retomados. Para mudar o protocolo,
use uma nova pasta de resultados/checkpoints ou remova conscientemente os
resultados antigos antes da execução.

A segunda fonte de detecções roda sem retreinar:

```powershell
python -m mot_pa2.detector_experiment
```

Esse comando processa **todos** os quadros da sequência 09, mantém cache
retomável e reporta mAP e métricas de tracking para as duas fontes. O download
inicial dos pesos oficiais do torchvision requer internet. O NMS interno do
detector é substituído pela implementação deste projeto.
Para viabilizar a avaliação em CPU, Faster R-CNN MobileNet V3 COCO usa lado menor 320 e lado
maior no máximo 640, com limiar de score 0,5. Essa resolução faz parte da
configuração fixa dessa fonte e é registrada no resultado.

Para reproduzir tudo, incluindo download, experimentos, vídeo e cópia dos
artefatos da entrega:

```powershell
python -m mot_pa2.reproduce
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

mAP é AP de uma classe, agrupando detecções dos quadros, com 101 pontos de
recall e média nos limiares IoU 0,50:0,05:0,95. Não é a implementação completa
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

## Inferência e apresentação

Abra `inferencia.ipynb` no ambiente instalado e edite `sequence_root`.
O notebook não usa GT, carrega `checkpoints/gru.pt`, exporta vídeo com cores
determinísticas por ID e mostra a contagem de objetos únicos. Pode usar
detecções públicas ou torchvision. Para sequências públicas com outro detector,
passe também `detector="DPM"` ou `"SDP"` à função.

Em vídeos longos, reiniciar o tracker em cada janela perde os estados, a idade,
as caixas previstas e a numeração de IDs, fragmentando identidades na fronteira.
A trilha A permite costurar janelas mantendo a mesma instância do tracker
ou serializando esses elementos entre blocos. Isso não garante recuperar uma
identidade depois de sua track morrer: o modelo usa geometria, sem memória de
aparência para reidentificação.

O checkpoint principal é pequeno e fica incluído na entrega. `results/`
recebe uma cópia das tabelas e figuras finais e um manifesto com hashes dos
dados, código e checkpoint. Os vídeos grandes ficam em `outputs/` e são
reproduzíveis pelo notebook. Não há relatório obrigatório: use as figuras e
diagnósticos como suporte à apresentação, sem afirmar melhoria onde os números
não a mostram.
