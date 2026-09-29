# AI_LOG

## 2026-09-18 — Auditoria e correção do PA2

Ferramenta: assistente de código Codex.

Solicitações: verificar o repositório contra PA2.pdf; corrigir os requisitos
ausentes e executar as verificações e experimentos necessários.

Decisões e arquivos afetados:

- metrics.py: substituir a dependência de gt_id na predição por IoU, atribuição
  global ótima, eventos explícitos e AP de uma classe. A demonstração inicial
  produzia IDF1=1 para uma caixa errada com gt_id correto; virou teste regressivo.
- pyproject.toml: mapear o pacote mot_pa2 para src e declarar scipy/torchvision.
- src/temporal.py, tracking.py, training.py: estado por track, rollout sem
  observação, treino em trajetórias reais, BPTT truncado, gradientes de estados
  ocultos e checkpoint. A célula PyTorch é permitida; associação e gestão próprias.
- src/synthetic.py: mapa de profundidade por pixel, oclusão completa controlada,
  ruído e contraste. Verificação do número exato de quadros sem a identidade.
- src/analysis.py, reporting.py: 36 execuções de ablação, três seeds, orçamento
  aproximado de parâmetros, comparação, falhas, horizonte, correção e estresse.
- src/detectors.py: detector COCO congelado, substituindo NMS interno por NMS
  próprio. Sem rastreador externo nem biblioteca de métricas.
- src/inference.py, inferencia.ipynb: vídeo sem GT nem retreino, IDs persistentes
  e contagem. README atualizado para execução sem parsers.
- src/download.py: endereço antigo por sequência retornou HTTP 410; usar faixas
  do arquivo oficial MOT17.zip para obter só as imagens necessárias.
- src/reproduce.py: cópia compacta dos resultados e manifesto de hashes.

Verificação: testes automatizados de métricas espaciais, troca e divisão de IDs,
fragmentação, frames vazios, gradiente, treino e sobrevivência; experimentos
reais em CPU e artefatos em results/. O manifesto registra o ambiente e os hashes.
Não são inventados resultados nem garantido ganho do modelo temporal.

A dupla deve revisar as decisões, compreender as perdas, métricas e estado,
e complementar este registro com as próprias intervenções e preparação da apresentação.

## 2026-09-19 — Conclusão dos experimentos

- As 36 combinações de célula, janela e seed foram treinadas sobre trajetórias
  MOT17 reais. O cenário sintético fácil atingiu IDF1=1.
- A configuração temporal principal apresentou ganhos pequenos em três das
  quatro sequências reservadas e uma pequena queda na 09. A extensão da vida
  das tracks de 3 para 16 piorou o IDF1 na validação, apesar de reduzir fragmentações.
- Para a segunda fonte, foi adotado Faster R-CNN MobileNet V3 320 COCO,
  congelado e com NMS próprio. A escolha prioriza execução em CPU; os caches
  dos ensaios preliminares com ResNet não entram nos resultados finais.
- Foi acrescentado teste contra vazamento de uma mesma cena entre splits
  por uso de variantes diferentes de detector. A suíte completa passou 16 testes.
- As notas da apresentação distinguem norma do gradiente, estado da LSTM,
  sobrevivência imposta pela política de tracks e erros reais de associação.

## 2026-09-29 — Refatoração, auditoria e correções solicitadas

Ferramenta: assistente de código Codex. O usuário pediu a organização do código
por responsabilidades, depois a comparação com PA2.pdf e a correção de cinco
ressalvas encontradas. A auditoria e sua atualização estão em PA2_AUDIT.md.

- O código foi organizado em `src/mot_pa2/`, com subpacotes de dados, modelos,
  tracking, treino, avaliação, visualização, experimentos e pipelines. Imports,
  testes, instalação e notebook foram ajustados; a API de métricas na raiz foi
  preservada por reexportação. A configuração dos splits, as anotações de quadros
  e a serialização de resultados foram separadas das rotinas que as utilizam.
- `training/config.py` define três épocas para treino principal e ablação.
  `experiments.analysis` só avalia o checkpoint recebido, registra seu SHA-256
  e verifica que ele não mudou. O treino da ablação tem comando próprio;
  reprodução completa chama treino e ablação explicitamente. O cache exige
  checkpoint e metadados compatíveis, além do JSON.
- A correção fixa de vida máxima 3→16 ganhou uma comparação do mesmo objeto,
  na mesma sequência e nos mesmos quadros do primeiro caso da galeria. O JSON
  acompanha o ID original, a associação ao GT, as ausências e a IoU. A figura
  destaca a região do objeto. Essa análise do teste é declarada pós-hoc; não
  altera o modelo principal nem substitui o resultado agregado na validação.
- `data/video.py` e a inferência aceitam MOT, imagens e vídeos em streaming.
  O detector público é inferido do sufixo DPM/FRCNN/SDP; a fonte automática usa
  torchvision quando faltam detecções públicas. O notebook expõe detector e FPS,
  e seus textos com acentos corrompidos foram corrigidos.
- As tabelas, gráficos e JSONs distinguem `mAP_frame_mean` de `mAP_sequence`.
  O AP individual fica em `mAP_per_frame`; quadros sem GT recebem AP=0 e entram
  na média por quadro. A chave antiga `mAP` mantém a agregação por sequência.
- O manifesto passa a cobrir recursivamente código, testes, documentação,
  entradas, checkpoints e artefatos, sem incluir o próprio hash. Descreve o
  snapshot atual, sem atribuir os treinos históricos ao código refatorado.

Verificações: 25 testes automatizados passaram. Foram executadas a avaliação
das quatro sequências reservadas, a memória, o estresse, a galeria, a correção
agregada e a correção no mesmo caso. A comparação dos dois detectores foi
recalculada usando o cache existente de 525 quadros. Os pesos principais foram
preservados; as 36 configurações de ablação não foram retreinadas nesta revisão.
No caso 1, o ID original é mantido no quadro 173 com vida máxima 16; isso não
elimina a piora agregada de IDF1 na validação, associada ao aumento de IDFP.

A abertura automática de um kernel Jupyter pelo nbclient encontrou uma restrição
de permissões do Windows ao configurar o arquivo de conexão. A validação das
células usa execução sequencial em Python, sem desativar a proteção do Jupyter.
Todas as células de código foram executadas sobre os 525 quadros, gerando 55
identidades e preservando o checkpoint. O detector real também foi executado
sobre uma pasta PNG e sobre um MP4, além dos testes automatizados sem downloads.
