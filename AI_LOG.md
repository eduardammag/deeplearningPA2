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
