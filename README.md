# PA2 - Identidade ao longo do tempo

Implementacao modular de multi-object tracking (MOT) para o PA2 de Aprendizado Profundo. O repositorio comeca pelo ambiente sintetico reproduzivel e aceita sequencias MOT17 sem depender de trackers prontos.

## O que esta implementado

- gerador 128x128 de elipses com profundidade, oclusao real e ground truth;
- simulador de deteccoes com FN, ruido e FP;
- IoU, NMS proprio, associacao gulosa e Hungarian opcional;
- gestao explicita de nascimento, idade, oclusao e morte de tracks;
- IDF1, ID switches, fragmentacoes e erro de contagem, sem `motmetrics`/TrackEval;
- baseline por quadro e modelo temporal GRU/LSTM/RNN de movimento;
- leitura do formato MOT17, treino, ablaçao de celula, horizonte de memoria e estresse;
- notebook `inferencia.ipynb` para inferencia em qualquer sequencia com deteccoes MOT17;
- testes sintéticos e casos construidos manualmente.

## Ambiente

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[full,test]"
```

Para testar o núcleo sem baixar dados:

```powershell
pytest -q
python -m mot_pa2.cli synthetic --frames 40 --objects 8 --output outputs/synthetic
```

## MOT17

Baixe o pacote de anotações e/ou uma sequência do [MOTChallenge](https://motchallenge.net/data/MOT17/) e coloque-o em `data/MOT17`. O código espera `train/<SEQ>/{gt,det,img1}` no layout oficial.

```powershell
python -m mot_pa2.cli evaluate-mot --sequence MOT17-04 --root data/MOT17 --detector FRCNN
python -m mot_pa2.cli train --root data/MOT17 --sequence MOT17-04 --epochs 10 --checkpoint checkpoints/gru.pt
python -m mot_pa2.cli stress --output outputs/stress
```

As sequências devem ser separadas inteiras entre treino, validação e teste. A escolha padrão recomendada é treinar em `MOT17-02, MOT17-04`, validar em `MOT17-05` e reservar `MOT17-09` para teste, registrando qualquer alteração no `AI_LOG.md`.

## Organização

`src/` contém dados, detecçao, associaçao, trackers, treino e visualizaçao. `metrics.py` é a API pedida pelo enunciado. `tests/` valida o comportamento determinístico. `outputs/` e `checkpoints/` são gerados e ignorados pelo Git.

O notebook não treina novamente: recebe uma sequência, roda a fonte de detecçoes selecionada, colore IDs de forma consistente e exporta o vídeo anotado.
