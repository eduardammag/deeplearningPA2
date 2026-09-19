# Resultados para a apresentação

Valores executados; configuração principal GRU/T=16/seed=0, sem escolha pelo teste.

| Sequência | Baseline IDF1 | Temporal IDF1 | mAP |
|---|---:|---:|---:|
| MOT17-09 | 0.5049 | 0.5047 | 0.4754 |
| MOT17-10 | 0.3741 | 0.3770 | 0.3826 |
| MOT17-11 | 0.5373 | 0.5395 | 0.5026 |
| MOT17-13 | 0.3534 | 0.3568 | 0.3917 |

## Ablação na validação

| Célula/janela | IDF1 média ± desvio |
|---|---:|
| rnn_T4 | 0.5426 ± 0.0012 |
| rnn_T8 | 0.5426 ± 0.0011 |
| rnn_T16 | 0.5434 ± 0.0002 |
| rnn_T32 | 0.5424 ± 0.0016 |
| lstm_T4 | 0.5424 ± 0.0011 |
| lstm_T8 | 0.5426 ± 0.0031 |
| lstm_T16 | 0.5420 ± 0.0010 |
| lstm_T32 | 0.5425 ± 0.0012 |
| gru_T4 | 0.5411 ± 0.0003 |
| gru_T8 | 0.5426 ± 0.0019 |
| gru_T16 | 0.5429 ± 0.0004 |
| gru_T32 | 0.5428 ± 0.0007 |

## Horizonte de memória

| Célula | Redução da norma em 8 passos | Redução em 16 passos |
|---|---:|---:|
| rnn | 90.7× | 7090.6× |
| lstm | 779.6× | 41106.3× |
| gru | 76.9× | 3171.9× |

Nas 24 trajetórias isoladas, o maior intervalo recuperado variou de 3 a 3 quadros. A política mata tracks depois de três ausências; esse limite domina o horizonte empírico desta configuração.

A ablação não mostrou uma quebra clara da RNN em IDF1 ao aumentar T. O gradiente se atenua fortemente em todas as células neste treino. Portas não garantem memória longa automaticamente. Na LSTM, a curva mede h, não o caminho independente de c.

## Estresse sem retreino

| Intensidade | mAP | Baseline IDF1 | Temporal IDF1 |
|---|---:|---:|---:|
| 1 | 0.4212 | 0.4426 | 0.4425 |
| 2 | 0.3483 | 0.3605 | 0.3607 |
| 3 | 0.2102 | 0.1695 | 0.1695 |

As diferenças entre temporal e baseline são pequenas neste experimento: não há evidência de absorção relevante da degradação do detector. A perda de detecções e o ruído fragmentam ambas as soluções.

## Duas fontes de detecção, sequência 09 inteira

| Fonte | mAP | IDF1 | ID switches |
|---|---:|---:|---:|
| public_FRCNN | 0.4754 | 0.5049 | 38 |
| torchvision_COCO | 0.2657 | 0.3192 | 220 |

## Interpretação

A previsão geométrica recorrente não garante ganho: examine os valores acima. O treinamento por teacher forcing e a perda de caixa não otimizam diretamente identidade. Sob oclusão, o modelo realimenta previsões; erros acumulam e o portão de IoU pode rejeitar a observação correta.

Increasing lifetime changes IDF1 by -0.03507141347764975; a nonpositive change means longer survival alone does not solve association/prediction errors.

A sobrevivência empírica inclui o limite max_missed=3. Não confundir esse limite de engenharia com desaparecimento do gradiente. A figura analítica mede estados ocultos; a empírica mede manutenção da identidade após oclusões injetadas em trajetórias reais isoladas.

As curvas e tabelas permitem verificar se há evidência de colapso da RNN simples. Se as médias forem próximas e seus desvios se sobrepuserem, não afirmar que a ablação demonstrou superioridade das portas; o horizonte de gradiente e a dificuldade dos dados precisam ser considerados.

Veja failures/diagnoses.json para os três erros concretos, visibilidade, IoU da previsão e figuras. As limitações do protocolo de métricas e a costura de janelas estão no README.
