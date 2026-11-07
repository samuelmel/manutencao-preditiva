# Guia das métricas

Os números que aparecem no projeto e o que cada um quer dizer.

## RMSE: o erro médio em ciclos

Raiz do erro quadrático médio. Se o RMSE é 13, o modelo erra, em média, uns 13 ciclos de vida restante. Erros
grandes pesam mais que erros pequenos porque entram ao quadrado. Menor é melhor.

## Score da NASA: errar para mais custa mais caro

O RMSE trata igual prever a falha cedo demais e tarde demais. Na manutenção, as duas coisas não custam o mesmo:

- Prever cedo demais (o motor ainda tinha vida): você troca a peça antes da hora e perde um pouco de dinheiro.
- Prever tarde demais (o motor falha antes do aviso): parada não planejada, risco de segurança, custo alto.

O Score do C-MAPSS embute essa diferença. Com `d = RUL previsto − RUL real`:

```
d < 0   (previsão adiantada):   exp(−d / 13) − 1
d ≥ 0   (previsão atrasada):    exp( d / 10) − 1
```

O Score é a soma disso sobre todos os motores. Menor é melhor, e zero seria perfeito. Os dois lados crescem de forma
exponencial, mas o atrasado cresce mais depressa (divisor 10 contra 13):

| Erro de | Adiantada (cedo demais) | Atrasada (tarde demais) |
|---|---|---|
| 10 ciclos | 1,16 | 1,72 |
| 20 ciclos | 3,66 | 6,39 |
| 30 ciclos | 9,1 | 19,1 |

Um erro de 30 ciclos para mais custa o dobro de um erro de 30 para menos, e a penalidade explode com o tamanho do erro.
Por isso o Score pune os erros grandes bem mais que o RMSE. No FD001: Random Forest 218, LSTM entre 250 e 270, média
constante 19.887.

## F1-macro e recall da classe crítica

Para a triagem de OS, com quatro gravidades:

- Recall de uma classe: das OS que eram de fato daquela gravidade, que fração o modelo acertou. O recall da
  classe crítica é a métrica que mais importa, porque tratar uma OS crítica como baixa é o pior erro possível.
- F1: equilíbrio entre acertar as OS que o modelo aponta (precisão) e não deixar nenhuma de fora (recall).
- F1-macro: a média simples do F1 das quatro classes. Cada classe pesa igual, mesmo as raras.

Cada resultado sai em dois rótulos. O ruidoso é o que o sistema real veria (10% das OS têm o rótulo trocado de
propósito). O de referência é a gravidade derivada do RUL, sem troca, e mede o modelo de fato.

## PSI e KS: o dado mudou?

Servem para avisar quando o que chega ao sistema deixa de se parecer com o que ele viu no treino.

- PSI (índice de estabilidade da população): divide os valores da referência em 10 faixas e compara a proporção
  atual em cada faixa. Regra comum: abaixo de 0,1 estável, entre 0,1 e 0,2 atenção, acima de 0,2 mudou. O projeto
  usa 0,2.
- KS (Kolmogorov-Smirnov): teste estatístico que mede a maior distância entre as duas distribuições acumuladas.
  Calculado só sob demanda no `/drift`, porque é caro.

Gatilhos de retreino do projeto:
1. PSI acima de 0,2 em mais de 3 sensores.
2. Queda de 10 pontos ou mais no F1-macro numa amostra rotulada (`POST /triagem/avaliar`).

## Bootstrap: o resultado é real ou sorte?

Com só 100 motores de teste, uma diferença de um ciclo no RMSE pode ser sorte. O bootstrap sorteia motores (com
reposição) 2.000 vezes e recalcula a diferença a cada sorteio. Se o intervalo de 95% inclui o zero, a diferença não é
distinguível de ruído. Foi assim que o projeto mostrou que o Random Forest é provavelmente melhor que o LSTM no FD001
(95% de chance), e que a mistura dos dois não é melhor que o Random Forest sozinho.
