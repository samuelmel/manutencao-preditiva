# Como o LSTM prevê a vida útil restante

Este texto explica a ideia por trás do modelo de RUL deste projeto. Não precisa de cálculo avançado, só de
paciência com algumas fórmulas.

## O problema

Um motor gera 14 leituras de sensor (temperatura, pressão, rotação e outras) a cada ciclo de uso. Uma leitura
isolada diz pouco: o que importa é como os valores **mudam ao longo dos ciclos**. Um motor saudável tem leituras
estáveis; um motor perto de falhar mostra uma deriva lenta. Por isso o modelo recebe uma **janela de 30 ciclos
seguidos** (uma matriz de 30 linhas por 14 colunas) e devolve um número: quantos ciclos ainda restam.

Uma rede comum, que olha a janela inteira de uma vez, ignoraria a ordem dos ciclos. Uma rede **recorrente** lê um
ciclo por vez, na ordem, e carrega consigo uma memória do que já viu.

## Estado oculto: a memória da rede

A rede recorrente mais simples mantém um vetor `h`, o **estado oculto**. A cada ciclo `t` ela combina a leitura nova
`x_t` com a memória anterior `h_(t-1)`:

```
h_t = tanh(W · x_t + U · h_(t-1) + b)
```

Ao terminar os 30 ciclos, o último `h` resume a janela e uma camada final transforma esse resumo no RUL previsto.

## Pesos compartilhados entre os passos de tempo

As matrizes `W`, `U` e o vetor `b` são **os mesmos nos 30 passos**. A rede não aprende uma regra para o ciclo 1,
outra para o ciclo 2 e assim por diante: aprende uma única regra de "como atualizar a memória ao ver uma leitura" e a
reaplica 30 vezes. Isso tem duas consequências boas. O número de parâmetros não cresce com o tamanho da janela, e o
que a rede aprende vale em qualquer posição da janela.

## Treinar: BPTT

Para ajustar `W`, `U` e `b`, o treino precisa saber como o erro final depende deles. Como a mesma regra foi aplicada
30 vezes, "desenrolamos" a rede no tempo, como se fossem 30 camadas empilhadas com pesos iguais, e aplicamos a
retropropagação normal. Isso se chama **BPTT** (retropropagação através do tempo). O gradiente do erro em relação
ao estado do ciclo 1 passa por todos os 29 ciclos seguintes:

```
∂Erro/∂h_1  =  ∂Erro/∂h_30 · (∂h_30/∂h_29) · (∂h_29/∂h_28) · ... · (∂h_2/∂h_1)
```

## O gradiente que desaparece

Cada fator `∂h_t/∂h_(t-1)` na rede simples é `diag(1 - h_t²) · U`. A derivada da `tanh` nunca passa de 1 e, quando a
`tanh` satura, fica perto de zero. Se `U` também for "pequena" (maior valor singular abaixo de 1), o produto de 29
fatores encolhe de forma exponencial. O sinal de erro que chega ao começo da janela é praticamente nulo, e a rede
não consegue aprender que um sinal fraco lá atrás explica a falha de agora. Se `U` for "grande", acontece o oposto: o
gradiente explode.

## O que o LSTM muda

O LSTM acrescenta um segundo vetor, o **estado da célula** `c`, e três **portões** que controlam o que entra, o que
fica e o que sai. Cada portão é uma pequena camada com saída entre 0 e 1:

```
f_t = σ(W_f x_t + U_f h_(t-1) + b_f)        portão de esquecimento
i_t = σ(W_i x_t + U_i h_(t-1) + b_i)        portão de entrada
o_t = σ(W_o x_t + U_o h_(t-1) + b_o)        portão de saída
g_t = tanh(W_g x_t + U_g h_(t-1) + b_g)     candidato a nova memória

c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t             memória: soma, não substituição
h_t = o_t ⊙ tanh(c_t)                       o que a célula mostra para fora
```

O ponto decisivo é a linha de `c_t`. A memória antiga passa para o próximo passo **multiplicada só por `f_t`** e
somada ao que há de novo. O gradiente que volta ao longo de `c` também é multiplicado por `f_t`, sem passar por uma
`tanh` nem por uma matriz cheia a cada passo. Quando a rede aprende `f_t` perto de 1, o erro viaja muitos ciclos para
trás quase sem perder força. Os portões deixam a rede escolher o que guardar (uma deriva lenta de temperatura) e o que
descartar (ruído de um ciclo).

Em janelas de 30 ciclos o problema do gradiente é mais brando do que em textos longos, mas a mesma ideia explica por
que o LSTM costuma ser a primeira escolha para séries de sensores.

## A rede deste projeto

| Camada | Configuração | Parâmetros |
|---|---|---|
| Entrada | 30 ciclos × 14 sensores | 0 |
| LSTM | 64 unidades, devolve a sequência toda | 20.224 |
| Dropout | 0,2 | 0 |
| LSTM | 32 unidades, devolve só o último estado | 12.416 |
| Dropout | 0,2 | 0 |
| Dense | 16 neurônios, ReLU | 528 |
| Dense | 1 neurônio, linear (RUL ÷ 125) | 17 |
| **Total** | | **33.185** |

O primeiro LSTM entrega um vetor de 64 números por ciclo, e o segundo lê essa sequência e a resume em 32 números. As
duas camadas permitem que a primeira aprenda padrões curtos (a leitura subiu e desceu) e a segunda padrões mais longos
(a tendência dos últimos 30 ciclos). O dropout desliga 20% das unidades ao acaso no treino, para a rede não decorar os
motores. O alvo é o RUL dividido por 125, porque o RUL é limitado em 125 ciclos (no início da vida do motor não há
degradação observável) e uma saída entre 0 e 1 treina melhor.

A fórmula de cada LSTM é `4 · unidades · (entradas + unidades + 1)`. Para o primeiro: `4 · 64 · (14 + 64 + 1) = 20.224`.

## O que o LSTM entregou aqui

Pouco, e vale dizer. No FD001, o LSTM ficou com erro médio de cerca de 13 ciclos, e um Random Forest sobre estatísticas
da janela (média, desvio, último valor e inclinação) ficou com 11,96. Os experimentos estão em
[`RESULTADOS.md`](RESULTADOS.md). O FD001 tem poucos motores e degradação quase linear, o que favorece modelos simples;
a vantagem do LSTM costuma aparecer em conjuntos mais difíceis, como FD002 e FD004.
