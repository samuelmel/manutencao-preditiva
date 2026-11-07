# Como o projeto foi feito

Este texto conta o caminho do projeto: o que foi planejado, o que funcionou de primeira, o que deu errado e o que mudou
por causa disso. O resultado final está no [README](README.md), e os números em
[docs/RESULTADOS.md](docs/RESULTADOS.md). Aqui entra o raciocínio, inclusive as partes em que o plano não se confirmou.

## O ponto de partida

O plano era um MVP com dois módulos que conversam: previsão de vida útil de motores com um LSTM e triagem de ordens de
serviço com processamento de linguagem natural. As restrições eram claras desde o começo. O projeto seria de portfólio,
publicado no GitHub, com treino no Colab e no máximo de 2 a 5 GB de disco no total. Visão computacional e áudio ficariam
para uma fase opcional no final.

Três regras do plano guiaram quase tudo. Todo modelo precisa de um baseline simples ao lado, porque um número sozinho não
diz se o modelo é bom. A divisão entre treino e teste é sempre por motor, nunca por linha, para uma janela do mesmo motor
não vazar de um lado para o outro. E o que for sintético precisa ser dito como sintético.

## Fase 0: os dados

A primeira camada foi a de ingestão. Os arquivos originais do C-MAPSS são copiados byte a byte para o bronze, com
checksum, e o pipeline falha se algum mudar. No silver entram as decisões que mais influenciam o modelo: o RUL é calculado
com teto de 125 ciclos, os sensores constantes são descobertos pela variância no treino (e não por uma lista escrita à
mão) e a normalização é ajustada só no treino. No FD002 e no FD004, onde o motor opera em várias condições, a
normalização é feita por grupo, com K-Means.

Os testes dessa fase usam arquivos falsos gerados na hora, porque o objetivo era garantir a lógica antes de depender do
download. O C-MAPSS real (12 MB, do repositório da NASA) só entrou depois, quando a fase 1 precisou dele. O GitHub Actions
rodando lint e testes também nasceu aqui.

## Fase 1: o LSTM que não convergia

A primeira versão do treino não aprendia. A perda quase não descia, e o commit dessa etapa até dizia na mensagem que o
LSTM ainda não convergia. A causa estava em duas linhas que não se conheciam. O modelo terminava numa camada que
multiplicava a saída por 125, e o script de treino já dividia o alvo por 125. A escala era aplicada duas vezes, e a perda
comparava números de ordens de grandeza diferentes. Depois de separar as responsabilidades (o modelo treina em escala de
0 a 1 e uma camada extra converte para ciclos só na hora de servir), o erro médio caiu para 13 ciclos.

O baseline mostrou o outro problema. Um Random Forest sobre quatro estatísticas da janela (média, desvio, último valor e
inclinação) erra 11,96 ciclos, menos que o LSTM. Sem esse baseline, 13 ciclos pareceria um bom resultado, já que chutar a
média erra 40,7.

O que veio depois foi uma sequência de tentativas de fazer o LSTM ganhar, cada uma com hipótese própria:

1. Doze treinos variando batch, learning rate, dropout e scheduler, com três seeds cada. Nada passou de 0,2 ciclo de
   diferença, o que é ruído.
2. Janelas de 50 e 70 ciclos. A validação melhorou e o teste piorou, para os dois modelos. A explicação está no teste, que
   avalia só o fim de cada motor: alguns motores têm poucos ciclos, e a janela maior é preenchida com o primeiro valor
   repetido, o que cria informação falsa.
3. Média de cinco LSTMs e mistura com o Random Forest. A média quase não ajudou, porque os modelos erram do mesmo jeito. A
   mistura ficou 0,2 ciclo melhor, mas o bootstrap mostrou que isso é indistinguível de sorte, e a proporção da mistura
   tinha sido escolhida olhando o teste.
4. Um modelo híbrido, com um ramo LSTM e outro ramo denso alimentado pelas mesmas estatísticas do Random Forest. Melhorou
   (12,62), mas continuou atrás.

Com só 100 motores de teste, nenhuma diferença pequena merece confiança, e foi isso que levou ao bootstrap (2.000
reamostragens). O resultado ficou documentado como é: no FD001, o Random Forest é provavelmente melhor que o LSTM. O
FD001 é o conjunto mais simples, e a comparação nos outros três ficou como trabalho futuro.

## Fase 2: texto que não existe

Não há base pública de ordens de serviço em português. A solução foi gerar o texto por templates, ligado ao RUL real de
cada motor, para que cada ordem tenha um motor e um ciclo e a fusão com o módulo de vida útil possa ser testada de ponta a
ponta.

O risco óbvio de um corpus sintético é ficar fácil demais e produzir um F1 perto de 1,0 que não significa nada. Para evitar
isso o gerador mistura sintomas de gravidades vizinhas em 30% das ordens, escreve com abreviações e erros de digitação, e
troca 10% dos rótulos de propósito. O baseline TF-IDF com regressão logística deu F1-macro de 0,69, longe de 1,0, que é o
que se queria. O NER por regras achou o código de peça em 100% das ordens, o Word2Vec recuperou ordens parecidas mesmo com
palavras escritas errado, e o LDA ficou com coerência baixa e tópicos que repetem as palavras dos templates, o que é
esperado em texto gerado por template.

Um detalhe de ambiente atrasou essa fase. O ambiente virtual estava em Python 3.14, e o `gensim` ainda não tem pacote
pronto para essa versão, então foi preciso criar um segundo ambiente em Python 3.12, que é a versão que o Makefile e o CI
já indicavam.

## Fase 3: do experimento ao serviço

Esta foi a fase mais longa. A API (FastAPI com ONNX Runtime) calcula o RUL, tria a ordem, combina os dois numa nota e
expõe métricas ao Prometheus. O Grafana lê essas métricas e a fila de prioridade, e tudo sobe com um comando de compose.
Duas decisões do plano valeram a pena: tirar o PyTorch da imagem (777 MB no fim) e registrar dashboards e alertas em
arquivos, sem clicar em interface.

Vários problemas só apareceram quando o sistema rodou de verdade:

- O teste KS rodava a cada requisição de `/rul`, e a latência foi para 170 ms contra 1 ms da previsão. O PSI passou a ser
  recalculado a cada 20 previsões, e o KS ficou sob demanda.
- O alerta de drift de classes das ordens disparava sempre. O corpus de treino é balanceado por construção, e qualquer
  população real parece diferente dele. A referência passou a ser uma linha de base formada pelas primeiras 100 ordens em
  produção.
- O simulador de drift deslocava sensores que o modelo nem usa (três deles são constantes e foram descartados), então o
  alerta, corretamente, não disparava. O erro era do simulador.
- O Grafana mostrava a tabela da fila com erro no navegador, embora as consultas pela API funcionassem. O plugin Infinity
  era instalado na versão mais nova, que exige Grafana 11.6 ou superior, e o projeto usa o 11.3. A versão foi fixada em
  3.7.1. A lição foi abrir o painel num navegador de verdade: nenhuma verificação pela API teria achado isso.
- O Docker Desktop não estava em execução, e o ambiente passou a usar Podman. O `podman` ficou fora do PATH até o terminal
  ser reaberto, e um `machine start` interrompido deixou a máquina num estado em que o compose não conseguia falar com
  ela. Reiniciar a máquina resolveu. Os nomes de imagem ganharam o prefixo `docker.io/`, porque o Podman não resolve nomes
  curtos.

O DistilBERT veio por último. O PyTorch instalado era só para CPU, e a versão com CUDA pesa uns 2,5 GB, então o treino foi
para o Colab. Antes de gastar tempo lá, o pipeline inteiro (treino, exportação ONNX, quantização INT8 e leitura pelo
serviço) foi testado com um DistilBERT minúsculo de pesos aleatórios, o que pegou erros de código sem baixar o modelo de
verdade. O treino real levou 143 segundos numa GPU T4. O resultado empatou com o TF-IDF (F1-macro de 0,696 contra 0,693),
pelo mesmo motivo do corpus: o limite de acerto está no desenho dos dados, não no modelo.

O último item do plano era o gatilho de retreino. A API recebe uma amostra rotulada, calcula o F1-macro e dispara um
alerta se ele cair 10 pontos em relação ao do treino. No contêiner, uma amostra boa deu queda de 3,4 pontos e uma amostra
com rótulos embaralhados deu 47,5 pontos, e o alerta correspondente ficou em `firing` no Grafana.

## Erros que custaram tempo

Além dos já contados, alguns tropeços foram de processo e não de modelagem:

- Um teste do DistilBERT minúsculo passava às vezes e falhava outras, com a mesma entrada. A causa era o treinador de
  vocabulário do `tokenizers`, que roda em paralelo e gera vocabulários diferentes a cada execução. O vocabulário passou a
  ser montado de forma determinística.
- Um commit saiu com erros de lint porque o comando encadeado terminava num `tail`, que sempre devolve sucesso e escondia a
  falha do `ruff`. Passou a valer conferir o código de saída de cada passo.
- O Makefile estava com finais de linha do Windows, e uma edição automática não foi aplicada sem avisar. O README chegou a
  citar comandos `make` que não existiam. Hoje há um teste que confere se todo `make` citado na documentação existe, e um
  `.gitattributes` mantém o arquivo com finais de linha do Unix.
- Três commits antigos estavam com um autor automático e foram reescritos para o autor do repositório, com um push forçado
  protegido por `--force-with-lease`.

## O que ficou de fora

- Visão computacional (MVTec AD), áudio (MIMII) e resumos gerados por modelo de linguagem, a fase 4 opcional. O áudio pede
  arquivos de 7 a 11 GB por tipo de máquina, o que passa do orçamento de disco. Entram, talvez, numa próxima rodada.
- Ordens de serviço reais. Sem elas, não dá para saber se o DistilBERT vale o custo extra sobre o TF-IDF.
- O LSTM nos conjuntos FD002 a FD004, que é onde ele tem mais chance de superar o Random Forest.
- Publicação dos modelos em GitHub Releases e registro dos experimentos em MLflow.
- Teste do contêiner em arquitetura arm64 e em hardware de fábrica.

## O que o processo ensinou

O baseline simples decidiu mais do que qualquer ajuste fino de modelo. Os dois resultados que contrariaram o plano, o
LSTM abaixo do Random Forest e o DistilBERT igual ao TF-IDF, só ficaram visíveis porque havia uma referência barata do lado.
Quando o conjunto de teste é pequeno, vale medir a incerteza antes de comemorar uma diferença. E um sistema só está
verificado quando alguém olha o resultado onde o usuário olharia, como o painel num navegador, e não só a resposta da API.
