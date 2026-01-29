# Hackathon CGDF 2026 - Categoria Acesso à Informação
## Classificação Automática de Pedidos com Dados Pessoais

**Solução desenvolvida para identificar automaticamente pedidos de acesso à informação que contêm dados pessoais.**

---

### 📌 Classificação:

- **Classe 0 = PÚBLICO**: Pedido NÃO contém dados pessoais
- **Classe 1 = NÃO PÚBLICO**: Pedido CONTÉM dados pessoais

O modelo identifica automaticamente pedidos que contenham informações pessoais 
e que, portanto, deveriam ser classificados como não públicos.


## 📁 Estrutura de Pastas do Projeto

### ⚠️ Observação Importante para Execução

Todos os comandos descritos neste documento devem ser executados **no diretório raiz
do projeto**, denominado:

submissao_cgdf_lai/

Esse diretório é aquele obtido após a extração do arquivo compactado da solução e
contém os arquivos `README.md`, `requirements.txt` e as pastas `src/`, `models/` e
`data/`.

O projeto está organizado da seguinte forma:
```
submissao_cgdf_lai/
│
├── 📄 README.md                          # Esta documentação
├── 📄 requirements.txt                   # Dependências (pip install -r requirements.txt)
│
├── 📂 src/                               # Código-fonte
│   └── predicao_submissao.py            # Script principal de predição
│
├── 📂 models/                            # Modelos treinados
│   ├── modelo_otimizado_maximo.pkl      # Modelo ensemble (1.04 MB)
│   ├── vectorizer_otimizado.pkl         # Vectorizador TF-IDF (129 KB)
│   └── config_otimizado.pkl             # Configurações (238 B)
│
└── 📂 data/                              # Diretório para dados
    ├── input/                            # Arquivos de entrada (teste)
    │   └── instrucoesDadosTeste.txt            # Instruções para avaliadores
    └── output/                           # Resultados das predições
        └── InstrucoesDadosResultados.txt           # Explicação dos resultados
```

### 🎯 Organização (Critério 3c)

**Por que esta estrutura?**

- **`src/`**: Mantém o código-fonte separado e organizado
- **`models/`**: Centraliza todos os modelos treinados (fácil de localizar)
- **`data/`**: Separa claramente entrada e saída
  - **`input/`**: Local recomendado para colocar arquivos de teste
  - **`output/`**: Local onde os resultados são salvos
- **Raiz**: Apenas documentação e configuração


| Arquivo | Função |
|---------|--------|
| `predicao_submissao.py` | **Script principal de predição**. Contém toda a lógica para carregar modelos, extrair features e fazer predições. |
| `modelo_otimizado_maximo.pkl` | **Modelo ensemble treinado** (1.04 MB). Combina Random Forest, Logistic Regression e Gradient Boosting. |
| `vectorizer_otimizado.pkl` | **Vectorizador TF-IDF treinado** (129 KB). Converte texto em 3000 features numéricas. |
| `config_otimizado.pkl` | **Configurações do modelo** (238 bytes). Armazena threshold otimizado (0.50) e lista de modelos BERT. |
| `requirements.txt` | **Dependências do projeto**. Lista todas as bibliotecas necessárias com versões específicas. |
| `README.md` | **Este arquivo**. Documentação completa com instruções de instalação e execução. |

---



## 🔧 1. Instruções de Instalação e Dependências

### 1.1 Pré-requisitos

**Software necessário:**
- **Python 3.11.9** (testado em Python 3.11.9)
- **pip** (gerenciador de pacotes do Python, geralmente incluído)

**Hardware recomendado:**
- 4 GB de RAM mínimo (8 GB recomendado)
- 2 GB de espaço em disco
- CPU: qualquer processador moderno (GPU opcional, acelera em 10x)

**Verificar versão do Python:**
```bash
python --version
```
Saída esperada: `Python 3.9.x`, `Python 3.10.x` ou `Python 3.11.x`

Se não tiver Python instalado, baixe em: https://www.python.org/downloads/

---

### 1.2 Criar Ambiente Virtual

Um ambiente virtual isola as dependências do projeto, evitando conflitos.

**No Windows:**
```bash
# 1. Criar ambiente virtual
python -m venv venv

# 2. Ativar ambiente virtual
venv\Scripts\Activate.ps1

# Verificar ativação: o prompt deve mostrar (venv) no início
```

**No Linux/Mac:**
```bash
# 1. Criar ambiente virtual
python3 -m venv venv

# 2. Ativar ambiente virtual
source venv/bin/activate

# Verificar ativação: o prompt deve mostrar (venv) no início
```

---

### 1.3 Instalar Dependências

O arquivo `requirements.txt` contém todas as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

**Dependências instaladas:**

   joblib==1.5.3
   numpy==2.4.1
   openpyxl==3.1.5
   pandas==2.2.2
   scipy==1.16.3
   scikit-learn==1.8.0
   torch==2.9.0
   transformers==4.57.6


**Tempo de instalação:** 5 a 10 minutos

---

### 1.4 Verificar Instalação

```bash
python -c "import pandas, sklearn, transformers, torch; print('✓ Instalação bem-sucedida!')"
```

Saída esperada: `✓ Instalação bem-sucedida!`

---

### 2.1 Comando de Execução

**Estrutura recomendada:**
```bash
python src/predicao_submissao.py data/input/<arquivo_teste.xlsx> data/output/<resultado.xlsx>
```

**Exemplo prático:**
```bash
# 1. Coloque seu arquivo de teste em data/input/
# (exemplo: data/input/controle_cgdf.xlsx)

# 2. Execute:
python src/predicao_submissao.py data/input/controle_cgdf.xlsx data/output/resultado_oficial.xlsx

# 3. O resultado será salvo em data/output/resultado_oficial.xlsx
```

**Forma alternativa (caminhos personalizados):**

Se preferir usar outros caminhos:
```bash
python src/predicao_submissao.py C:\Downloads\teste.xlsx resultado.xlsx
```

O script é flexível e aceita qualquer caminho válido para entregar os resultados.

---

### 📂 Dica para Avaliadores:

**Não sabe onde colocar o arquivo de teste?**

1. Coloque em `data/input/` (recomendado)
2. Consulte o arquivo `data\input\instrucoesDadosTeste.txt` e o `data\output\InstrucoesDadosResultados.txt` para instruções detalhadas de entrada e saida.
3. Após execução, verifique `data/output/` para os resultados
4. Leia `data\output\InstrucoesDadosResultados.txt` para entender o formato da saída
---

### 2.2 Formato de Dados de Entrada

**Arquivo:** Excel (.xlsx)

**Coluna obrigatória:**
- `Texto Mascarado` (string): Contém o texto do pedido de acesso à informação

**Coluna opcional:**
- `ID` (inteiro ou string): Identificador único do pedido
  - Se não existir, o script gerará IDs automaticamente (1, 2, 3, ...)

**Exemplo de estrutura de entrada:**

| ID | Texto Mascarado |
|----|----------------|
| 1  | Solicito informações sobre editais de concursos públicos do DF |
| 2  | Preciso de uma cópia do meu prontuário médico do hospital regional |
| 3  | Gostaria de saber quais são os horários de atendimento da ouvidoria |

**Observações:**
- O arquivo deve estar em formato Excel (.xlsx)
- A coluna "Texto Mascarado" pode conter textos de qualquer tamanho
- Caracteres especiais e acentuação são suportados
- Linhas vazias ou com texto vazio serão processadas normalmente

---

### 2.3 Formato de Dados de Saída

**Arquivo:** Excel (.xlsx)

**Colunas geradas:**

| Coluna | Tipo | Descrição | Exemplo |
|--------|------|-----------|---------|
| `ID` | int | Identificador do pedido (copiado da entrada ou gerado) | 1, 2, 3... |
| `Texto Mascarado` | string | Texto original do pedido (copiado para facilitar verificação) | "Solicito informações..." |
| `Classificação` | **int** | **0** = PÚBLICO (não contém dados pessoais)<br>**1** = NÃO PÚBLICO (contém dados pessoais) | 0 ou 1 |
| `Status` | string | Classificação textual:<br>**"PÚBLICO"** ou **"NÃO PÚBLICO"** | PÚBLICO |
| `Justificativa` | string | Explicação da classificação:<br>"Não contém dados pessoais" ou<br>"Contém dados pessoais" | Não contém dados pessoais |
| `Probabilidade_Dados_Pessoais` | float | Probabilidade de conter dados pessoais (0.0 a 1.0) | 0.15 |
| `Confiança` | float | Confiança da predição (0.0 a 1.0) | 0.85 |

---

### 📊 IMPORTANTE: Interpretação das Classificações

**De acordo com o edital do Hackathon CGDF:**

> "Os participantes desenvolvam modelos capazes de identificar automaticamente pedidos que contenham informações pessoais e que, portanto, deveriam ser classificados como não públicos."

**Portanto:**

```
┌─────────────────────────────────────────────────────────────┐
│  Classificação 0 = PÚBLICO                                  │
│  ↳ Pedido NÃO contém dados pessoais                        │
│  ↳ Pode ser divulgado publicamente                         │
│  ↳ Exemplo: "Horários de atendimento da ouvidoria"         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Classificação 1 = NÃO PÚBLICO                              │
│  ↳ Pedido CONTÉM dados pessoais                            │
│  ↳ NÃO deve ser divulgado publicamente                     │
│  ↳ Exemplo: "Solicito cópia do meu processo XXXX"      │
└─────────────────────────────────────────────────────────────┘
```

---

### 🔍 Exemplo Completo de Saída

**Arquivo de entrada (`teste.xlsx`):**

| ID | Texto Mascarado |
|----|----------------|
| 1  | Solicito informações sobre editais de concursos públicos do DF |
| 2  | Preciso de uma cópia do meu prontuário médico do hospital regional |
| 3  | Gostaria de saber quais são os horários de atendimento da ouvidoria |

**Arquivo de saída gerado (`resultado.xlsx`):**

| ID | Texto Mascarado | Classificação | Status | Justificativa | Probabilidade_Dados_Pessoais | Confiança |
|----|----------------|---------------|--------|---------------|------------------------------|-----------|
| 1  | Solicito informações sobre editais... | **0** | **PÚBLICO** | Não contém dados pessoais | 0.12 | 0.88 |
| 2  | Preciso de uma cópia do meu prontuário... | **1** | **NÃO PÚBLICO** | Contém dados pessoais | 0.95 | 0.95 |
| 3  | Gostaria de saber quais são os horários... | **0** | **PÚBLICO** | Não contém dados pessoais | 0.08 | 0.92 |

---

### 📋 Interpretação Linha por Linha:

**Linha 1 (ID=1):**
- ✅ **Classificação: 0 (PÚBLICO)**
- ✅ Não contém dados pessoais
- ✅ Pedido genérico sobre editais de concursos
- ✅ Pode ser divulgado publicamente
- Confiança: 88% (alta)

**Linha 2 (ID=2):**
- 🔒 **Classificação: 1 (NÃO PÚBLICO)**
- 🔒 Contém dados pessoais ("meu prontuário")
- 🔒 Solicita documento específico do solicitante
- 🔒 NÃO deve ser divulgado publicamente
- Confiança: 95%

**Linha 3 (ID=3):**
- ✅ **Classificação: 0 (PÚBLICO)**
- ✅ Não contém dados pessoais
- ✅ Pergunta genérica sobre horários
- ✅ Pode ser divulgado publicamente
- Confiança: 92%

---

### ⚖️ Critérios de Classificação

O modelo classifica como **NÃO PÚBLICO (1)** quando detecta:

1. **Pronomes possessivos** indicando solicitação pessoal:
   - "meu", "minha", "meus", "minhas"
   - Exemplo: "**meu** prontuário", "**minha** ficha"

2. **Documentos pessoais específicos:**
   - Prontuário, laudo, exame, atestado, cadastro
   - Exemplo: "cópia do prontuário médico"

3. **Identificadores pessoais:**
   - CPF, matrícula, número de processo específico
   - Exemplo: "matrícula 12345", "processo 2024/001"

4. **Contexto individual:**
   - Verbos em primeira pessoa: "solicito", "preciso", "quero"
   - Exemplo: "**Solicito** acesso ao **meu** processo"

O modelo classifica como **PÚBLICO (0)** quando detecta:

1. **Perguntas genéricas** sobre políticas/procedimentos
2. **Solicitações de informações gerais** (sem identificação pessoal)
3. **Pedidos de dados estatísticos/agregados**
4. **Informações sobre serviços públicos** (horários, endereços, etc.)

---

### ⏱️ Tempo de Execução

O tempo varia conforme o número de registros:

| Registros | CPU (aprox.) | GPU (aprox.) |
|-----------|--------------|--------------|
| 10        | 30 segundos  | 5 segundos   |
| 100       | 3-4 minutos  | 30 segundos  |
| 500       | 15-18 minutos| 2-3 minutos  |
| 1000      | 30-35 minutos| 5-7 minutos  |

**Nota:** A primeira execução é mais lenta (~2 minutos extras) devido ao download dos modelos BERT do HuggingFace. Downloads subsequentes usam cache local.

---

## 🔬 3. Explicação da Lógica Implementada

### 3.1 Pipeline de Predição

O script executa 5 etapas principais:

```
1. Carregar dados de teste (Excel)
         ↓
2. Carregar modelos treinados (.pkl)
         ↓
3. Extrair features dos textos
   ├── Embeddings BERT (1536 dimensões)
   ├── TF-IDF (3000 dimensões)
   └── Features de regras (9 dimensões)
         ↓
4. Combinar features (4545 dimensões totais)
         ↓
5. Fazer predições com ensemble
         ↓
6. Salvar resultado (Excel)
```

---

### 3.2 Funções Principais

#### `extrair_features_adicionais(texto)`
**Propósito:** Extrai 9 features baseadas em regras que indicam dados pessoais.

**Lógica implementada:**
1. **Detecção de padrões:**
   - CPF (formato: 123.456.789-00)
   - Matrícula (palavra "matrícula" seguida de números)
   - Processo (palavra "processo" seguida de números)

2. **Contagem de palavras-chave:**
   - Termos como: cadastro, prontuário, laudo, exame, atestado, ficha
   - Indicam solicitação de documentos pessoais

3. **Análise de pronomes possessivos:**
   - Conta ocorrências de: meu, minha, meus, minhas
   - Forte indicador de pedido pessoal

4. **Identificação de verbos em 1ª pessoa:**
   - solicito, preciso, gostaria, quero, estou
   - Indicam solicitação individual

5. **Características do texto:**
   - Tamanho total (número de caracteres)
   - Número de palavras

**Por que funciona:** Pedidos com dados pessoais geralmente usam linguagem em primeira pessoa e solicitam documentos específicos sobre o solicitante.

---

#### `extrair_embeddings_dual_bert(textos)`
**Propósito:** Extrai representações semânticas profundas usando 2 modelos BERT.

**Modelos utilizados:**
1. **BERTimbau** (`neuralmind/bert-base-portuguese-cased`)
   - Especializado em português brasileiro
   - Treinado em 2.7 bilhões de palavras
   - Gera 768 dimensões por texto

2. **DistilBERT PT** (`adalbertojunior/distilbert-portuguese-cased`)
   - Versão mais leve e rápida
   - Mantém 95% da qualidade do BERT original
   - Gera 768 dimensões por texto

**Processo:**
1. Tokeniza o texto (converte palavras em números)
2. Passa pelo modelo BERT
3. Extrai o vetor do token [CLS] (representa o texto inteiro)
4. Concatena embeddings dos 2 modelos (768 + 768 = 1536 dimensões)

**Por que usar 2 modelos:** Aumenta a robustez. Cada modelo captura aspectos diferentes da linguagem, melhorando a generalização.

---

#### `prever(arquivo_entrada, arquivo_saida)`
**Propósito:** Função principal que orquestra todo o processo de predição.

**Fluxo de execução:**
1. Carrega dados do Excel
2. Verifica se coluna "Texto Mascarado" existe
3. Carrega modelos treinados (.pkl)
4. Para cada texto:
   - Extrai embeddings BERT (1536 dim)
   - Extrai features TF-IDF (3000 dim)
   - Extrai features de regras (9 dim)
   - Combina tudo (4545 dimensões)
5. Aplica modelo ensemble (voting de 3 modelos)
6. Aplica threshold otimizado (0.50)
7. Salva predições em Excel

---

### 3.3 Técnicas Utilizadas

**1. Ensemble Voting**
- Combina 3 modelos diferentes: Random Forest, Logistic Regression, Gradient Boosting
- Cada modelo "vota" e a decisão final é tomada por maioria ponderada
- Reduz erro e aumenta confiabilidade

**2. Data Augmentation (no treinamento)**
- Gerou variações sintéticas dos textos originais
- Duplicou o dataset de 99 → 191 registros
- Ajudou a evitar overfitting

**3. Threshold Optimization**
- Testou thresholds de 0.1 a 0.9
- Selecionou 0.50 como ótimo para maximizar F1-Score
- Balanceia precisão e recall

**4. Multi-model BERT**
- Usa 2 modelos BERT diferentes
- Captura aspectos complementares da linguagem
- Mais robusto que usar apenas 1 modelo

---

## ❓ 4. Solução de Problemas

### Erro: "ModuleNotFoundError: No module named 'pandas'"
**Solução:** Instale as dependências
```bash
pip install -r requirements.txt
```

---

### Erro: "FileNotFoundError: [Errno 2] No such file or directory: 'modelo_otimizado_maximo.pkl'"
**Solução:** Certifique-se de que todos os arquivos .pkl estão na pasta models
```bash
# Listar arquivos na pasta
dir  # Windows
ls   # Linux/Mac
```
Deve mostrar a estrutura dos arquivos

---

### Aviso: "Some weights of BertForSequenceClassification were not initialized..."
**Solução:** Este é um aviso esperado e pode ser ignorado. Os modelos BERT estão sendo usados apenas para extração de embeddings, não para classificação direta.

---

### Execução muito lenta
**Solução:**
- Normal em CPU (3 a 7 minutos para 100 registros)
- Para acelerar:
  - Use GPU se disponível (10x mais rápido)
  - Ou aguarde a execução completa
  - Modelos BERT são pesados mas precisos

---

### Erro: "RuntimeError: Couldn't load custom C++ ops..."
**Solução:** Pode ser ignorado. Não afeta a funcionalidade.

---

## 📞 6. Suporte

Para questões sobre execução:
1. Verifique esta documentação completa
2. Revise os comentários no código (`predicao_submissao.py`)
3. Execute o teste rápido da seção 4.1

---

## ✅ 7. Checklist de Execução


- [ ] 1. Extrair arquivo ZIP
- [ ] 2. Abrir terminal na pasta extraída
- [ ] 3. Criar ambiente virtual: `python -m venv venv`
- [ ] 4. Ativar ambiente: `venv\Scripts\activate` (Windows) ou `source venv/bin/activate` (Linux/Mac)
- [ ] 5. Instalar dependências: `pip install -r requirements.txt`
- [ ] 6. Executar predição: `python src/predicao_submissao.py data/input/<seu_arquivo_teste.xlsx> data/output/resultado.xlsx`
- [ ] 7. Verificar arquivo `resultado.xlsx` gerado
- [ ] 8. Calcular métricas (Precisão, Recall, F1-Score) com base nas predições

---

## 📄 8. Informações Técnicas Adicionais

**Linguagem:** Python 3.11.9
**Frameworks principais:** scikit-learn, Transformers (HuggingFace), PyTorch  
**Modelos utilizados:** BERTimbau, DistilBERT-PT, Random Forest, Logistic Regression, Gradient Boosting   
**Memória RAM necessária:** 2-4 GB durante execução  

---
---

**Última atualização:** 28/01/2026  
**Versão:** 1.0  
