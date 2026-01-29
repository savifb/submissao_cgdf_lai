"""
HACKATHON CGDF - CATEGORIA ACESSO À INFORMAÇÃO
Script de Predição para Submissão

Este script classifica pedidos de acesso à informação como:
- PÚBLICO (classe 0): Não contém dados pessoais
- NÃO PÚBLICO (classe 1): Contém dados pessoais

A CGDF usará este script para avaliar o modelo no conjunto de controle.
"""

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from scipy.sparse import hstack
import sys
import os
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Caminho base do projeto (pasta onde está este script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'models')


# ============================================================================
# FUNÇÕES DE EXTRAÇÃO DE FEATURES (Mesmas do treinamento)
# ============================================================================

def extrair_features_adicionais(texto):
    """
    Extrai features baseadas em regras para identificar dados pessoais.
    
    Args:
        texto (str): Texto do pedido de acesso à informação
        
    Returns:
        dict: Dicionário com 9 features numéricas
    """
    import re
    texto = str(texto).lower()
    
    features = {}
    
    # Padrões que indicam dados pessoais
    features['tem_cpf'] = 1 if re.search(r'\d{3}\.?\d{3}\.?\d{3}-?\d{2}', texto) else 0
    features['tem_matricula'] = 1 if re.search(r'matr[íi]cula\s*:?\s*\d+', texto) else 0
    features['tem_processo'] = 1 if re.search(r'processo\s*:?\s*\d+', texto) else 0
    features['tem_nome_proprio'] = 1 if any(palavra in texto for palavra in ['meu nome', 'minha', 'meu']) else 0
    
    # Palavras-chave relacionadas a dados pessoais
    palavras_pessoais = ['cadastro', 'prontuário', 'laudo', 'exame', 'atestado', 
                         'ficha', 'dados pessoais', 'meus dados', 'minhas informações',
                         'meu processo', 'minha situação', 'companheiro', 'familiar']
    features['palavras_pessoais'] = sum(1 for p in palavras_pessoais if p in texto)
    
    # Pronomes possessivos (forte indicador de dados pessoais)
    pronomes = ['meu', 'minha', 'meus', 'minhas']
    features['pronomes_possessivos'] = sum(texto.count(p) for p in pronomes)
    
    # Verbos em primeira pessoa
    verbos_primeira_pessoa = ['solicito', 'preciso', 'gostaria', 'quero', 'estou']
    features['verbos_primeira_pessoa'] = sum(1 for v in verbos_primeira_pessoa if v in texto)
    
    # Tamanho do texto
    features['tamanho_texto'] = len(texto)
    features['num_palavras'] = len(texto.split())
    
    return features


def extrair_embeddings_dual_bert(textos):
    """
    Extrai embeddings de 2 modelos BERT diferentes.
    
    Modelos utilizados:
    - BERTimbau (neuralmind/bert-base-portuguese-cased): 768 dimensões
    - DistilBERT PT (adalbertojunior/distilbert-portuguese-cased): 768 dimensões
    Total: 1536 dimensões concatenadas
    
    Args:
        textos (list): Lista de textos a processar
        
    Returns:
        numpy.ndarray: Array com shape (n_textos, 1536)
    """
    print(f"Extraindo embeddings BERT para {len(textos)} textos...")
    
    modelos = [
        "neuralmind/bert-base-portuguese-cased",
        "adalbertojunior/distilbert-portuguese-cased"
    ]
    
    all_embeddings = []
    
    for modelo_nome in modelos:
        tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
        modelo = AutoModel.from_pretrained(modelo_nome)
        modelo.to(device)
        modelo.eval()
        
        embeddings = []
        batch_size = 16
        
        with torch.no_grad():
            for i in range(0, len(textos), batch_size):
                batch = textos[i:i+batch_size]
                inputs = tokenizer(batch, padding=True, truncation=True, 
                                 max_length=96, return_tensors='pt')
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = modelo(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(cls_embeddings)
        
        all_embeddings.append(np.array(embeddings))
    
    combined = np.hstack(all_embeddings)
    print(f"✓ Embeddings extraídos: {combined.shape}")
    return combined


# ============================================================================
# FUNÇÃO PRINCIPAL DE PREDIÇÃO
# ============================================================================

def prever(arquivo_entrada, arquivo_saida, coluna_texto='Texto Mascarado', coluna_id='ID'):
    """
    Realiza predições no conjunto de controle da CGDF.
    
    CLASSIFICAÇÃO:
    - Classe 0 = PÚBLICO (não contém dados pessoais)
    - Classe 1 = NÃO PÚBLICO (contém dados pessoais)
    
    Args:
        arquivo_entrada (str): Caminho do arquivo Excel com os dados de teste
        arquivo_saida (str): Caminho onde salvar as predições
        coluna_texto (str): Nome da coluna com o texto dos pedidos (padrão: 'Texto Mascarado')
        coluna_id (str): Nome da coluna com ID (padrão: 'ID')
    
    Returns:
        pd.DataFrame: DataFrame com as predições
    """
    print("="*70)
    print("PREDIÇÃO - HACKATHON CGDF LAI")
    print("Classificação: PÚBLICO vs. NÃO PÚBLICO")
    print("="*70)
    
    # 1. Carregar dados de teste
    print(f"\n1. Carregando dados: {arquivo_entrada}")
    
    if not os.path.exists(arquivo_entrada):
        print(f"   ✗ ERRO: Arquivo não encontrado: {arquivo_entrada}")
        print(f"   Certifique-se de que o arquivo existe no caminho especificado.")
        return
    
    df_teste = pd.read_excel(arquivo_entrada)
    print(f"   ✓ {len(df_teste)} registros carregados")
    
    # Verificar se coluna existe
    if coluna_texto not in df_teste.columns:
        print(f"   ✗ ERRO: Coluna '{coluna_texto}' não encontrada")
        print(f"   Colunas disponíveis: {df_teste.columns.tolist()}")
        return
    
    # 2. Carregar modelos treinados
    print("\n2. Carregando modelos...")
    
    # Tentar carregar da pasta models/ primeiro, depois da raiz
    caminhos_modelo = [
        os.path.join(MODELS_DIR, 'modelo_otimizado_maximo.pkl'),
        os.path.join(BASE_DIR, '..', 'models', 'modelo_otimizado_maximo.pkl'),
        'modelo_otimizado_maximo.pkl',
        os.path.join(BASE_DIR, '..', 'modelo_otimizado_maximo.pkl')
    ]
    
    caminhos_vectorizer = [
        os.path.join(MODELS_DIR, 'vectorizer_otimizado.pkl'),
        os.path.join(BASE_DIR, '..', 'models', 'vectorizer_otimizado.pkl'),
        'vectorizer_otimizado.pkl',
        os.path.join(BASE_DIR, '..', 'vectorizer_otimizado.pkl')
    ]
    
    caminhos_config = [
        os.path.join(MODELS_DIR, 'config_otimizado.pkl'),
        os.path.join(BASE_DIR, '..', 'models', 'config_otimizado.pkl'),
        'config_otimizado.pkl',
        os.path.join(BASE_DIR, '..', 'config_otimizado.pkl')
    ]
    
    # Carregar modelo
    modelo = None
    for caminho in caminhos_modelo:
        if os.path.exists(caminho):
            modelo = joblib.load(caminho)
            print(f"   ✓ Modelo carregado de: {caminho}")
            break
    
    if modelo is None:
        print(f"   ✗ ERRO: modelo_otimizado_maximo.pkl não encontrado")
        print(f"   Procurado em: {caminhos_modelo[0]}")
        return
    
    # Carregar vectorizer
    vectorizer = None
    for caminho in caminhos_vectorizer:
        if os.path.exists(caminho):
            vectorizer = joblib.load(caminho)
            print(f"   ✓ Vectorizer carregado")
            break
    
    if vectorizer is None:
        print(f"   ✗ ERRO: vectorizer_otimizado.pkl não encontrado")
        return
    
    # Carregar config
    config = None
    for caminho in caminhos_config:
        if os.path.exists(caminho):
            config = joblib.load(caminho)
            print(f"   ✓ Config carregado")
            break
    
    if config is None:
        print(f"   ✗ ERRO: config_otimizado.pkl não encontrado")
        return
    
    threshold = config.get('threshold', 0.5)
    print(f"   ✓ Threshold: {threshold:.2f}")
    
    # 3. Extrair features
    print("\n3. Extraindo features...")
    
    # 3a. Embeddings BERT
    textos = df_teste[coluna_texto].astype(str).tolist()
    embeddings = extrair_embeddings_dual_bert(textos)
    
    # 3b. TF-IDF
    print("   Extraindo TF-IDF...")
    tfidf_features = vectorizer.transform(df_teste[coluna_texto].astype(str))
    
    # 3c. Features tradicionais
    print("   Extraindo features tradicionais...")
    features_trad = pd.DataFrame([extrair_features_adicionais(t) for t in textos])
    
    # 3d. Combinar tudo
    print("   Combinando features...")
    X_final = hstack([tfidf_features, embeddings, features_trad.values])
    print(f"   ✓ {X_final.shape[1]} features combinadas")
    
    # 4. Fazer predições
    print("\n4. Realizando predições...")
    y_proba = modelo.predict_proba(X_final)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    print(f"   ✓ Predições concluídas")
    
    # 5. Preparar resultado
    print("\n5. Preparando arquivo de saída...")
    
    # Criar DataFrame de saída
    df_resultado = pd.DataFrame()
    
    # Adicionar ID se existir
    if coluna_id in df_teste.columns:
        df_resultado[coluna_id] = df_teste[coluna_id]
    else:
        df_resultado['ID'] = range(1, len(df_teste) + 1)
    
    # Adicionar texto original (para facilitar verificação)
    df_resultado['Texto Mascarado'] = df_teste[coluna_texto]
    
    # CLASSIFICAÇÃO NUMÉRICA (0 ou 1)
    df_resultado['Classificação'] = y_pred
    
    # CLASSIFICAÇÃO TEXTUAL (PÚBLICO ou NÃO PÚBLICO)
    df_resultado['Status'] = df_resultado['Classificação'].map({
        0: 'PÚBLICO',
        1: 'NÃO PÚBLICO'
    })
    
    # EXPLICAÇÃO
    df_resultado['Justificativa'] = df_resultado['Classificação'].map({
        0: 'Não contém dados pessoais',
        1: 'Contém dados pessoais'
    })
    
    # Probabilidades e confiança (para análise)
    df_resultado['Probabilidade_Dados_Pessoais'] = y_proba
    df_resultado['Confiança'] = np.maximum(y_proba, 1 - y_proba)
    
    # 6. Salvar resultado
    # Criar diretório de saída se não existir
    output_dir = os.path.dirname(arquivo_saida)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"   ✓ Diretório criado: {output_dir}")
    
    df_resultado.to_excel(arquivo_saida, index=False)
    print(f"   ✓ Resultado salvo: {arquivo_saida}")
    
    # 7. Estatísticas
    print("\n" + "="*70)
    print("ESTATÍSTICAS DAS PREDIÇÕES")
    print("="*70)
    print(f"\nTotal de pedidos classificados: {len(df_resultado)}")
    
    print(f"\n{'Status':<20} {'Quantidade':<12} {'Percentual'}")
    print("-"*50)
    for status in ['PÚBLICO', 'NÃO PÚBLICO']:
        qtd = (df_resultado['Status'] == status).sum()
        pct = (qtd / len(df_resultado)) * 100
        print(f"{status:<20} {qtd:<12} {pct:>6.1f}%")
    
    print(f"\nConfiança média das predições: {df_resultado['Confiança'].mean():.2%}")
    
    alta_confianca = (df_resultado['Confiança'] > 0.8).sum()
    print(f"Predições com alta confiança (>80%): {alta_confianca}/{len(df_resultado)} ({alta_confianca/len(df_resultado):.1%})")
    
    print("\n" + "="*70)
    print("LEGENDA:")
    print("  Classificação 0 = PÚBLICO = Não contém dados pessoais")
    print("  Classificação 1 = NÃO PÚBLICO = Contém dados pessoais")
    print("="*70)
    
    print("\n✅ PREDIÇÃO CONCLUÍDA COM SUCESSO!")
    print(f"   Arquivo gerado: {arquivo_saida}")
    print("="*70)
    
    return df_resultado


# ============================================================================
# INTERFACE DE LINHA DE COMANDO
# ============================================================================

def main():
    """
    Ponto de entrada do script.
    Aceita argumentos de linha de comando para facilitar automação.
    """
    if len(sys.argv) < 2:
        print("="*70)
        print("HACKATHON CGDF - SCRIPT DE PREDIÇÃO")
        print("Classificação: PÚBLICO vs. NÃO PÚBLICO")
        print("="*70)
        print("\n📋 USO RECOMENDADO:")
        print("  python src/predicao_submissao.py data/input/<arquivo.xlsx> data/output/<resultado.xlsx>")
        print("\n📝 EXEMPLO:")
        print("  python src/predicao_submissao.py data/input/teste_cgdf.xlsx data/output/resultado.xlsx")
        print("\n💡 USO ALTERNATIVO (caminhos personalizados):")
        print("  python src/predicao_submissao.py <caminho/entrada.xlsx> <caminho/saida.xlsx>")
        print("\n⚠️  OBSERVAÇÕES:")
        print("  - Se não especificar arquivo de saída, será 'predicoes.xlsx'")
        print("  - O arquivo de entrada deve ter a coluna 'Texto Mascarado'")
        print("  - Recomenda-se colocar arquivo de teste em data/input/")
        print("  - Os resultados serão salvos em data/output/")
        print("\n📊 CLASSIFICAÇÃO:")
        print("  0 = PÚBLICO (não contém dados pessoais)")
        print("  1 = NÃO PÚBLICO (contém dados pessoais)")
        print("="*70)
        sys.exit(1)
    
    arquivo_entrada = sys.argv[1]
    arquivo_saida = sys.argv[2] if len(sys.argv) > 2 else 'predicoes.xlsx'
    
    prever(arquivo_entrada, arquivo_saida)


if __name__ == "__main__":
    main()