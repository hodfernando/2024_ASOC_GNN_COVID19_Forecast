import os
import glob
import shutil
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix


# --- FUNÇÃO 1: BUSCAR ARQUIVOS DENTRO DE UMA PASTA DE EXPERIMENTO ---
def buscar_arquivos_de_resultado(caminho_experimento):
    if not os.path.isdir(caminho_experimento):
        return None, None
    files = os.listdir(caminho_experimento)
    y_real_files, y_pred_files = [], []
    y_real_files_no_norm = sorted([os.path.join(caminho_experimento, f) for f in files if
                                   f.startswith('y_real_no_norm_rep_') and f.endswith('.npy')])
    y_pred_files_no_norm = sorted([os.path.join(caminho_experimento, f) for f in files if
                                   f.startswith('y_pred_no_norm_rep_') and f.endswith('.npy')])
    if y_real_files_no_norm and y_pred_files_no_norm:
        y_real_files, y_pred_files = y_real_files_no_norm, y_pred_files_no_norm
    else:
        y_real_files_std = sorted([os.path.join(caminho_experimento, f) for f in files if
                                   f.startswith('y_real_rep_') and '_no_norm_' not in f and f.endswith('.npy')])
        y_pred_files_std = sorted([os.path.join(caminho_experimento, f) for f in files if
                                   f.startswith('y_pred_rep_') and '_no_norm_' not in f and f.endswith('.npy')])
        if y_real_files_std and y_pred_files_std:
            y_real_files, y_pred_files = y_real_files_std, y_pred_files_std
    if y_real_files and y_pred_files:
        return y_real_files, y_pred_files
    return None, None


# --- FUNÇÃO 2A: CALCULAR MÉTRICAS DE REGRESSÃO ---
def calcular_metricas_regressao(y_real, y_pred):
    num_reps, _, num_cidades, _ = y_real.shape
    rmse_reps, mae_reps, r2_reps = np.zeros(num_reps), np.zeros(num_reps), np.zeros(num_reps)
    for i in range(num_reps):
        real_flat, pred_flat = y_real[i].flatten(), y_pred[i].flatten()
        rmse_reps[i] = np.sqrt(mean_squared_error(real_flat, pred_flat))
        mae_reps[i] = mean_absolute_error(real_flat, pred_flat)
        r2_reps[i] = r2_score(real_flat, pred_flat)
    return {'rmse': rmse_reps, 'mae': mae_reps, 'r2': r2_reps}


# --- FUNÇÃO 2B: NOVA FUNÇÃO PARA CALCULAR MÉTRICAS DE CLASSIFICAÇÃO ---
def calcular_metricas_classificacao(y_real, y_pred):
    """
    Calcula Acurácia, Precision, Recall e F1-Score a partir dos dados de classificação.
    """
    num_reps = y_real.shape[0]
    accuracies, precisions, recalls, f1s = [], [], [], []

    for i in range(num_reps):
        y_true_flat = y_real[i].flatten().astype(int)
        y_pred_flat = y_pred[i].flatten().astype(int)

        # Matriz de confusão: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1]).ravel()

        # Acurácia
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracies.append(accuracy)

        # Precision (evita divisão por zero)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precisions.append(precision)

        # Recall (evita divisão por zero)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recalls.append(recall)

        # F1-Score (evita divisão por zero)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(f1)

    return {
        'accuracy': np.array(accuracies),
        'precision': np.array(precisions),
        'recall': np.array(recalls),
        'f1_score': np.array(f1s)
    }


# --- FUNÇÃO 3: GERAR TABELAS ---
# Esta função continua útil para gerar as tabelas individuais de regressão.
def gerar_tabelas(metricas, caminho_saida, num_cidades):
    df_geral = pd.DataFrame({'Metrica': ['RMSE', 'MAE', 'R2'],
                             'Media': [metricas['rmse'].mean(), metricas['mae'].mean(), metricas['r2'].mean()],
                             'DesvioPadrao': [metricas['rmse'].std(), metricas['mae'].std(), metricas['r2'].std()],
                             'Min': [metricas['rmse'].min(), metricas['mae'].min(), metricas['r2'].min()],
                             'Max': [metricas['rmse'].max(), metricas['mae'].max(), metricas['r2'].max()]})
    df_geral.to_csv(os.path.join(caminho_saida, 'metrics_general_regression.csv'), index=False)


# --- FUNÇÃO 4A: GERAR GRÁFICOS DE REGRESSÃO ---
def gerar_graficos_regressao(df_metricas, caminho_saida):
    plt.style.use('seaborn-v0_8-whitegrid')
    metricas_plot = ['RMSE', 'MAE', 'R2']
    for metrica in metricas_plot:
        plt.figure(figsize=(14, 8))
        sns.barplot(data=df_metricas, x='Identificador', y=metrica, hue='Identificador', palette='viridis',
                    legend=False)
        plt.title(f'Comparação de {metrica}', fontsize=16, weight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(caminho_saida, f'comparacao_{metrica}.png'), dpi=300)
        plt.close()
    print(f"Gráficos de regressão salvos em: {caminho_saida}")


# --- FUNÇÃO 4B: NOVA FUNÇÃO PARA GERAR HEATMAPS DE CLASSIFICAÇÃO ---
def gerar_heatmaps_classificacao(df_resultados, caminho_saida):
    """
    Gera heatmaps 14x14 (lags vs outs) para cada modelo e cada métrica de classificação.
    """
    print("--- Gerando heatmaps para métricas de classificação ---")
    metricas_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
    modelos = df_resultados['model_name'].unique()

    for model_name in modelos:
        df_model = df_resultados[df_resultados['model_name'] == model_name]
        for metrica in metricas_plot:
            try:
                # Cria a matriz 14x14 para o heatmap
                pivot_df = df_model.pivot(index='lags', columns='out', values=metrica)

                plt.figure(figsize=(12, 10))
                sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="viridis", linewidths=.5)
                plt.title(f'Heatmap de {metrica} para o Modelo: {model_name}', fontsize=16, weight='bold')
                plt.xlabel('Horizonte de Previsão (Out)', fontsize=12)
                plt.ylabel('Janela de Features (Lags)', fontsize=12)

                nome_arquivo = f'heatmap_{model_name}_{metrica}.png'
                caminho_arquivo_salvo = os.path.join(caminho_saida, nome_arquivo)
                plt.savefig(caminho_arquivo_salvo, dpi=300)
                plt.close()
                print(f"Heatmap salvo em: {caminho_arquivo_salvo}")
            except Exception as e:
                print(f"  ERRO ao gerar heatmap para {model_name} - {metrica}: {e}")


# --- FUNÇÃO PRINCIPAL ---
def analisar_experimentos():
    PAISES = ['China', 'Brazil']
    TIPOS_DE_GRAFO = ['grafo_original']
    TASK_TYPES = ['classification', 'regression']
    ALPHA = 0.01
    USA_BACKBONE = True
    lags, outs = 14, 14
    model_names = ['GCRN', 'GCLSTM', 'LSTM']
    k_list, hc_list = [1], [256]

    try:
        pasta_atual = os.path.dirname(os.path.realpath(__file__))
        pasta_projeto = os.path.dirname(pasta_atual)
    except NameError:
        pasta_projeto = os.getcwd()
        print(f"AVISO: __file__ não definido. Usando pasta de trabalho atual: {pasta_projeto}")

    if pasta_projeto.startswith('/media/work/'):
        pasta_projeto = pasta_projeto.replace('/media/work/', '/media/data/', 1)
        print(f"AVISO: Caminho do projeto modificado para usar '/media/data/'.")

    for pais in PAISES:
        for tipo_grafo in TIPOS_DE_GRAFO:
            for task_type in TASK_TYPES:
                print("\n" + "#" * 80)
                print(f"# INICIANDO ANÁLISE PARA: País='{pais}', Tarefa='{task_type}', Grafo='{tipo_grafo}'")
                print("#" * 80)

                # --- Definição de Pastas ---
                subpasta_resultado = tipo_grafo
                if USA_BACKBONE:
                    subpasta_resultado = os.path.join(tipo_grafo, f'backbone_alpha_{ALPHA * 100:.0f}')

                pasta_base_results = os.path.join(pasta_projeto, 'results_daily', pais, task_type, subpasta_resultado)
                pasta_analise_geral = os.path.join(pasta_projeto, 'results_daily', 'results_analysis', pais, task_type,
                                                   tipo_grafo)
                pasta_intermediaria = os.path.join(pasta_analise_geral, 'intermediate_csvs')
                os.makedirs(pasta_intermediaria, exist_ok=True)

                # --- FASE 1: Processamento em Lotes ---
                print("\n--- FASE 1: Processando experimentos em lotes ---")

                for lag, out in product(range(1, lags + 1), range(1, outs + 1)):
                    metricas_lote_atual = []
                    print(f"  Processando lote: lag={lag}, out={out}")

                    for k, hc, model_name in product(k_list, hc_list, model_names):
                        caminho_experimento = os.path.join(pasta_base_results, f'k_{k}_hc_{hc}', model_name,
                                                           f'lags_{lag}_out_{out}')
                        y_real_paths, y_pred_paths = buscar_arquivos_de_resultado(caminho_experimento)

                        if not (y_real_paths and y_pred_paths):
                            continue

                        try:
                            lista_y_real = [np.load(p) for p in y_real_paths]
                            lista_y_pred = [np.load(p) for p in y_pred_paths]
                            y_real_completo = np.stack(lista_y_real, axis=0)
                            y_pred_completo = np.stack(lista_y_pred, axis=0)

                            # =======================================================
                            # INÍCIO DA LÓGICA CONDICIONAL PARA MÉTRICAS
                            # =======================================================
                            if task_type == 'classification':
                                metricas = calcular_metricas_classificacao(y_real_completo, y_pred_completo)
                                resultado = {
                                    'model_name': model_name, 'lags': lag, 'out': out,
                                    'Accuracy': metricas['accuracy'].mean(),
                                    'Precision': metricas['precision'].mean(),
                                    'Recall': metricas['recall'].mean(),
                                    'F1_Score': metricas['f1_score'].mean()
                                }
                            else:  # Regression
                                metricas = calcular_metricas_regressao(y_real_completo, y_pred_completo)
                                identificador = f"{model_name}_k{k}_h{hc}_lags{lag}_out{out}_alpha{ALPHA}"
                                resultado = {
                                    'Identificador': identificador, 'tipo_grafo': tipo_grafo,
                                    'RMSE': metricas['rmse'].mean(),
                                    'MAE': metricas['mae'].mean(),
                                    'R2': metricas['r2'].mean()
                                }
                                # Gera tabelas individuais apenas para regressão
                                pasta_analise_exp = os.path.join(caminho_experimento, 'analysis_results')
                                os.makedirs(pasta_analise_exp, exist_ok=True)
                                gerar_tabelas(metricas, pasta_analise_exp, y_real_completo.shape[2])
                            # =======================================================
                            # FIM DA LÓGICA CONDICIONAL
                            # =======================================================

                            metricas_lote_atual.append(resultado)

                        except Exception as e:
                            print(f"    ERRO ao processar {caminho_experimento}: {e}")

                    if metricas_lote_atual:
                        df_lote = pd.DataFrame(metricas_lote_atual)
                        df_lote.to_csv(os.path.join(pasta_intermediaria, f'lag_{lag}_out_{out}.csv'), index=False)

                # --- FASE 2: Agregação Final ---
                print("\n--- FASE 2: Agregando resultados intermediários ---")
                arquivos_csv = glob.glob(os.path.join(pasta_intermediaria, '*.csv'))
                if not arquivos_csv:
                    print("Nenhum resultado intermediário foi gerado. Pulando.")
                    continue

                df_metricas_finais = pd.concat([pd.read_csv(f) for f in arquivos_csv], ignore_index=True)

                caminho_csv_sumario = os.path.join(pasta_analise_geral, 'all_models_metrics_summary.csv')
                df_metricas_finais.to_csv(caminho_csv_sumario, index=False)
                print(f"Sumário de métricas finais salvo em: {caminho_csv_sumario}")

                # =======================================================
                # LÓGICA CONDICIONAL PARA PLOTS
                # =======================================================
                if task_type == 'classification':
                    gerar_heatmaps_classificacao(df_metricas_finais, pasta_analise_geral)
                else:  # Regression
                    gerar_graficos_regressao(df_metricas_finais, pasta_analise_geral)
                # =======================================================

                try:
                    shutil.rmtree(pasta_intermediaria)
                    print(f"Pasta intermediária '{pasta_intermediaria}' removida.")
                except OSError as e:
                    print(f"Erro ao remover pasta intermediária: {e}")

    print("\n" + "#" * 80)
    print("Análise de todas as combinações concluída com sucesso!")
    print("#" * 80)


if __name__ == "__main__":
    analisar_experimentos()
