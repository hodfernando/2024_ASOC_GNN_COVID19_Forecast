import os
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Funções de Cálculo de Métricas de Erro
# --------------------------------------------------------------------------- #

def calculate_metrics(y_real, y_pred):
    """
    Calcula métricas de erro consolidadas (RMSE e MAE) e suas estatísticas
    (média, desvio padrão, mínimo, máximo) a partir dos valores reais e previstos.

    A média do RMSE/MAE é calculada sobre as médias dos resultados por repetição e por saída.
    As estatísticas (std, min, max) são calculadas sobre os valores de RMSE/MAE de cada repetição.

    Args:
        y_real (np.array): Array numpy com os valores reais.
                           Shape esperado: (num_repeticoes, num_amostras, num_nos, num_saidas_previstas)
        y_pred (np.array): Array numpy com os valores previstos.
                           Shape esperado: (num_repeticoes, num_amostras, num_nos, num_saidas_previstas)

    Returns:
        dict: Dicionário contendo as métricas:
              RMSE_mean, RMSE_std, RMSE_min, RMSE_max,
              MAE_mean, MAE_std, MAE_min, MAE_max.
              Retorna NaNs se os arrays de entrada estiverem vazios.
    """
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)

    # Retorna NaN para todas as métricas se os dados de entrada estiverem vazios
    if y_real.size == 0 or y_pred.size == 0:
        return {'RMSE_mean': np.nan, 'RMSE_std': np.nan, 'RMSE_min': np.nan, 'RMSE_max': np.nan,
                'MAE_mean': np.nan, 'MAE_std': np.nan, 'MAE_min': np.nan, 'MAE_max': np.nan}

    # Calcula RMSE e MAE médios globais:
    # 1. Média sobre as repetições (axis 0)
    # 2. Média sobre as saídas previstas (axis 3 do original, torna-se axis 2 após a média das repetições)
    # Resulta em arrays (num_amostras, num_nos) que são achatados para o cálculo do erro.
    y_real_avg_over_reps_outputs = y_real.mean(axis=(0, 3))
    y_pred_avg_over_reps_outputs = y_pred.mean(axis=(0, 3))

    rmse_overall = np.sqrt(
        mean_squared_error(y_real_avg_over_reps_outputs.flatten(), y_pred_avg_over_reps_outputs.flatten()))
    mae_overall = mean_absolute_error(y_real_avg_over_reps_outputs.flatten(), y_pred_avg_over_reps_outputs.flatten())

    # Calcula RMSE e MAE para cada repetição individualmente
    rmse_values_per_rep, mae_values_per_rep = [], []
    for rep_idx in range(y_real.shape[0]):  # Itera sobre as repetições
        current_y_real_rep = y_real[rep_idx]
        current_y_pred_rep = y_pred[rep_idx]
        # Achatamento de todas as dimensões (amostras, nós, saídas) para calcular o erro da repetição
        rmse_values_per_rep.append(
            np.sqrt(mean_squared_error(current_y_real_rep.flatten(), current_y_pred_rep.flatten())))
        mae_values_per_rep.append(mean_absolute_error(current_y_real_rep.flatten(), current_y_pred_rep.flatten()))

    if not rmse_values_per_rep:  # Segurança, caso o loop não execute (improvável se y_real.size > 0)
        rmse_values_per_rep, mae_values_per_rep = [np.nan], [np.nan]

    # Monta o dicionário de métricas
    metrics = {'RMSE_mean': rmse_overall, 'RMSE_std': np.std(rmse_values_per_rep),
               'RMSE_min': np.min(rmse_values_per_rep), 'RMSE_max': np.max(rmse_values_per_rep),
               'MAE_mean': mae_overall, 'MAE_std': np.std(mae_values_per_rep),
               'MAE_min': np.min(mae_values_per_rep), 'MAE_max': np.max(mae_values_per_rep)}
    return metrics


def calculate_metrics_per_node(y_real_concat, y_pred_concat):
    """
    Calcula RMSE e MAE para cada nó/cidade individualmente.

    Os valores de erro são calculados considerando todas as repetições, amostras e saídas
    para um nó específico.

    Args:
        y_real_concat (np.array): Array numpy com os valores reais.
                                  Shape: (num_repeticoes, num_amostras, num_nos, num_saidas_previstas)
        y_pred_concat (np.array): Array numpy com os valores previstos.
                                  Shape: (num_repeticoes, num_amostras, num_nos, num_saidas_previstas)

    Returns:
        list: Lista de dicionários, onde cada dicionário contém 'node_id', 'RMSE' e 'MAE'
              para um nó. Retorna lista vazia se os arrays de entrada estiverem vazios.
    """
    y_real_concat, y_pred_concat = np.array(y_real_concat), np.array(y_pred_concat)
    if y_real_concat.size == 0 or y_pred_concat.size == 0:
        return []

    num_nodes = y_real_concat.shape[2]  # Dimensão dos nós
    metrics_per_node_list = []

    for node_idx in range(num_nodes):
        # Seleciona os dados para o nó atual, achatando todas as outras dimensões (repetições, amostras, saídas)
        y_real_node_flat = y_real_concat[:, :, node_idx, :].reshape(-1)
        y_pred_node_flat = y_pred_concat[:, :, node_idx, :].reshape(-1)

        rmse_node, mae_node = np.nan, np.nan
        if y_real_node_flat.size > 0 and y_pred_node_flat.size > 0:
            rmse_node = np.sqrt(mean_squared_error(y_real_node_flat, y_pred_node_flat))
            mae_node = mean_absolute_error(y_real_node_flat, y_pred_node_flat)
        metrics_per_node_list.append({'node_id': node_idx, 'RMSE': rmse_node, 'MAE': mae_node})
    return metrics_per_node_list


# --------------------------------------------------------------------------- #
# Funções de Análise Estatística e Plotagem
# --------------------------------------------------------------------------- #

def general_analysis_with_custom_cd_plot(results_df, metric_col, metric_name, analysis_folder, model_names_ordered,
                                         current_lag, current_out):
    """
    Realiza análise estatística geral (Testes de Friedman e Nemenyi) para uma métrica específica
    e gera um Diagrama de Diferença Crítica (CD Plot).

    A análise é feita para uma combinação específica de 'lag' e 'out'.
    Os 'blocos' para o teste de Friedman são as diferentes configurações de hiperparâmetros (K, hidden_channel).

    Args:
        results_df (pd.DataFrame): DataFrame contendo os resultados dos modelos para UMA combinação lag/out.
                                   Deve incluir colunas 'K', 'hidden_channel', 'model', e a 'metric_col'.
        metric_col (str): Nome da coluna no DataFrame que contém a métrica a ser analisada (ex: 'RMSE_mean').
        metric_name (str): Nome legível da métrica para títulos e nomes de arquivo (ex: 'RMSE').
        analysis_folder (str): Caminho da pasta para salvar os resultados da análise (CSV do Nemenyi, PDF do CD plot).
        model_names_ordered (list): Lista com os nomes dos modelos na ordem desejada para algumas saídas.
        current_lag (int): Valor do 'lag' para o qual a análise está sendo feita.
        current_out (int): Valor do 'out' (horizonte de previsão) para o qual a análise está sendo feita.
    """
    print(f"\n--- ANÁLISE ESTATÍSTICA GERAL PARA {metric_name.upper()} (Lag: {current_lag}, Out: {current_out}) ---")

    if results_df.empty:
        print(
            f" DataFrame de resultados vazio para Lag: {current_lag}, Out: {current_out}. Análise para {metric_name} não realizada.")
        return

    if metric_col not in results_df.columns:
        print(f"  Coluna da métrica '{metric_col}' não encontrada no DataFrame de resultados gerais.")
        return

    # Cria uma coluna 'block_id_for_pivot' combinando K e hidden_channel.
    # Esta coluna será usada como índice (blocos) na tabela pivotada para o teste de Friedman.
    # É importante que results_df seja uma cópia se for modificado aqui,
    # o que é garantido pela chamada .copy() em main().
    if 'K' in results_df.columns and 'hidden_channel' in results_df.columns:
        results_df['block_id_for_pivot'] = "K" + results_df['K'].astype(str) + "_hc" + results_df[
            'hidden_channel'].astype(str)
        pivot_geral = results_df.pivot_table(index='block_id_for_pivot', columns='model', values=metric_col)
    else:  # Fallback caso K ou hidden_channel não estejam presentes
        pivot_geral = results_df.pivot_table(index='param_config', columns='model', values=metric_col)

    pivot_geral_cleaned = pivot_geral.dropna()  # Remove linhas (blocos) com NaN para qualquer modelo

    # Verifica se há dados suficientes para o teste de Friedman
    if pivot_geral_cleaned.empty or len(pivot_geral_cleaned) < 2 or len(pivot_geral_cleaned.columns) < 2:
        print(
            f"  Lag {current_lag}, Out {current_out}, {metric_name}: Dados insuficientes para Friedman (após dropna: {len(pivot_geral_cleaned)} blocos, {len(pivot_geral_cleaned.columns)} modelos).")
        return

    # Garante que os modelos no pivot estejam na ordem desejada e que todos os modelos esperados estejam presentes
    actual_models_in_pivot = [col for col in model_names_ordered if col in pivot_geral_cleaned.columns]
    if len(actual_models_in_pivot) < 2:
        print(
            f"  Lag {current_lag}, Out {current_out}, {metric_name}: Menos de dois modelos no pivot. Modelos: {actual_models_in_pivot}")
        return
    pivot_geral_cleaned = pivot_geral_cleaned[actual_models_in_pivot]

    # Teste de Friedman
    stat, p_value = np.nan, 1.0  # Inicializa p_value para 1.0 (sem significância)
    try:
        # friedmanchisquare requer que os argumentos sejam arrays separados por grupo (modelo)
        stat, p_value = friedmanchisquare(*[pivot_geral_cleaned[col] for col in actual_models_in_pivot])
        print(
            f"  Teste de Friedman ({metric_name}, L{current_lag} O{current_out}): Estatística={stat:.4f}, p-valor={p_value:.4g}")
    except ValueError as e:
        print(f"  Erro Friedman ({metric_name}, L{current_lag} O{current_out}): {e}. Teste não realizado.")
        # p_value permanece 1.0, Nemenyi não será executado, que é o comportamento desejado.

    # Calcula Ranks Médios para todos os casos (mesmo se Friedman não for significativo)
    ranks_geral = pivot_geral_cleaned.rank(axis=1, method='average', ascending=True)  # Menor erro = menor rank (melhor)
    mean_ranks_geral = ranks_geral.mean().sort_values()  # Ordena do melhor para o pior rank
    print(f"  Ranks Médios ({metric_name}, L{current_lag} O{current_out}):")
    print(mean_ranks_geral)

    # Teste Post-hoc de Nemenyi (apenas se Friedman for significativo)
    nemenyi_results = None
    if not pd.isna(p_value) and p_value < 0.05:
        try:
            nemenyi_data = pivot_geral_cleaned.to_numpy()
            # Nemenyi precisa de pelo menos 2 grupos (modelos) e 2 blocos (configurações K_hc)
            if nemenyi_data.shape[0] > 1 and nemenyi_data.shape[1] > 1:
                nemenyi_results = sp.posthoc_nemenyi_friedman(nemenyi_data)
                nemenyi_results.columns = actual_models_in_pivot
                nemenyi_results.index = actual_models_in_pivot
                print(f"\n  Resultados Nemenyi ({metric_name}, L{current_lag} O{current_out} - p-valores):")
                print(nemenyi_results.round(4))
                # Salva a matriz de p-valores do Nemenyi
                nemenyi_filename = os.path.join(analysis_folder,
                                                f"nemenyi_geral_lag{current_lag}_out{current_out}_{metric_name.lower()}.csv")
                nemenyi_results.to_csv(nemenyi_filename)
            else:
                print(
                    f"  Dados insuficientes para Nemenyi ({metric_name}, L{current_lag} O{current_out}) após limpeza: {nemenyi_data.shape}")
                nemenyi_results = None  # Garante que não tentará usar resultados parciais do Nemenyi
        except Exception as e:
            print(f"  Erro Nemenyi ({metric_name}, L{current_lag} O{current_out}): {e}")
            nemenyi_results = None  # Garante que não tentará usar resultados parciais do Nemenyi
    elif not pd.isna(p_value):  # Friedman foi bem-sucedido, mas não significativo
        print(
            f"  Friedman não significativo (p={p_value:.3g}), Nemenyi não aplicado para {metric_name} (L{current_lag} O{current_out}).")
    # Se p_value for NaN (erro no Friedman), este bloco é pulado e nemenyi_results permanece None.

    # Plotagem Customizada do Diagrama de Diferença Crítica (CD Plot)
    fig_cd = None
    try:
        text_offsets = {model: 0.3 for model in actual_models_in_pivot}  # Offset vertical para nomes dos modelos
        fig_cd, ax = plt.subplots(figsize=(8, max(2.5, len(actual_models_in_pivot) * 0.6)), facecolor='white')

        # Plota os pontos dos ranks médios dos modelos
        y_positions = np.zeros(len(mean_ranks_geral))  # Todos os modelos na mesma linha y=0
        ax.scatter(mean_ranks_geral.values, y_positions, s=120, color='black',
                   zorder=3)  # zorder para ficar na frente das linhas

        # Adiciona nomes dos modelos acima dos pontos
        for model, rank_val in mean_ranks_geral.items():
            offset_y = text_offsets.get(model, 0.3)
            ax.text(rank_val, offset_y, model, ha='center', va='bottom', fontsize=10)

        # Configurações visuais do gráfico
        ax.set_ylim(-0.8, 1.2)  # Limites do eixo Y para dar espaço às linhas de conexão
        min_rank_display = mean_ranks_geral.min() - 0.7
        max_rank_display = mean_ranks_geral.max() + 0.7
        ax.set_xlim(min_rank_display, max_rank_display)  # Limites do eixo X para dar espaço aos nomes

        ax.set_yticks([])  # Remove marcações do eixo Y
        ax.set_xlabel(f"Rank Médio de {metric_name} (menor é melhor)", fontsize=10)
        ax.set_title(f"Diagrama CD - {metric_name} Geral (Lag: {current_lag}, Out: {current_out})", fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.7)  # Grade vertical suave

        line_threshold = 0.05  # Nível de significância para Nemenyi (p-valor)

        # Desenha linhas conectando modelos que NÃO são estatisticamente diferentes,
        # apenas se o teste de Nemenyi foi realizado e o Friedman foi significativo.
        if nemenyi_results is not None and not pd.isna(p_value) and p_value < 0.05:
            line_drawing_y_offset = -0.15  # Posição Y inicial para a primeira linha de conexão
            line_drawing_step = -0.08  # Espaçamento vertical entre linhas de conexão

            sorted_models = mean_ranks_geral.index.tolist()  # Modelos ordenados pelo rank médio
            num_models = len(sorted_models)

            for i in range(num_models):
                for j in range(i + 1, num_models):
                    model_i = sorted_models[i]
                    model_j = sorted_models[j]

                    # Verifica se os modelos estão presentes nos resultados do Nemenyi
                    if model_i not in actual_models_in_pivot or model_j not in actual_models_in_pivot:
                        continue

                    pval_ij = np.nan  # Inicializa p-valor como NaN
                    try:
                        # Acessa o p-valor da matriz Nemenyi (que é simétrica)
                        pval_ij = nemenyi_results.loc[model_i, model_j]
                    except KeyError:
                        # Fallback caso o acesso direto falhe (ex: se a matriz for triangular)
                        try:
                            pval_ij = nemenyi_results.loc[model_j, model_i]
                        except KeyError:
                            # O par de modelos não foi encontrado na matriz Nemenyi.
                            # Isso pode acontecer se algum modelo foi filtrado ou não participou.
                            # print(f"  Aviso CD Plot: Par de p-valor ({model_i}, {model_j}) não encontrado na matriz Nemenyi.")
                            pass  # pval_ij permanece NaN, a linha não será desenhada
                    except Exception as e_access:  # Captura outros erros de acesso
                        print(f"  Erro acessando p-valor Nemenyi para ({model_i}, {model_j}): {e_access}")
                        pass  # pval_ij permanece NaN

                    # Desenha a linha se o p-valor não for NaN e for maior ou igual ao limiar (não há diferença significativa)
                    if not pd.isna(pval_ij) and pval_ij >= line_threshold:
                        start_rank_val = mean_ranks_geral[model_i]
                        end_rank_val = mean_ranks_geral[model_j]

                        ax.plot([start_rank_val, end_rank_val],
                                [line_drawing_y_offset, line_drawing_y_offset],
                                color='black', lw=1.5)

                        # Move a posição Y para a próxima linha para evitar sobreposição total
                        line_drawing_y_offset += line_drawing_step
                        if line_drawing_y_offset < -0.75:  # Reseta a posição Y se descer muito
                            line_drawing_y_offset = -0.15 - np.random.rand() * 0.02  # Adiciona pequeno jitter no reset
        else:  # Nemenyi não foi executado ou Friedman não foi significativo
            friedman_p_text = f"p={p_value:.3g}" if not pd.isna(p_value) else "não calculado"
            print(
                f"  CD Plot ({metric_name}, L{current_lag} O{current_out}): Nenhuma conexão Nemenyi para desenhar (Friedman {friedman_p_text}). Mostrando apenas ranks.")

        # Adiciona legenda explicativa abaixo do gráfico
        caption_y_pos = -0.15 if fig_cd.get_figheight() < 5 else -0.1  # Ajusta posição do caption
        friedman_p_for_caption = f"{p_value:.3g}" if not pd.isna(p_value) else "N/A"
        plt.figtext(0.5, caption_y_pos,
                    f"Modelos conectados por linha não são estatisticamente diferentes (Nemenyi, p ≥ {line_threshold}).\nFriedman p-valor: {friedman_p_for_caption}",
                    ha='center', fontsize=8)

        # Salva a imagem do CD Plot em formato PDF
        pdf_filename = os.path.join(analysis_folder,
                                    f"cd_plot_geral_lag{current_lag}_out{current_out}_{metric_name.lower()}.pdf")
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Ajusta layout para dar espaço ao caption
        plt.savefig(pdf_filename, bbox_inches='tight', facecolor='white')
        print(f"  ✅ Imagem CD Plot PDF ({metric_name}, L{current_lag} O{current_out}) salva em: {pdf_filename}")

    except Exception as e_plot:
        print(f"  Erro ao gerar CD Diagram para {metric_name} (L{current_lag} O{current_out}): {e_plot}")
    finally:
        if fig_cd is not None:  # Fecha a figura para liberar memória
            plt.close(fig_cd)


def per_city_analysis_tabular(results_per_node_df, analysis_folder, model_names_ordered, current_lag, current_out):
    """
    Realiza análise estatística por nó/cidade (Testes de Friedman e Nemenyi)
    para as métricas RMSE e MAE.

    A análise é feita para uma combinação específica de 'lag' e 'out'.
    Retorna um DataFrame com o sumário dos resultados estatísticos por cidade.

    Args:
        results_per_node_df (pd.DataFrame): DataFrame com resultados por nó para UMA combinação lag/out.
                                            Deve incluir 'node_id', 'K', 'hidden_channel', 'model', 'RMSE', 'MAE'.
        analysis_folder (str): Caminho da pasta para (potencialmente) salvar resultados detalhados (não usado atualmente para salvar CSVs nesta função).
        model_names_ordered (list): Lista com os nomes dos modelos na ordem desejada.
        current_lag (int): Valor do 'lag' da análise.
        current_out (int): Valor do 'out' (horizonte) da análise.

    Returns:
        pd.DataFrame: DataFrame contendo o sumário da análise estatística por cidade
                      (node_id, metric, friedman_p_value, best_model_by_rank, significant_diff_found).
                      Retorna DataFrame vazio se não houver dados.
    """
    print(f"\n--- ANÁLISE ESTATÍSTICA POR CIDADE/NÓ (Lag: {current_lag}, Out: {current_out}) ---")
    if results_per_node_df.empty:
        print(f"  DataFrame de resultados por nó vazio para Lag: {current_lag}, Out: {current_out}.")
        return pd.DataFrame()  # Retorna DataFrame vazio

    unique_nodes = sorted(results_per_node_df['node_id'].unique())
    city_summary_list = []  # Lista para coletar os sumários de cada cidade

    for node_id in unique_nodes:
        # Filtra os dados para o nó/cidade atual.
        # .copy() é usado para evitar SettingWithCopyWarning ao adicionar 'block_id_for_pivot'.
        city_df = results_per_node_df[results_per_node_df['node_id'] == node_id].copy()

        for metric_col_node, metric_name_node in [('RMSE', 'RMSE'), ('MAE', 'MAE')]:
            if metric_col_node not in city_df.columns:
                continue  # Pula se a coluna da métrica não existir

            # Cria 'block_id_for_pivot' para usar como blocos no teste de Friedman
            if 'K' in city_df.columns and 'hidden_channel' in city_df.columns:
                city_df.loc[:, 'block_id_for_pivot'] = "K" + city_df['K'].astype(str) + "_hc" + city_df[
                    'hidden_channel'].astype(str)
                pivot_city = city_df.pivot_table(index='block_id_for_pivot', columns='model', values=metric_col_node)
            else:  # Fallback
                pivot_city = city_df.pivot_table(index='param_config', columns='model', values=metric_col_node)

            pivot_city_cleaned = pivot_city.dropna()

            # Dados comuns para o dicionário de sumário, caso os testes não possam ser realizados
            common_data_for_summary = {'lag': current_lag, 'out': current_out, 'node_id': node_id,
                                       'metric': metric_name_node,
                                       'friedman_p_value': np.nan, 'best_model_by_rank': 'N/A',
                                       'significant_diff_found': False}

            if pivot_city_cleaned.empty or len(pivot_city_cleaned) < 2 or len(pivot_city_cleaned.columns) < 2:
                city_summary_list.append(common_data_for_summary)
                continue  # Próxima métrica ou cidade

            actual_models_in_city_pivot = [col for col in model_names_ordered if col in pivot_city_cleaned.columns]
            if len(actual_models_in_city_pivot) < 2:
                city_summary_list.append(common_data_for_summary)
                continue
            pivot_city_cleaned = pivot_city_cleaned[actual_models_in_city_pivot]

            stat_f, p_value_f = np.nan, 1.0  # Inicializa p_value_f com 1.0
            try:
                stat_f, p_value_f = friedmanchisquare(*[pivot_city_cleaned[col] for col in actual_models_in_city_pivot])
            except ValueError:  # Pode ocorrer se todos os valores para um modelo forem iguais
                p_value_f = 1.0  # Assume que não há diferença se o teste falhar por falta de variância

            # Calcula ranks médios e identifica o melhor modelo (menor rank)
            ranks_city = pivot_city_cleaned.rank(axis=1, method='average', ascending=True)
            mean_ranks_city = ranks_city.mean().sort_values()
            best_model_this_city = mean_ranks_city.index[0] if not mean_ranks_city.empty else "N/A"

            significant_diff_nemenyi = False
            if not pd.isna(p_value_f) and p_value_f < 0.05:  # Se Friedman foi significativo
                try:
                    nemenyi_city_data = pivot_city_cleaned.to_numpy()
                    if nemenyi_city_data.shape[0] > 1 and nemenyi_city_data.shape[1] > 1:
                        nemenyi_city_results = sp.posthoc_nemenyi_friedman(nemenyi_city_data)
                        nemenyi_city_results.columns = actual_models_in_city_pivot
                        nemenyi_city_results.index = actual_models_in_city_pivot
                        np.fill_diagonal(nemenyi_city_results.values,
                                         1)  # Ignora a diagonal (comparação de um modelo consigo mesmo)
                        if (nemenyi_city_results < 0.05).any().any():  # Verifica se há algum p-valor < 0.05
                            significant_diff_nemenyi = True
                except Exception:  # Ignora erros no Nemenyi para não interromper a execução
                    pass

            city_summary_list.append(
                {'lag': current_lag, 'out': current_out, 'node_id': node_id, 'metric': metric_name_node,
                 'friedman_p_value': p_value_f,
                 'best_model_by_rank': best_model_this_city,
                 'significant_diff_found': significant_diff_nemenyi})

    if city_summary_list:
        city_summary_df_current_config = pd.DataFrame(city_summary_list)
        return city_summary_df_current_config
    return pd.DataFrame()  # Retorna DataFrame vazio se nada foi processado


# --------------------------------------------------------------------------- #
# Função Principal (main)
# --------------------------------------------------------------------------- #

def main():
    """
    Função principal para executar a análise dos resultados dos modelos.

    Configura os parâmetros, itera sobre países, lags, horizons de previsão e
    configurações de modelos. Carrega os resultados, calcula métricas,
    realiza análises estatísticas e salva os outputs (CSVs e plots).
    """
    # --- Configurações da Análise ---
    task_type = 'regression'  # Tipo de tarefa (ex: 'regression', 'classification')
    backbone = True  # Indica se foi usado um backbone (afeta o nome da pasta)
    threshold = 0.01  # Threshold usado (afeta o nome da pasta)
    max_reps = 2  # Número máximo de repetições dos experimentos a serem carregadas

    # Hiperparâmetros dos modelos que foram variados no tuning
    K_values = [1, 2, 3, 4]
    hidden_channels = [32, 64, 128, 256]

    # Nomes dos modelos a serem analisados (a ordem pode ser importante para algumas visualizações)
    namemodels = ['GCRN', 'GCLSTM', 'LSTM']

    # Configurações de lags (janelas de entrada) e outs (horizontes de previsão) a serem analisadas
    lags_options = [7, 14]
    outs_options = [7, 14]

    # Lista de países para processar
    countries = ['Brazil', 'China']
    # --- Fim das Configurações ---

    # Determina o caminho base do projeto (pasta pai da pasta atual do script)
    try:
        pasta_atual = os.path.dirname(os.path.realpath(__file__))
    except NameError:  # Ocorre se executado interativamente (ex: Jupyter Notebook)
        pasta_atual = os.getcwd()

    pasta_projeto_base = os.path.dirname(pasta_atual)
    # Lógica para encontrar a pasta raiz do projeto, caso o script esteja em subpastas como "scripts" ou "code"
    if not pasta_projeto_base or pasta_projeto_base == pasta_atual or \
            os.path.basename(pasta_projeto_base).lower() == os.path.basename(pasta_atual).lower():
        pasta_projeto_base = pasta_atual if not (os.path.basename(pasta_atual).lower() in ["scripts", "code", "src",
                                                                                           "notebooks"]) else os.path.dirname(
            pasta_atual)
    print(f"Pasta base do projeto: {pasta_projeto_base}")

    # Loop principal sobre os países configurados
    for current_country in countries:
        print(f"\n===== PROCESSANDO PAÍS: {current_country.upper()} =====")

        pasta_projeto = pasta_projeto_base  # Caminho base para este país

        # Constrói o caminho para a pasta de resultados brutos dos modelos para o país atual
        pasta_results_base_dir = os.path.join('results_tune_1', current_country, task_type,
                                              f'backbone_threshold_{threshold * 100:.0f}' if backbone else '')
        pasta_results_abs = os.path.join(pasta_projeto, pasta_results_base_dir.strip(os.sep))

        print(f"Procurando resultados em: {pasta_results_abs}")
        if not os.path.exists(pasta_results_abs):
            print(f"Erro: Diretório de resultados não existe para {current_country}: {pasta_results_abs}");
            continue  # Pula para o próximo país se a pasta de resultados não existir

        # Cria a pasta para salvar os resultados da análise para o país atual
        analysis_folder_country = os.path.join(pasta_projeto, 'results_tune_1', 'analysis_results', current_country)
        os.makedirs(analysis_folder_country, exist_ok=True)
        print(f"Pasta de análise para {current_country}: {analysis_folder_country}")

        # Listas para armazenar todos os resultados coletados para o país atual
        all_results_list_country = []  # Para métricas gerais
        all_results_per_node_list_country = []  # Para métricas por nó/cidade

        # Define o iterador para todas as combinações de lags, outs, modelos e hiperparâmetros
        iteravel_configs = product(lags_options, outs_options, namemodels, [task_type], K_values, hidden_channels)

        # Loop para carregar os dados de cada configuração de modelo
        for lag, out, model, _, K, hc in iteravel_configs:
            y_real_reps_content, y_pred_reps_content = [], []  # Listas para armazenar dados das repetições

            for rep_idx in range(max_reps):
                # Constrói o caminho para a pasta da configuração específica do modelo e repetição
                model_dir = os.path.join(pasta_results_abs, f'k_{K}_hc_{hc}', model, f'lags_{lag}_out_{out}')

                # Otimização: se o diretório não existe para a primeira repetição, provavelmente não existe para as outras
                if not os.path.exists(model_dir) and rep_idx == 0:
                    break  # Sai do loop de repetições
                if not os.path.exists(model_dir):
                    continue  # Pula para a próxima repetição se o diretório específico da repetição não existir

                # Caminhos para os arquivos .npy de valores reais e previstos
                y_real_path = os.path.join(model_dir, f'y_real_no_norm_rep_{rep_idx}.npy')
                y_pred_path = os.path.join(model_dir, f'y_pred_no_norm_rep_{rep_idx}.npy')

                # Carrega os dados se ambos os arquivos existirem
                if os.path.exists(y_real_path) and os.path.exists(y_pred_path):
                    try:
                        y_r, y_p = np.load(y_real_path), np.load(y_pred_path)
                        # Validação da dimensionalidade dos arrays carregados
                        if y_r.ndim != 3 or y_p.ndim != 3:  # Esperado: (amostras, nós, saídas_previstas)
                            print(
                                f"Shape inesperado {y_r.shape if 'y_r' in locals() else 'N/A'} K{K} hc{hc} {model} L{lag} O{out} rep {rep_idx}, pulando.");
                            continue
                        y_real_reps_content.append(y_r)
                        y_pred_reps_content.append(y_p)
                    except Exception as e:
                        print(f"Erro carregando rep {rep_idx} para {model} K{K} hc{hc} L{lag} O{out}: {e}")

            # Processa os dados coletados se houver conteúdo para a configuração atual
            if y_real_reps_content:
                # Empilha os arrays das repetições para formar um único array
                # Shape resultante: (num_repeticoes, num_amostras, num_nos, num_saidas_previstas)
                y_real_concat = np.stack(y_real_reps_content)
                y_pred_concat = np.stack(y_pred_reps_content)
                num_reps_found = y_real_concat.shape[0]
                print(
                    f"  País {current_country} - Processado L{lag} O{out} K{K} hc{hc} {model}: {num_reps_found} repetições.")

                # String identificadora da configuração de hiperparâmetros
                param_config_str = f"K{K}_hc{hc}_lag{lag}_out{out}"

                # Calcula métricas gerais para esta configuração
                metrics_overall = calculate_metrics(y_real_concat, y_pred_concat)
                all_results_list_country.append({'param_config': param_config_str, 'K': K, 'hidden_channel': hc,
                                                 'model': model, 'lag': lag, 'out': out,
                                                 'num_reps_found': num_reps_found,
                                                 **metrics_overall})

                # Calcula métricas por nó para esta configuração
                metrics_nodes = calculate_metrics_per_node(y_real_concat, y_pred_concat)
                for node_metric in metrics_nodes:
                    all_results_per_node_list_country.append(
                        {'param_config': param_config_str, 'K': K, 'hidden_channel': hc,
                         'model': model, 'lag': lag, 'out': out,
                         'num_reps_found_for_config': num_reps_found, **node_metric})

        # --- Fim do loop de carregamento de dados para o país atual ---

        # Salva os resultados gerais detalhados para o país
        if not all_results_list_country:
            print(f"Nenhum resultado geral coletado para {current_country}.")
        else:
            results_df_country = pd.DataFrame(all_results_list_country)
            results_df_country.to_csv(
                os.path.join(analysis_folder_country, f'all_results_detailed_stats_{current_country}.csv'), index=False)
            print(f"\nResultados gerais detalhados para {current_country} salvos.")

            # --- Início da Análise Geral (CD Plots) ---
            # Itera sobre cada combinação única de (lag, out) encontrada nos resultados
            unique_lag_out_configs = results_df_country[['lag', 'out']].drop_duplicates().values
            for current_lag_iter, current_out_iter in unique_lag_out_configs:
                print(
                    f"\n== Iniciando Análise Geral para {current_country}, Lag: {current_lag_iter}, Out: {current_out_iter} ==")
                # Filtra o DataFrame para a combinação (lag, out) atual
                # Adiciona .copy() para evitar SettingWithCopyWarning na função de análise ao adicionar colunas
                df_subset_general = results_df_country[
                    (results_df_country['lag'] == current_lag_iter) &
                    (results_df_country['out'] == current_out_iter)
                    ].copy()

                if not df_subset_general.empty:
                    # Chama a função de análise e plotagem para RMSE
                    general_analysis_with_custom_cd_plot(df_subset_general, 'RMSE_mean', 'RMSE',
                                                         analysis_folder_country, namemodels,
                                                         current_lag_iter, current_out_iter)
                    # Chama a função de análise e plotagem para MAE
                    general_analysis_with_custom_cd_plot(df_subset_general, 'MAE_mean', 'MAE',
                                                         analysis_folder_country, namemodels,
                                                         current_lag_iter, current_out_iter)
                else:
                    print(
                        f"  Nenhum dado para Análise Geral (Lag: {current_lag_iter}, Out: {current_out_iter}) para {current_country}.")
            # --- Fim da Análise Geral ---

        # Salva os resultados detalhados por nó para o país
        if not all_results_per_node_list_country:
            print(f"Nenhum resultado por nó coletado para {current_country}.")
        else:
            results_per_node_df_country = pd.DataFrame(all_results_per_node_list_country)
            results_per_node_df_country.to_csv(
                os.path.join(analysis_folder_country, f'all_results_per_node_{current_country}.csv'), index=False)
            print(f"Resultados detalhados por nó para {current_country} salvos.")

            # --- Início da Análise por Cidade/Nó ---
            all_city_summaries_country = []  # Lista para agregar sumários de todas as configs (lag, out)

            # Itera sobre cada combinação única de (lag, out) encontrada nos resultados por nó
            unique_lag_out_configs_node = results_per_node_df_country[['lag', 'out']].drop_duplicates().values
            for current_lag_iter, current_out_iter in unique_lag_out_configs_node:
                print(
                    f"\n== Iniciando Análise por Cidade para {current_country}, Lag: {current_lag_iter}, Out: {current_out_iter} ==")
                # Filtra o DataFrame para a combinação (lag, out) atual
                # Adiciona .copy() aqui também, embora a função per_city_analysis_tabular já faça uma cópia interna
                # para 'city_df', é uma boa prática se df_subset_node fosse modificado antes da chamada.
                df_subset_node = results_per_node_df_country[
                    (results_per_node_df_country['lag'] == current_lag_iter) &
                    (results_per_node_df_country['out'] == current_out_iter)
                    ].copy()

                if not df_subset_node.empty:
                    # Realiza a análise tabular por cidade e obtém o sumário
                    city_summary_df_for_config = per_city_analysis_tabular(df_subset_node, analysis_folder_country,
                                                                           namemodels, current_lag_iter,
                                                                           current_out_iter)
                    if not city_summary_df_for_config.empty:
                        all_city_summaries_country.append(city_summary_df_for_config)
                else:
                    print(
                        f"  Nenhum dado para Análise por Cidade (Lag: {current_lag_iter}, Out: {current_out_iter}) para {current_country}.")

            # Consolida e salva o sumário da análise por cidade para todas as configs (lag,out)
            if all_city_summaries_country:
                final_city_summary_df_country = pd.concat(all_city_summaries_country, ignore_index=True)
                summary_file_path = os.path.join(analysis_folder_country,
                                                 f"per_city_statistical_summary_ALL_LAGS_OUTS_{current_country}.csv")
                final_city_summary_df_country.to_csv(summary_file_path, index=False)
                print(
                    f"\nResumo da análise estatística por cidade (todas configs L/O) para {current_country} salvo em: {summary_file_path}")

                # Imprime um resumo geral no console
                print(
                    f"\n--- Resumo Geral (Melhor Modelo por Rank Médio por Cidade) para {current_country.upper()} (Todas Configs L/O) ---")
                for metric in ['RMSE', 'MAE']:
                    # Contagem de vezes que cada modelo foi o melhor (menor rank médio)
                    best_model_counts = final_city_summary_df_country[
                        (final_city_summary_df_country['metric'] == metric) &
                        (final_city_summary_df_country['best_model_by_rank'] != 'N/A')  # Considera apenas casos válidos
                        ]['best_model_by_rank'].value_counts()

                    print(f"\nContagem de 'Melhor Modelo por Rank Médio' para {metric} em {current_country}:")
                    if not best_model_counts.empty:
                        print(best_model_counts)
                    else:
                        print("  Nenhum resultado para exibir.")

                    # Contagem de configurações (cidade-lag-out) com diferenças significativas
                    num_city_metric_configs_with_sig_diff = final_city_summary_df_country[
                        (final_city_summary_df_country['metric'] == metric) &
                        (final_city_summary_df_country['significant_diff_found'] == True)
                        ].shape[0]

                    total_city_metric_configs = final_city_summary_df_country[
                        final_city_summary_df_country['metric'] == metric
                        ].shape[0]

                    print(
                        f"Número de configurações (cidade-lag-out) com diferenças significativas (Nemenyi p<0.05) para {metric} em {current_country}: {num_city_metric_configs_with_sig_diff} de {total_city_metric_configs}")
            else:
                print(f"  Nenhum dado processado para o resumo da análise por cidade para {current_country}.")
            # --- Fim da Análise por Cidade/Nó ---

        print(f"===== PROCESSAMENTO PARA {current_country.upper()} CONCLUÍDO =====")


# Ponto de entrada do script: executa a função main se o script for rodado diretamente
if __name__ == '__main__':
    main()
