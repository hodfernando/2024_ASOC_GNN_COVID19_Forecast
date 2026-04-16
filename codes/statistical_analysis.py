import os
import pandas as pd
import numpy as np
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, wilcoxon
from scipy.stats import studentized_range
import matplotlib.pyplot as plt
from itertools import combinations


# --- FUNÇÃO 1: CARREGAR E PREPARAR DADOS ---
def carregar_e_preparar_dados(pasta_analise_geral, task_type):
    caminho_csv_sumario = os.path.join(pasta_analise_geral, 'all_models_metrics_summary.csv')
    print(f"Procurando por arquivo de resumo em: {caminho_csv_sumario}")
    if not os.path.exists(caminho_csv_sumario):
        print(f"  AVISO: Arquivo de resumo '{os.path.basename(caminho_csv_sumario)}' não encontrado.")
        print(
            f"  ERRO: Certifique-se de que o arquivo '{os.path.basename(caminho_csv_sumario)}' está no caminho correto.")
        return None
    df_completo = pd.read_csv(caminho_csv_sumario)
    if task_type == 'regression':
        if 'Identificador' not in df_completo.columns:
            print("  ERRO: A coluna 'Identificador' não foi encontrada no CSV de regressão.")
            return None
        regex_pattern = r'(?P<model_name>[A-Z]+)_k(?P<k>\d+)_h(?P<hc>\d+)_lags(?P<lags>\d+)_out(?P<out>\d+)_alpha(?P<alpha>[\d.]+)'
        df_params = df_completo['Identificador'].str.extract(regex_pattern)
        if df_params.empty or df_params['model_name'].isnull().all():
            print("  ERRO: Regex não conseguiu extrair parâmetros da coluna 'Identificador'.")
            return None
        for col in ['k', 'hc', 'lags', 'out', 'alpha']:
            df_params[col] = pd.to_numeric(df_params[col], errors='coerce')
        df_final = pd.concat([df_completo, df_params], axis=1)
    else:
        df_final = df_completo
        for col in ['lags', 'out']:
            if col in df_final.columns:
                df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    df_final.dropna(subset=['model_name', 'lags', 'out'], inplace=True)
    print(f"  Dados carregados e preparados com sucesso. Total de {len(df_final)} registros.")
    return df_final


def gerar_diagrama_cd_manual(df_resultados, metrica, pasta_saida, ascending_rank=True, alpha=0.05):
    print(f"\n--- Gerando Diagrama de Diferença Crítica (CD Real) para a métrica: {metrica} ---")

    HORIZON_WINDOWS = [[1, 14], [1, 5], [6, 10], [11, 14]]
    HORIZON_WINDOWS_LABELS = ['(all horizons)', '(first days only)', '(intermediate horizon)', '(long horizon)']

    fig, axes = plt.subplots(len(HORIZON_WINDOWS), 1, figsize=(10, 1.2 * len(HORIZON_WINDOWS)), dpi=100)

    if len(HORIZON_WINDOWS) == 1:
        axes = [axes]

    for i, horizon_window in enumerate(HORIZON_WINDOWS):
        print(f"\nAnalisando horizonte de previsão: {horizon_window[0]} a {horizon_window[1]} dias à frente")

        # Filtragem dos dados para o horizonte atual
        df_resultados_filtered = df_resultados[
            (df_resultados['out'] > horizon_window[0] - 1) & (df_resultados['out'] < horizon_window[1] + 1)
            ]

        # --- GERAÇÃO DE CSV DE ESTATÍSTICAS
        if i == 0:
            try:
                print(f"  [INFO] Calculando estatísticas descritivas para {HORIZON_WINDOWS_LABELS[i]}...")

                # Agrupa por modelo e calcula as métricas solicitadas na tabela LaTeX
                stats_df = df_resultados_filtered.groupby('model_name')[metrica].agg(
                    Maximum='max',
                    Minimum='min',
                    Mean='mean',
                    Standard_Deviation='std',
                    First_Quartile=lambda x: x.quantile(0.25),
                    Median='median',
                    Third_Quartile=lambda x: x.quantile(0.75)
                )

                # Transpõe para ficar igual à estrutura das tabelas (Estatísticas nas linhas, Modelos nas colunas)
                stats_df_transposed = stats_df.transpose()

                # Nome do arquivo CSV
                csv_filename = f"tabela_estatisticas_{metrica}_all_horizons.csv"
                csv_path = os.path.join(pasta_saida, csv_filename)

                # Salva o CSV
                stats_df_transposed.to_csv(csv_path)
                print(f"  [SUCESSO] Arquivo de estatísticas salvo em: {csv_path}")

                # Exibe uma prévia no console para conferência rápida
                print(stats_df_transposed)

            except Exception as e:
                print(f"  [ERRO] Falha ao gerar CSV de estatísticas: {e}")

        # --- Preparação dos dados para o CD Diagram ---
        pivot_table = df_resultados_filtered.pivot_table(index=['lags', 'out'], columns='model_name', values=metrica)
        pivot_table.dropna(inplace=True)

        if len(pivot_table) < 2 or len(pivot_table.columns) < 2:
            print("  ⚠️ Dados insuficientes para o teste de Friedman.")
            # Se for a última iteração ou return, cuidado para não matar o loop antes de processar os outros se houvesse,
            # mas como é return, aborta a função.
            return

        models = pivot_table.columns

        # --- Teste de Friedman ---
        stat, p_value = friedmanchisquare(*[pivot_table[col] for col in models])
        print(f"  Teste de Friedman: Estatística={stat:.4f}, p-valor={p_value:.4g}")

        # --- Cálculo dos ranks médios ---
        mean_ranks = pivot_table.rank(axis=1, ascending=ascending_rank).mean().sort_values()
        print(f"\n  Ranks Médios (Posição relativa - Para o Gráfico CD):\n{mean_ranks}")

        # --- Cálculo dos valores reais ---
        mean_values = pivot_table.mean().sort_values()
        print(f"\n  Média Real dos Valores ({metrica}):\n{mean_values}")

        # --- Teste de Nemenyi (opcional) ---
        if p_value < 0.05:
            # nemenyi_results = sp.posthoc_nemenyi_friedman(pivot_table.to_numpy()) # Versão antiga/alternativa
            # É mais seguro usar o wrapper do scikit-posthocs que aceita o DataFrame diretamente se a versão permitir,
            # mas mantendo sua estrutura original:
            nemenyi_results = sp.posthoc_nemenyi_friedman(pivot_table.to_numpy())
            nemenyi_results.columns = models
            nemenyi_results.index = models

            ax = axes[i]
            ax.set_title(f"Horizon {horizon_window[0]}–{horizon_window[1]} days {HORIZON_WINDOWS_LABELS[i]}",
                         fontsize=10)

            sp.critical_difference_diagram(
                ranks=mean_ranks,
                sig_matrix=nemenyi_results,
                label_fmt_left='{label} [{rank:.4f}]  ',
                label_fmt_right='  [{rank:.4f}] {label}',
                text_h_margin=0.3,
                crossbar_props={'color': None, 'marker': 'o'},
                elbow_props={'color': 'black'},
                ax=ax
            )
        else:
            axes[i].text(0.5, 0.5,
                         f"Horizon {horizon_window[0]}–{horizon_window[1]} days {HORIZON_WINDOWS_LABELS[i]}\nThe Friedman test showed no significant difference\n(p={p_value:.3f})",
                         ha='center', va='center', fontsize=10)
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(pasta_saida, f"cd_diagram_{metrica}_all_horizons.pdf"))


# --- FUNÇÃO 3: NOVA ANÁLISE DE BOOTSTRAP ---
def gerar_analise_bootstrap(df_resultados, metrica, pasta_saida, modelo_base='LSTM', ascending=True):
    """
    Realiza análise de bootstrap para calcular o Intervalo de Confiança de 95%
    da diferença de performance entre os modelos e o modelo base.
    """
    print(f"\n--- Gerando Análise de Bootstrap para a métrica: {metrica} (vs. {modelo_base}) ---")

    modelos_para_comparar = [m for m in df_resultados['model_name'].unique() if m != modelo_base]

    resultados_bootstrap = {}
    np.random.seed(42)  # Para reprodutibilidade
    n_resamples = 9999  # Número de reamostragens

    for modelo in modelos_para_comparar:
        df_pivot = df_resultados.pivot_table(index=['lags', 'out'], columns='model_name', values=metrica)
        scores_base = df_pivot[modelo_base].dropna()
        scores_modelo = df_pivot[modelo].dropna()

        common_index = scores_base.index.intersection(scores_modelo.index)
        if len(common_index) < 2: continue

        scores_base = scores_base.loc[common_index]
        scores_modelo = scores_modelo.loc[common_index]

        # A diferença é calculada para que um valor positivo sempre signifique "melhora"
        diffs = (scores_base - scores_modelo) if ascending else (scores_modelo - scores_base)

        bootstrap_means = np.array([
            np.mean(np.random.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_resamples)
        ])

        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)

        resultados_bootstrap[modelo] = (ci_lower, diffs.mean(), ci_upper)

    if not resultados_bootstrap:
        print("  Dados insuficientes para a análise de Bootstrap.")
        return

    # Plotando os resultados
    fig, ax = plt.subplots(figsize=(10, len(resultados_bootstrap) * 2))
    model_names = list(resultados_bootstrap.keys())

    for i, model in enumerate(model_names):
        lower, mean, upper = resultados_bootstrap[model]
        ax.plot([lower, upper], [i, i], lw=5, solid_capstyle='round', alpha=0.8)
        ax.plot(mean, i, 'o', color='white', markersize=8, markeredgecolor='black', zorder=5)

    ax.axvline(x=0, color='red', linestyle='--', label='No difference')
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel(f"Improvement of '{metrica}' over {modelo_base}")
    ax.set_title(f"95% Confidence Interval (Bootstrap) of the Improvement over the {modelo_base}", fontsize=14)
    ax.legend()
    plt.tight_layout()

    caminho_arquivo = os.path.join(pasta_saida, f'bootstrap_ci_{metrica}.pdf')
    fig.savefig(caminho_arquivo, dpi=300)
    plt.close(fig)
    print(f"  ✅ Gráfico de Bootstrap salvo em: {caminho_arquivo}")


# --- FUNÇÃO PRINCIPAL (ORQUESTRADOR) ---
def main():
    PAISES = ['Brazil', 'China']
    TASK_TYPES = ['regression', 'classification']

    try:
        pasta_atual = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        pasta_atual = os.getcwd()
    pasta_projeto = os.path.dirname(pasta_atual)
    if pasta_projeto.startswith('/media/work/'):
        pasta_projeto = pasta_projeto.replace('/media/work/', '/media/data/', 1)
        print(f"Caminho do projeto modificado para usar '/media/data/'.")

    for pais in PAISES:
        for task_type in TASK_TYPES:
            print(f"\n{'=' * 30} INICIANDO ANÁLISE ESTATÍSTICA PARA: {pais.upper()} - {task_type.upper()} {'=' * 30}")

            pasta_analise = os.path.join(pasta_projeto, 'results_daily', 'results_analysis', pais, task_type,
                                         'grafo_original')
            print(f'{pasta_projeto=}')
            df_completo = carregar_e_preparar_dados(pasta_analise, task_type)
            if df_completo is None or df_completo.empty: continue

            pasta_figuras = os.path.join(pasta_analise, 'statistical_figures')
            os.makedirs(pasta_figuras, exist_ok=True)

            if task_type == 'regression':
                for metrica in ['RMSE']:  # , 'MAE'
                    gerar_diagrama_cd_manual(df_completo, metrica, pasta_figuras, ascending_rank=True)
                    # gerar_analise_bootstrap(df_completo, metrica, pasta_figuras, modelo_base='LSTM', ascending=True)
            elif task_type == 'classification':
                for metrica in ['F1_Score', 'Precision', 'Recall']:  # 'Accuracy',
                    gerar_diagrama_cd_manual(df_completo, metrica, pasta_figuras, ascending_rank=False)
                    # gerar_analise_bootstrap(df_completo, metrica, pasta_figuras, modelo_base='LSTM', ascending=False)

    print("\nAnálise estatística e geração de figuras concluídas com sucesso!")


if __name__ == '__main__':
    main()
