from pyspark.sql.functions import col, count, max, lit
from pyspark.sql.window import Window


def rename_columns(df_pyspark_agg, target_column_name, date_column_name):
    """
    Retorna uma PySpark DataFrame com as colunas renomeadas adequadamente para realizar as previsões.

    :param df_pyspark_agg: PySpark DataFrame agregado por data e grupos de classificação com a contagem, ou valores que servirão para treinar os algoritmos.
    :param target_column_name: Nome da coluna com os valores alvos.
    :param date_column_name: Nome da coluna com os valores de data/hora.

    :return: PySpark DataFrame com as colunas "ds" e "y".
    """

    df_preprocessed = df_pyspark_agg.withColumnRenamed(
        date_column_name, "ds"
    ).withColumnRenamed(target_column_name, "y")

    return df_preprocessed


def filter_dataframe(df_pyspark_agg):
    """
    Retorna dois PySpark DataFrames filtrados e preparados para o treino e previsão usando os algoritmos (modelo e média). Para o modelo, a contagem de valores deve ser maior que 80% do número máximo de dias preenchidos. Para os casos que não se encaixam na regra anterior, a média é usada para a previsão. As bandas superior e inferior são calculadas como o valor médio mais/menos duas vezes o desvio padrão.

    :param df_pyspark_agg: PySpark DataFrame agregado por data e grupos de classificação com a contagem, ou valores que servirão para treinar os algoritmos.

    :return: Dois PySpark DataFrames filtrados e prontos para o uso para a previsão.
    """
    columns_partition_by = [
        col for col in df_pyspark_agg.columns if col not in ["ds", "y"]
    ]

    df_enriched = df_pyspark_agg.withColumn(
        "filled_ds", count(col("ds")).over(Window.partitionBy(columns_partition_by))
    ).withColumn(
        "max_filled_ds", max(col("filled_ds")).over(Window.partitionBy(lit("all")))
    )

    df_filtered_model = df_enriched.filter(
        col("filled_ds") > col("max_filled_ds") * 0.8
    ).drop(*["filled_ds", "max_filled_ds"])
    df_filtered_avg = df_enriched.filter(
        col("filled_ds") <= col("max_filled_ds") * 0.8
    ).drop(*["filled_ds", "max_filled_ds"])
    return df_filtered_model, df_filtered_avg
