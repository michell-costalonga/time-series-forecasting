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
