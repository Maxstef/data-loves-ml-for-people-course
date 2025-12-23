

def save_df_zip_csv(df, filename):
    """
    Save a pandas DataFrame as a compressed ZIP CSV file.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to save.
    filename : str
        The base filename (without extension). The resulting file will be saved as `filename.zip`
        containing a CSV named `filename.csv`.

    Returns
    -------
    None
        Writes a ZIP-compressed CSV file to disk.

    Notes
    -----
    - The CSV inside the ZIP will not include the index.
    - Compression is handled using pandas' built-in ZIP support.
    
    Example
    -------
    >>> save_df_zip_csv(df, 'my_data')
    This creates 'my_data.zip' containing 'my_data.csv'.
    """

    compression_options = dict(method='zip', archive_name=f'{filename}.csv')
    df.to_csv(f'{filename}.zip', compression=compression_options, index=False)