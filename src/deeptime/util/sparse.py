from scipy.sparse import coo_matrix


def remove_negative_entries(input_matrix):
    r"""Remove all negative entries from sparse matrix.

    Parameters
    ----------
    input_matrix : (M, M) scipy.sparse matrix
        Input matrix

    Returns
    -------
    non_negative_mat : (M, M) scipy.sparse matrix
        Input matrix with negative entries set to zero.
    """
    input_matrix = input_matrix.tocoo()

    data = input_matrix.data
    row = input_matrix.row
    col = input_matrix.col

    pos = data > 0.0
    datap = data[pos]
    rowp = row[pos]
    colp = col[pos]
    return coo_matrix((datap, (rowp, colp)), shape=input_matrix.shape)
