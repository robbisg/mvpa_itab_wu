
def setup_analysis(file_name):
    '''
    Load file given filename, guessing filetype.

    Parameters
    ----------
    file_name : string
        Filename to be loaded.

    Returns
    -------
    data: :rtype ndarray
       score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

    See also
    --------
    adjusted_rand_score: Adjusted Rand Index
    adjusted_mutual_info_score: Adjusted Mutual Information (adjusted
        against chance)

    Examples
    --------

    Perfect labelings are both homogeneous and complete, hence have
    score 1.0::

      >>> from sklearn.metrics.cluster import normalized_mutual_info_score
      >>> normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1])
      1.0
      >>> normalized_mutual_info_score([0, 0, 1, 1], [1, 1, 0, 0])
      1.0

    If classes members are completely split across different clusters,
    the assignment is totally in-complete, hence the NMI is null::

      >>> normalized_mutual_info_score([0, 0, 0, 0], [0, 1, 2, 3])
      0.0

    '''
    
    return
    