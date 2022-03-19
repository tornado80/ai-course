def reproduce(n, par1, par2):
    lcs = [[0 for _ in range(n)] for _ in range(n)]  # longest common substring
    prev = [[0 for _ in range(n)] for _ in range(n)]  # longest common substring
    for i in range(n):
        for j in range(n):
            val = par2[j]
            ind = par1.index(val)
            if (j > 0):
                lcs[i][j] = lcs[i][j - 1]
                prev[i][j] = prev[i][j - 1]
            if (ind <= i):
                if (ind > 0 and j > 0 and lcs[i][j] < lcs[ind - 1][j - 1] + 1):
                    lcs[i][j] = lcs[ind - 1][j - 1] + 1
                    prev[i][j] = val
                elif lcs[i][j] == 0:
                    lcs[i][j] = 1
                    prev[i][j] = val
    real_lcs = []
    iter = 0
    i, j = n - 1, n - 1
    while iter < lcs[n - 1][n - 1]:
        real_lcs.insert(0, prev[i][j])
        i, j = par1.index(prev[i][j]) - 1, par2.index(prev[i][j]) - 1
        iter += 1
    real_lcs_set = set(real_lcs)
    complement = []
    for gene in par2:
        if gene not in real_lcs_set:
            complement.append(gene)
    child1 = []
    complement_iter = 0
    for gene in par1:
        if gene in real_lcs_set:
            child1.append(gene)
        else:
            child1.append(complement[complement_iter])
            complement_iter += 1
    return child1

print(reproduce(6, [1, 2, 6, 3, 5, 4], [5, 1, 6, 2, 3, 4]))

