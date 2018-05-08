import numpy as np
import pdb
from cvxopt import matrix, solvers
import scipy.spatial.distance as ssd
import sklearn.metrics as sklm

solvers.options['show_progress'] = False


def gl_sig_model(inp_signal, max_iter, alpha, beta):
    """
    Returns Output Signal Y, Graph Laplacian L
    """
    eps = 1e-4
    Y = inp_signal.T
    # Each column is a signal
    num_vertices = inp_signal.shape[1]
    M_mat, P_mat, A_mat, b_mat, G_mat, h_mat = create_static_matrices_for_L_opt(num_vertices, beta)
    # M_c = matrix(M_mat)
    P_c = matrix(P_mat)
    A_c = matrix(A_mat)
    b_c = matrix(b_mat)
    G_c = matrix(G_mat)
    h_c = matrix(h_mat)
    curr_cost = np.linalg.norm(np.ones((num_vertices, num_vertices)), 'fro')
    q_mat = alpha * np.dot(np.ravel(np.dot(Y, Y.T)), M_mat)
    for it in range(max_iter):
        # pdb.set_trace()
        # Update L
        prev_cost = curr_cost
        # pdb.set_trace()
        q_c = matrix(q_mat)
        sol = solvers.qp(P_c, q_c, G_c, h_c, A_c, b_c)
        l_vech = np.array(sol['x'])
        l_vec = np.dot(M_mat, l_vech)
        L = l_vec.reshape(num_vertices, num_vertices)
        # Assert L is correctly learnt.
        # assert L.trace() == num_vertices
        try:
            assert np.allclose(L.trace(), num_vertices)
            assert np.all(L - np.diag(np.diag(L)) <= eps)
            assert np.allclose(np.dot(L, np.ones(num_vertices)), np.zeros(num_vertices))
        except AssertionError as e:
            pdb.set_trace()
        # Update Y
        Y = np.dot(np.linalg.inv(np.eye(num_vertices) + alpha * L), inp_signal.T)

        curr_cost = (np.linalg.norm(inp_signal.T - Y, 'fro')**2 +
                     alpha * np.dot(np.dot(Y.T, L), Y).trace() +
                     beta * np.linalg.norm(L, 'fro')**2)
        q_mat = alpha * np.dot(np.ravel(np.dot(Y, Y.T)), M_mat)
        # pdb.set_trace()
        calc_cost = (0.5 * np.dot(np.dot(l_vech.T, P_mat), l_vech).squeeze() +
                     np.dot(q_mat, l_vech).squeeze() + np.linalg.norm(inp_signal.T - Y, 'fro')**2)
        # pdb.set_trace()
        assert np.allclose(curr_cost, calc_cost)
        # print(curr_cost)
        if np.abs(curr_cost - prev_cost) < 1e-4:
            # print('Stopped at Iteration', it)
            break
        # print
    return L, Y, it


def create_static_matrices_for_L_opt(num_vertices, beta):
    # Static matrices are those independent of Y
    #
    M_mat = create_dup_matrix(num_vertices)
    P_mat = 2 * beta * np.dot(M_mat.T, M_mat)
    A_mat = create_A_mat(num_vertices)
    b_mat = create_b_mat(num_vertices)
    G_mat = create_G_mat(num_vertices)
    h_mat = np.zeros(G_mat.shape[0])
    return M_mat, P_mat, A_mat, b_mat, G_mat, h_mat


def get_u_vec(i, j, n):
    u_vec = np.zeros(n*(n+1)//2)
    pos = (j-1) * n + i - j*(j-1)//2
    u_vec[pos-1] = 1
    return u_vec


def get_T_mat(i, j, n):
    Tij_mat = np.zeros((n, n))
    Tij_mat[i-1, j-1] = Tij_mat[j-1, i-1] = 1
    return np.ravel(Tij_mat)


def create_dup_matrix(num_vertices):
    M_mat = np.zeros((num_vertices**2, num_vertices*(num_vertices + 1)//2))
    # tmp_mat = np.arange(num_vertices**2).reshape(num_vertices, num_vertices)
    for j in range(1, num_vertices+1):
        for i in range(j, num_vertices+1):
            u_vec = get_u_vec(i, j, num_vertices)
            Tij = get_T_mat(i, j, num_vertices)
            # pdb.set_trace()
            M_mat += np.outer(u_vec, Tij).T

    return M_mat


def get_a_vec(i, n):
    a_vec = np.zeros(n*(n+1)//2)
    if i == 0:
        a_vec[np.arange(n)] = 1
    else:
        tmp_vec = np.arange(n-1, n-i-1, -1)
        tmp2_vec = np.append([i], tmp_vec)
        tmp3_vec = np.cumsum(tmp2_vec)
        a_vec[tmp3_vec] = 1
        end_pt = tmp3_vec[-1]
        a_vec[np.arange(end_pt, end_pt + n-i)] = 1

    return a_vec


def create_A_mat(n):
    A_mat = np.zeros((n+1, n*(n+1)//2))
    # A_mat[0, 0] = 1
    # A_mat[0, np.cumsum(np.arange(n, 0, -1))] = 1
    for i in range(0, A_mat.shape[0] - 1):
        A_mat[i, :] = get_a_vec(i, n)
    A_mat[n, 0] = 1
    A_mat[n, np.cumsum(np.arange(n, 1, -1))] = 1

    return A_mat


def create_b_mat(n):
    b_mat = np.zeros(n+1)
    b_mat[n] = n
    return b_mat


def create_G_mat(n):
    G_mat = np.zeros((n*(n-1)//2, n*(n+1)//2))
    tmp_vec = np.cumsum(np.arange(n, 1, -1))
    tmp2_vec = np.append([0], tmp_vec)
    tmp3_vec = np.delete(np.arange(n*(n+1)//2), tmp2_vec)
    for i in range(G_mat.shape[0]):
        G_mat[i, tmp3_vec[i]] = 1

    return G_mat


def graph_eval(L_gt, L_pred, thr):
    W_gt_tmp = -(L_gt - np.diag(np.diag(L_gt)))
    W_pred_tmp = -(L_pred - np.diag(np.diag(L_pred)))

    assert (W_gt_tmp >= 0).all()
    assert (W_pred_tmp >= 0).all()

    W_pred_tmp[W_pred_tmp < thr] = 0

    edge_gt = (ssd.squareform(W_gt_tmp) != 0)
    edge_pred = (ssd.squareform(W_pred_tmp) != 0)
    # pdb.set_trace()
    num_edges = np.sum(edge_pred)
    if num_edges > 0:
        prec, rec, fscore, _ = sklm.precision_recall_fscore_support(edge_gt, edge_pred)
    return prec, rec, fscore


def get_f_score(prec, recall):
    return 2 * prec * recall / (prec + recall)


def get_MSE(L_out, L_gt):
    return np.linalg.norm(L_out - L_gt, 'fro')


# if __name__ == "__main__":
#     # np.random.seed(0)
#     solvers.options['show_progress'] = False
#     syn = synthetic_data_gen()
#     num_nodes = syn.num_vertices

#     prec_er_list = []
#     prec_ba_list = []
#     prec_rnd_list = []

#     recall_er_list = []
#     recall_ba_list = []
#     recall_rnd_list = []

#     f_score_er_list = []
#     f_score_ba_list = []
#     f_score_rnd_list = []

#     mse_rnd_list = []
#     for i in tqdm(range(100)):
#         np.random.seed(i)
#         graph_signals_er, graph_signals_ba, graph_signals_rand = syn.get_graph_signals()
#         L_er, Y_er = gl_sig_model(graph_signals_er, 1000, syn.alpha_er, syn.beta_er)
#         L_ba, Y_ba = gl_sig_model(graph_signals_ba, 1000, syn.alpha_er, syn.beta_er)
#         L_rnd, Y_rnd = gl_sig_model(graph_signals_rand, 1000, syn.alpha_rnd, syn.beta_rnd)

#         L_er[np.abs(L_er) < 1e-4] = 0
#         L_ba[np.abs(L_ba) < 1e-4] = 0

#         L_er_gt = syn.er_normL.toarray()
#         L_ba_gt = syn.ba_normL.toarray()
#         L_rnd_gt = syn.rg_normL.toarray()

#         prec_er, recall_er, f_score_er = graph_eval(L_er_gt, L_er, syn.thr_er)
#         prec_ba, recall_ba, f_score_ba = graph_eval(L_ba_gt, L_ba, syn.thr_ba)
#         # WHY is L_RND not symmetric??
#         # pdb.set_trace()
#         prec_rnd, recall_rnd, fscore_rnd = graph_eval(L_rnd_gt, L_rnd, syn.thr_rnd)

#         mse_rnd_list.append(get_MSE(L_rnd_gt, L_rnd))

#         prec_er_list.append(prec_er)
#         recall_er_list.append(recall_er)
#         f_score_er_list.append(f_score_er)

#         prec_ba_list.append(prec_ba)
#         recall_ba_list.append(recall_ba)
#         f_score_ba_list.append(f_score_ba)

#         # pdb.set_trace()
#         prec_rnd_list.append(prec_rnd)
#         recall_rnd_list.append(recall_rnd)
#         f_score_rnd_list.append(fscore_rnd)

#     print('Avg Prec ER', np.mean(prec_er_list))
#     print('Avg Recall ER', np.mean(recall_er_list))
#     print('Avg F-score ER', np.mean(f_score_er_list))

#     print('Avg Prec BA', np.mean(prec_ba_list))
#     print('Avg Recall BA', np.mean(recall_ba_list))
#     print('Avg F-score BA', np.mean(f_score_ba_list))

#     print('Avg Prec Rnd', np.mean(prec_rnd_list))
#     print('Avg Recall Rnd', np.mean(recall_rnd_list))
#     print('Avg F-score Rnd', np.mean(f_score_rnd_list))

#     print('Avg MSE Rnd', np.mean(mse_rnd_list))
