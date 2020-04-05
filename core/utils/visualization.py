from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch.nn.functional as F


label_colours = [(178, 45, 45), (153, 115, 115), (64, 36, 32), (255, 68, 0), (89, 24, 0), (191, 121, 96), (191, 102, 0),
                 (76, 41, 0), (153, 115, 38), (102, 94, 77), (242, 194, 0), (191, 188, 143), (226, 242, 0),
                 (119, 128, 0), (59, 64, 0), (105, 191, 48), (81, 128, 64), (0, 255, 0), (0, 51, 7), (191, 255, 208),
                 (96, 128, 113), (0, 204, 136), (13, 51, 43), (0, 191, 179), (0, 204, 255), (29, 98, 115), (0, 34, 51),
                 (163, 199, 217), (0, 136, 255), (41, 108, 166), (32, 57, 128), (0, 22, 166), (77, 80, 102),
                 (119, 54, 217), (41, 0, 77), (222, 182, 242), (103, 57, 115), (247, 128, 255), (191, 0, 153),
                 (128, 96, 117), (127, 0, 68), (229, 0, 92), (76, 0, 31), (255, 128, 179), (242, 182, 198)]


def process_image(image, image_mean):
    image = image.cpu().numpy() + image_mean[:, None, None]
    return image.astype(np.uint8)


def process_seg_label(pred, gt, num_classes=40):
    B, C, H, W = gt.size()
    pred = F.interpolate(pred, (H, W), mode='bilinear', align_corners=True)
    pred = pred.argmax(dim=1)[0].detach()
    gt = gt.squeeze(1)[0]
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    h, w = gt.shape
    pred_img = Image.new('RGB', (w, h), (255, 255, 255))  # unlabeled part is white (255, 255, 255)
    gt_img = Image.new('RGB', (w, h), (255, 255, 255))
    pred_pixels = pred_img.load()
    gt_pixels = gt_img.load()
    for j_, j in enumerate(gt):
        for k_, k in enumerate(j):
            if k < num_classes:
                gt_pixels[k_, j_] = label_colours[k]
                pred_pixels[k_, j_] = label_colours[pred[j_, k_]]
    return np.array(pred_img).transpose([2, 0, 1]), np.array(gt_img).transpose([2, 0, 1])


def normalize(x):
    return x / np.linalg.norm(x, ord=2, axis=0, keepdims=True)


def process_normal_label(pred, gt, ignore_label):
    B, C, H, W = gt.size()
    pred = F.interpolate(pred, (H, W), mode='bilinear', align_corners=True)
    pred = pred[0].detach()
    gt = gt[0]
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    mask = gt != ignore_label
    _, h, w = gt.shape
    pred = normalize(pred.reshape(3, -1)).reshape(3, h, w) * mask + (1 - mask)
    gt = normalize(gt.reshape(3, -1)).reshape(3, h, w) * mask + (1 - mask)
    return pred, gt


def save_heatmap(matrix, filename, vmin=0., vmax=1.):
    fig = plt.figure(0)
    fig.clf()
    plt.matshow(matrix, fignum=0, cmap=plt.cm.bwr, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(filename)
    img = Image.open(filename)
    return np.array(img).transpose((2, 0, 1))


def _process_params(G, center, dim):

    if not isinstance(G, nx.Graph):
        empty_graph = nx.Graph()
        empty_graph.add_nodes_from(G)
        G = empty_graph

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return G, center


def rescale_layout(pos, scale=1):
    # Find max length over all dimensions
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos


def task_layout(G, nodes, align='vertical',
                     scale=1, center=None, aspect_ratio=4./3):
    G, center = _process_params(G, center=center, dim=2)
    if len(G) == 0:
        return {}

    height = 1
    width = aspect_ratio * height
    offset = (width/2, height/2)

    top = set(nodes)
    bottom = set(G) - top
    top = sorted(top, key=lambda x: int(x.split('_')[-1]), reverse=align == 'vertical')
    bottom = sorted(bottom, key=lambda x: int(x.split('_')[-1]), reverse=align == 'vertical')
    nodes = list(top) + list(bottom)

    if align == 'vertical':
        left_xs = np.repeat(0, len(top))
        right_xs = np.repeat(width, len(bottom))
        left_ys = np.linspace(0, height, len(top))
        right_ys = np.linspace(0, height, len(bottom))

        top_pos = np.column_stack([left_xs, left_ys]) - offset
        bottom_pos = np.column_stack([right_xs, right_ys]) - offset

        pos = np.concatenate([top_pos, bottom_pos])
        pos = rescale_layout(pos, scale=scale) + center
        pos = dict(zip(nodes, pos))
        return pos

    if align == 'horizontal':
        top_ys = np.repeat(height, len(top))
        bottom_ys = np.repeat(0, len(bottom))
        top_xs = np.linspace(0, width, len(top))
        bottom_xs = np.linspace(0, width, len(bottom))

        top_pos = np.column_stack([top_xs, top_ys]) - offset
        bottom_pos = np.column_stack([bottom_xs, bottom_ys]) - offset

        pos = np.concatenate([top_pos, bottom_pos])
        pos = rescale_layout(pos, scale=scale) + center
        pos = dict(zip(nodes, pos))
        return pos

    msg = 'align must be either vertical or horizontal.'
    raise ValueError(msg)
    

def save_connectivity(net1, net2, connectivity1, connectivity2, filename, align='horizontal', with_labels=False):
    num_stages = net1.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from([("1_%d" % i, dict(label=i)) for i in range(num_stages)], task=0, color='xkcd:azure')
    G.add_nodes_from([("2_%d" % i, dict(label=i)) for i in range(num_stages)], task=1, color='xkcd:tomato')

    paths1 = list(zip(*np.nonzero(connectivity1)))
    paths2 = list(zip(*np.nonzero(connectivity2)))
    pos_edges_2_to_1 = [("2_%d" % s, "1_%d" % t, dict(value=net1[t, s])) for t, s in paths1 if net1[t, s] > 0.5]
    pos_edges_1_to_2 = [("1_%d" % s, "2_%d" % t, dict(value=net2[t, s])) for t, s in paths2 if net2[t, s] > 0.5]
    neg_edges_2_to_1 = [("2_%d" % s, "1_%d" % t, dict(value=net1[t, s])) for t, s in paths1 if net1[t, s] <= 0.5]
    neg_edges_1_to_2 = [("1_%d" % s, "2_%d" % t, dict(value=net2[t, s])) for t, s in paths2 if net2[t, s] <= 0.5]
    pos_edges = pos_edges_2_to_1 + pos_edges_1_to_2
    neg_edges = neg_edges_2_to_1 + neg_edges_1_to_2

    G.add_edges_from([("1_%d" % i, "1_%d" % (i + 1)) for i in range(num_stages - 1)], color='xkcd:black')
    G.add_edges_from([("2_%d" % i, "2_%d" % (i + 1)) for i in range(num_stages - 1)], color='xkcd:black')

    top = {n for n, d in G.nodes(data=True) if d['task'] == 0}
    pos = task_layout(G, top, align=align)

    figsize = (1.5, num_stages / 2.) if align == 'vertical' else (num_stages / 2., 1.5)
    fig = plt.figure(num=0, figsize=figsize)
    fig.clf()

    labels = {n: d['label'] for n, d in G.nodes(data=True)}
    node_color = [d['color'] for _, d in G.nodes(data=True)]
    edge_color = [d['color'] for _, _, d in G.edges(data=True)]
    nx.draw(G, pos=pos, labels=labels, node_color=node_color, edge_color=edge_color, with_labels=with_labels)
    nx.draw_networkx_edges(G, pos=pos, edgelist=pos_edges, edge_color='xkcd:violet')
    arcs = nx.draw_networkx_edges(G, pos=pos, edgelist=neg_edges, edge_color='xkcd:silver', alpha=0.3)
    for arc in arcs:
        arc.set_linestyle('dotted')
    plt.savefig(filename)
    img = Image.open(filename)
    return np.array(img).transpose((2, 0, 1))


if __name__ == '__main__':
    from core.models.stagewise_search import vgg_stage_wise_connectivity_matrix_cross_task
    mat1 = vgg_stage_wise_connectivity_matrix_cross_task() * np.random.randn(13, 13) > 0.
    mat2 = vgg_stage_wise_connectivity_matrix_cross_task() * np.random.randn(13, 13) > 0.
    vis_connectivity(mat1.astype(np.float),
                     mat2.astype(np.float),
                     vgg_stage_wise_connectivity_matrix_cross_task(),
                     vgg_stage_wise_connectivity_matrix_cross_task())
    plt.show()
