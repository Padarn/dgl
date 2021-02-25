import time
import dgl
import torch
import numpy as np

from .. import utils
# edge_ids is not supported on cuda
# @utils.skip_if_gpu()
@utils.benchmark('time', timeout=1200)
@utils.parametrize_cpu('graph_name', ['cora', 'livejournal'])
@utils.parametrize_gpu('graph_name', ['cora', 'livejournal'])
@utils.parametrize('format_order', [('coo', 'eid'), ('csr', 'srcdst')])
@utils.parametrize('form', ['uv', 'eid'])
def track_time(graph_name, format_order, form):
    format, order = format_order
    device = utils.get_bench_device()
    graph = utils.get_graph(graph_name, format)
    graph = graph.to(device)
    # dry run
    for i in range(2):
        out = graph.edges(form=form, order=order)

    # timing

    with utils.Timer() as t:
        for i in range(5):
            edges = graph.edges(form=form, order=order)

    return t.elapsed_secs / 5
