import sys
import networkx as nx
from Config import Config
import MetricCollector
import time
from typing import Dict, List
from graph import combine_ns_graphs, graph_weight_ns, graph_weight, GraphIndex, graph_index, get_hg, \
    HeteroWithGraphIndex, combine_graph, calculate_graph_score
from anomaly_detection import get_anomaly_by_df
from MicroCERC.util.utils import time_string_2_timestamp, timestamp_2_time_string, df_time_limit, top_k_node
from anomaly_detection import get_timestamp_index
import pandas as pd
from CausIL import runner
from statistics import print_pr
from graph import NodeType
import os
from python.scores import *
from python.bnutils import *


def run_graph_discovery_instance_sum_MicroCERC(dag_cg, datapath, dataset, dk, score_func):
    g = nx.DiGraph()
    service_graph = []
    fges_time = []
    edge_map = {}

    # def agg_and_rename(data, cols, name):
    #     combine = pd.DataFrame()
    #     for col in cols:
    #         combine = pd.concat([combine, data[col]], axis=0)
    #     combine.columns = [name]
    #     return combine

    # 根据call graph，取上游服务的工作负载指标，取下游服务的延时、异常指标，构建指标集合
    # For each service, construct a graph individually and then merge them
    new_data = pd.DataFrame()
    g_list = []
    for i, service in enumerate(dag_cg.nodes):
        print('===============')
        print("Service: {}".format(service))
        serv_data = pd.DataFrame()
        os_data = pd.DataFrame()
        # svc_data = pd.DataFrame()

        svc_data = dag_cg.nodes[service]['data']
        # filtered_cols = [col for col in data.columns if (str(service) in col)]
        # if t == 'os':
        #     os_data = agg_and_rename(latency_df, [col for col in latency_df.columns if service in col], str(service))
        # elif t == 'instance':
        #     filtered_cols = [col for col in data.columns if service in col and (
        #                 'cpu_usage' in col or 'mem_usage' in col or 'mem_usage_rate' in col or 'fs_usage' in col or 'net_receive' in col)]
        #     serv_data = agg_and_rename(data, filtered_cols, str(service))
        # elif t == 'svc':
        #     svc_data = agg_and_rename(latency_df, [col for col in latency_df.columns if
        #                                            col != 'timestamp' and (str(service) in col[col.index('_') + 1:])],
        #                               str(service))

        # For the child services, get aggregate level data for latency and error
        child = [n[1] for n in dag_cg.out_edges(service)]
        print("Child Services:{}".format(child))
        # child = [c.replace('_', '-', 1) for c in child]
        for ch in child:
            t = dag_cg.nodes[service]['type']
            # agg_cols = []
            # for col in data.columns:
            #     if ch in col and (
            #             'cpu_usage' in col or 'mem_usage' in col or 'mem_usage_rate' in col or 'fs_usage' in col or 'net_receive' in col):
            #         agg_cols.append(col)
            if t == NodeType.NODE.value:
                os_data = pd.concat(
                    [os_data, dag_cg.nodes[ch]['data']], axis=1)
            elif t == NodeType.POD.value:
                serv_data = pd.concat([serv_data, dag_cg.nodes[ch]['data']], axis=1)
            elif t == NodeType.SVC.value:
                svc_data = pd.concat([svc_data, dag_cg.nodes[ch]['data']], axis=1)

        # For parent services, get aggregate level data for workload (aggregate worload = total workload)
        parent = [n[0] for n in dag_cg.in_edges(service)]
        # parent = [p.replace('_', '-', 1) for p in parent]
        print("Parent Services:{}".format(parent))
        for p in parent:
            t = dag_cg.nodes[p]['type']
            # agg_cols = []
            # for col in data.columns:
            #     if p in col and (
            #             'cpu_usage' in col or 'mem_usage' in col or 'mem_usage_rate' in col or 'fs_usage' in col or 'net_receive' in col):
            #         agg_cols.append(col)
            if t == NodeType.NODE.value:
                os_data = pd.concat([serv_data, dag_cg.nodes[p]['data']], axis=1)
            elif t == NodeType.POD.value:
                serv_data = pd.concat([serv_data, dag_cg.nodes[p]['data']], axis=1)
            elif t == NodeType.SVC.value:
                svc_data = pd.concat([svc_data, dag_cg.nodes[p]['data']], axis=1)

        # Pool the data
        # 将服务的n个实例的指标放在同一列，聚集性指标每个实例复制一份
        # new_data = preprocess_data(serv_data)
        new_data = pd.concat([os_data, serv_data, svc_data], axis=1).fillna(0)
        new_data = new_data.loc[:, ~new_data.columns.duplicated()]

        # Use domain knowledge or not
        if dk == 'N':
            bl_edges = None
        else:
            bl_edges = list(pd.read_csv(os.path.join(datapath, dataset, 'prohibit_edges.csv'))[
                                ['edge_source', 'edge_destination']].values)

        # Run FGES
        print('Starting FGES')

        if score_func == 'L':
            st_time = time.time()
            g, dag = runner(g, new_data, None, 1, linear_gaussian_score_iid, bl_edges)
            fges_time.append(time.time() - st_time)
        elif score_func == 'P2':
            st_time = time.time()
            g, dag = runner(g, new_data, None, 1, polynomial_2_gaussian_score_iid, bl_edges)
            fges_time.append(time.time() - st_time)
        elif score_func == 'P3':
            st_time = time.time()
            g, dag = runner(g, new_data, None, 1, polynomial_3_gaussian_score_iid, bl_edges)
            fges_time.append(time.time() - st_time)

        print('Finished FGES')
        for edge in dag.edges():
            edge_map.setdefault(str(edge[0]) + str(edge[1]), 0)
            edge_map[str(edge[0]) + str(edge[1])] += 1

        service_graph.append(dag)
        print('\n')
        g_list.append(g)
    global_g = combine_graph(g_list)
    for node in global_g.nodes:
        # global_g.nodes[node]['type'] = dag_cg.nodes[node]['type']
        # try:
        #     g.nodes[source]['center'] = graph.nodes[source]['center']
        #     g.nodes[destination]['center'] = graph.nodes[destination]['center']
        # except:
        #     pass
        try:
            global_g.nodes[node]['data'] = pd.DataFrame(new_data[node])
        except:
            pass

    return dag_cg, global_g, edge_map, service_graph, fges_time, new_data.T.drop_duplicates().T


if __name__ == "__main__":
    b_dir = '/Users/zhuyuhan/Documents/391-WHU/experiment/researchProject/MicroCERC/data/'
    namespaces = ['bookinfo', 'hipster', 'cloud-sock-shop', 'horsecoder-test']
    config = Config()


    class Simple:
        def __init__(self, global_now_time, global_end_time, label, root_cause, dir):
            self.global_now_time = global_now_time
            self.global_end_time = global_end_time
            self.label = label
            self.root_cause = root_cause
            self.dir = dir


    simples: List[Simple] = [
        Simple(
            1705125240, 1705125960, 'label-details-cpu-load-1', 'details',
            'abnormal/bookinfo/details/bookinfo-details-cpu-1'
        ),
        Simple(
            1705126080, 1705126800, 'label-details-cpu-load-2', 'details',
            'abnormal/bookinfo/details/bookinfo-details-cpu-2'
        ),
        Simple(
            1705126920, 1705127640, 'label-details-cpu-load-3', 'details',
            'abnormal/bookinfo/details/bookinfo-details-cpu-3'
        ),
        Simple(
            1706024880, 1706025660, 'label-details-cpu-load-4', 'details',
            'abnormal/bookinfo/details/bookinfo-details-cpu-4'
        ),
        Simple(
            1706025780, 1706026560, 'label-details-cpu-load-5', 'details',
            'abnormal/bookinfo/details/bookinfo-details-cpu-5'
        ),
        Simple(
            1706026680, 1706027460, 'label-details-cpu-load-6', 'details',
            'abnormal/bookinfo/details/bookinfo-details-cpu-6'
        ),
        Simple(
            1706027580, 1706028360, 'label-details-cpu-load-7', 'details',
            'abnormal/bookinfo/details/bookinfo-details-cpu-7'
        ),
        Simple(
            1706028480, 1706029260, 'label-details-cpu-load-8', 'details',
            'abnormal/bookinfo/details/bookinfo-details-cpu-8'
        ),
        Simple(
            1705129320, 1705130040, 'label-details-mem-load-1', 'details',
            'abnormal/bookinfo/details/bookinfo-details-mem-1'
        ),
        Simple(
            1705130160, 1705130880, 'label-details-mem-load-2', 'details',
            'abnormal/bookinfo/details/bookinfo-details-mem-2'
        ),
        Simple(
            1706029380, 1706030160, 'label-details-mem-load-3', 'details',
            'abnormal/bookinfo/details/bookinfo-details-mem-3'
        ),
        Simple(
            1706030280, 1706031060, 'label-details-mem-load-4', 'details',
            'abnormal/bookinfo/details/bookinfo-details-mem-4'
        ),
        Simple(
            1706031180, 1706031960, 'label-details-mem-load-5', 'details',
            'abnormal/bookinfo/details/bookinfo-details-mem-5'
        ),
        Simple(
            1706032080, 1706032860, 'label-details-mem-load-6', 'details',
            'abnormal/bookinfo/details/bookinfo-details-mem-6'
        ),
        Simple(
            1706032980, 1706033760, 'label-details-mem-load-7', 'details',
            'abnormal/bookinfo/details/bookinfo-details-mem-7'
        ),
        Simple(
            1705131180, 1705131900, 'label-details-net-latency-1', 'details',
            'abnormal/bookinfo/details/bookinfo-details-net-1'
        ),
        Simple(
            1705132020, 1705132740, 'label-details-net-latency-2', 'details',
            'abnormal/bookinfo/details/bookinfo-details-net-2'
        ),
        Simple(
            1705132860, 1705133580, 'label-details-net-latency-3', 'details',
            'abnormal/bookinfo/details/bookinfo-details-net-3'
        ),
        Simple(
            1706033880, 1706034660, 'label-details-net-latency-4', 'details',
            'abnormal/bookinfo/details/bookinfo-details-net-4'
        ),
        Simple(
            1706034780, 1706035560, 'label-details-net-latency-5', 'details',
            'abnormal/bookinfo/details/bookinfo-details-net-5'
        ),
        Simple(
            1706035680, 1706036460, 'label-details-net-latency-6', 'details',
            'abnormal/bookinfo/details/bookinfo-details-net-6'
        )
    ]
    top_k_list = []
    for simple in simples:
        print(simple.label)
        global_now_time = simple.global_now_time
        global_end_time = simple.global_end_time
        now = int(time.time())
        if global_now_time > now:
            sys.exit("begin time is after now time")
        if global_end_time > now:
            global_end_time = now

        folder = '.'
        graphs_time_window: Dict[str, Dict[str, nx.DiGraph]] = {}
        base_dir = b_dir + str(simple.dir)
        base_output_dir = 'result'
        time_pair_list = []
        time_pair_index = {}
        now_time = global_now_time
        end_time = global_end_time
        while now_time < end_time:
            config.start = int(round(now_time))
            config.end = int(round(now_time + config.duration))
            if config.end > end_time:
                config.end = end_time
            now_time += config.duration + config.step
            time_pair_list.append((config.start, config.end))
            df = pd.read_csv(base_dir + '/hipster/' + 'latency.csv')
            df = df_time_limit(df, config.start, config.end)
            df_time_index, df_index_time = get_timestamp_index(df)
            time_pair_index[(config.start, config.end)] = df_time_index
        # 获取拓扑有变动的时间窗口
        topology_change_time_window_list = []
        for ns in namespaces:
            config.namespace = ns
            data_folder = base_dir + '/' + config.namespace
            time_change_ns = [timestamp_2_time_string(global_now_time), timestamp_2_time_string(global_end_time)]
            topology_change_time_window_list.extend(time_change_ns)
        topology_change_time_window_list = sorted(list(set(topology_change_time_window_list)))
        for ns in namespaces:
            config.namespace = ns
            config.svcs.clear()
            config.pods.clear()
            count = 1
            data_folder = base_dir + '/' + config.namespace
            for time_pair in time_pair_list:
                config.start = time_pair[0]
                config.end = time_pair[1]
                print('第' + str(count) + '次获取 [' + config.namespace + '] 数据')
                graphs_ns_time_window = MetricCollector.collect_and_build_graphs(config, data_folder,
                                                                                 topology_change_time_window_list,
                                                                                 config.window_size, config.collect)
                graph_time_key = str(time_pair[0]) + '-' + str(time_pair[1])
                if graph_time_key not in graphs_time_window:
                    graphs_time_window[graph_time_key] = graphs_ns_time_window
                else:
                    graphs_time_window[graph_time_key].update(graphs_ns_time_window)
                config.pods.clear()
                count += 1
        config.start = global_now_time
        config.end = global_end_time
        MetricCollector.collect_node(config, base_dir + '/node', config.collect)
        # 非云边基于指标异常检测
        anomalies = {}
        anomalies_index = {}
        anomaly_time_series = {}
        for time_pair in time_pair_list:
            time_key = str(time_pair[0]) + '-' + str(time_pair[1])
            for ns in namespaces:
                data_folder = base_dir + '/' + ns
                anomaly_list = anomalies.get(time_key, [])
                anomalies_ns, anomaly_time_series_index = get_anomaly_by_df(base_output_dir, data_folder, simple.label,
                                                                            time_pair[0], time_pair[1])
                anomaly_list.extend(anomalies_ns)
                anomaly_list = list(set(anomaly_list))
                anomalies[time_key] = anomaly_list
                anomaly_time_series_list = anomaly_time_series.get(time_key, {})
                anomaly_time_series_list = {**anomaly_time_series_list, **anomaly_time_series_index}
                anomaly_time_series[time_key] = anomaly_time_series_list
            anomalies_index[time_key] = {a: i for i, a in enumerate(anomalies[time_key])}
        # 赋权ns子图
        for time_window in graphs_time_window:
            anomaly_index = anomalies_index[time_window]
            t_index_time_window = time_pair_index[(int(time_window.split('-')[0]), int(time_window.split('-')[1]))]
            for graph_time_window in graphs_time_window[time_window]:
                graph: nx.DiGraph = graphs_time_window[time_window][graph_time_window]
                begin_t = graph_time_window.split('-')[0]
                end_t = graph_time_window.split('-')[1]
                ns = graph_time_window[graph_time_window.index(end_t) + len(end_t) + 1:]
                graph_weight_ns(begin_t, end_t, graph, base_dir, ns)
            # 合并混合部署图
            graphs_combine: Dict[str, nx.DiGraph] = combine_ns_graphs(graphs_time_window[time_window])
            graphs_anomaly_time_series_index = {}
            graphs_anomaly_time_series_index_map = {}
            graphs_index_time_map = {}
            for time_combine in graphs_combine:
                graph_index_time_map = {}
                graph = graphs_combine[time_combine]
                begin_t = time_combine.split('-')[0]
                end_t = time_combine.split('-')[1]


                def get_t(begin_t, t_index_time_window):
                    index = len(t_index_time_window.keys()) - 1
                    for i, t in enumerate(sorted(t_index_time_window.keys())):
                        if int(begin_t) <= time_string_2_timestamp(t):
                            index = i
                            break
                    return index


                graph_weight(begin_t, end_t, graph, base_dir)
                # graph dump
                # graph_dump(graph, base_dir, begin_t + '-' + end_t)
                for t in t_index_time_window:
                    if int(begin_t) <= time_string_2_timestamp(t) <= int(end_t):
                        graph_index_time_map[t_index_time_window[t] - get_t(begin_t, t_index_time_window)] = t
                graphs_index_time_map[time_combine] = graph_index_time_map
                # 赋值异常时序索引
                anomalies_series_time_window = anomaly_time_series[time_window]
                a_t_index = []
                anomaly_time_series_index = {}
                for anomaly in anomalies_series_time_window:
                    anomaly_t_index = []
                    anomaly_series_time_window = anomalies_series_time_window[anomaly]
                    anomaly_series_time_window = [time_string_2_timestamp(a) for a in anomaly_series_time_window]
                    if max(anomaly_series_time_window) < int(begin_t) or min(anomaly_series_time_window) > int(end_t):
                        continue
                    for t in anomaly_series_time_window:
                        if int(begin_t) <= t <= int(end_t):
                            a_t_index.append(
                                t_index_time_window[timestamp_2_time_string(t)] - get_t(begin_t, t_index_time_window))
                            anomaly_t_index.append(
                                t_index_time_window[timestamp_2_time_string(t)] - get_t(begin_t, t_index_time_window))
                    anomaly_time_series_index[anomaly] = anomaly_t_index
                a_t_index = list(set(a_t_index))
                graphs_anomaly_time_series_index[time_combine] = a_t_index
                graphs_anomaly_time_series_index_map[time_combine] = anomaly_time_series_index

            graphs_combine_index: Dict[str, GraphIndex] = {t_index: graph_index(graphs_combine[t_index]) for t_index in
                                                           graphs_combine}
            # 转化为dgl构建图网络栈
            hetero_graphs_combine: Dict[str, HeteroWithGraphIndex] = get_hg(graphs_combine, graphs_combine_index,
                                                                            anomalies,
                                                                            graphs_anomaly_time_series_index,
                                                                            graphs_anomaly_time_series_index_map
                                                                            , graphs_index_time_map)
            # train(simple.label, simple.root_cause, anomaly_index, hetero_graphs_combine, base_output_dir, config.train, rnn=config.rnn_type,
            #       attention=config.attention)
            _, g, _, _, _, _ = run_graph_discovery_instance_sum_MicroCERC(next(iter(graphs_combine.values())), None,
                                                                          None, 'N', 'P2')
            pagerank_scores = nx.pagerank(g, alpha=0.85)
            sorted_dict_node_pagerank = dict(sorted(pagerank_scores.items(), key=lambda item: item[1], reverse=True))
            with open(base_output_dir + '/result-' + simple.label + '.log', "a") as output_file:
                top_k = top_k_node(sorted_dict_node_pagerank, simple.root_cause, output_file)
                top_k_list.append(top_k)
    print_pr(top_k_list)
