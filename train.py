import numpy as np
import pandas as pd
import torch
import time
import json
import os
from tqdm import tqdm
from torch_geometric.data import Data
import argparse

from models.Model import CIKGRec
from utils import MyLoader, init_seed, generate_kg_batch
from evaluate import eval_model
from prettytable import PrettyTable
from torch_geometric.utils import coalesce


def main():
    def parse_args():
        parser = argparse.ArgumentParser(description="Run CIKGRec.")
        parser.add_argument('--seed', type=int, default=2024, help='Random seed.')
        parser.add_argument('--dataset', nargs='?', default='ml1m',
                            help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
        parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
        return parser.parse_args()

    args = parse_args()
    dataset = args.dataset
    config = json.load(open(f'./config/{dataset}.json'))
    config['device'] = f'cuda:{args.gpu_id}'
    seed = args.seed
    Ks = config['Ks']

    init_seed(seed, True)
    print('-' * 100)
    for k, v in config.items():
        print(f'{k}: {v}')
    print('-' * 100)

    loader = MyLoader(config)

    # ----------------------
    # logging & checkpointing
    # ----------------------
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    run_tag = time.strftime('%Y%m%d-%H%M%S')
    log_path = os.path.join('logs', f"train_{dataset}_seed{seed}_{run_tag}.txt")
    ckpt_path = os.path.join('checkpoints', f"best_{dataset}_seed{seed}_{run_tag}.pt")
    best_metrics_path = os.path.join('checkpoints', f"best_{dataset}_seed{seed}_{run_tag}.json")

    print(f"[log] epoch-wise logs -> {log_path}")
    print(f"[ckpt] best checkpoint -> {ckpt_path}")
    print(f"[ckpt] best metrics    -> {best_metrics_path}")

    # ----------------------
    # build graphs
    # ----------------------
    src = loader.train.loc[:, 'userid'].to_list() + loader.train.loc[:, 'itemid'].to_list()
    tgt = loader.train.loc[:, 'itemid'].to_list() + loader.train.loc[:, 'userid'].to_list()
    src_cf = loader.train.loc[:, 'userid'].to_list() + loader.train.loc[:, 'itemid'].to_list()
    tgt_cf = loader.train.loc[:, 'itemid'].to_list() + loader.train.loc[:, 'userid'].to_list()

    edge_index = [src, tgt]

    # kg
    src_k = loader.kg_org['head'].to_list() + loader.kg_org['tail'].to_list()
    tgt_k = loader.kg_org['tail'].to_list() + loader.kg_org['head'].to_list()
    edge_index[0].extend(src_k)
    edge_index[1].extend(tgt_k)

    # user interest graph
    src_in = loader.kg_interest['uid'].to_list() + loader.kg_interest['interest'].to_list()
    tgt_in = loader.kg_interest['interest'].to_list() + loader.kg_interest['uid'].to_list()
    edge_index[0].extend(src_in)
    edge_index[1].extend(tgt_in)

    edge_index_ig = [src_in + src_cf, tgt_in + src_cf]
    edge_index_ig = torch.LongTensor(edge_index_ig)
    edge_index_ig = coalesce(edge_index_ig)
    graph_ig = Data(edge_index=edge_index_ig.contiguous()).to(config['device'])
    print(f'Is ig no duplicate edge: {graph_ig.is_coalesced()}')

    edge_index_kg = [src_k + src_cf, tgt_k + src_cf]
    edge_index_kg = torch.LongTensor(edge_index_kg)
    edge_index_kg = coalesce(edge_index_kg)
    graph_kg = Data(edge_index=edge_index_kg.contiguous()).to(config['device'])
    print(f'Is kg no duplicate edge: {graph_kg.is_coalesced()}')

    edge_index = torch.LongTensor(edge_index)
    edge_index = coalesce(edge_index)
    graph = Data(edge_index=edge_index.contiguous()).to(config['device'])
    print(f'Is cikg no duplicate edge: {graph.is_coalesced()}')

    # pure cf graph
    edge_index_cf = torch.LongTensor([src, tgt])
    edge_index_cf = coalesce(edge_index_cf)
    graph_cf = Data(edge_index=edge_index_cf.contiguous()).to(config['device'])
    print(f'Is cg no duplicate edge: {graph_cf.is_coalesced()}')

    # ----------------------
    # model & optimizer
    # ----------------------
    model = CIKGRec(config, graph.edge_index).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    kg_optimizer = None
    if config['use_kge']:
        kg_optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_kg'])

    train_loader, _ = loader.get_cf_loader(bs=config['batch_size'])

    # ----------------------
    # training
    # ----------------------
    best_score = 0
    patience = config['patience']
    best_performance = []

    for epoch in range(config['num_epoch']):
        loss_list = []
        print(f'epoch {epoch + 1} start!')
        model.train()
        start = time.time()

        eval_result = None
        score = None

        # update mask rate (dynamic mask rate)
        model.gmae_p = model.calc_mask_rate(epoch)

        # train cf
        for data in train_loader:
            user, pos, neg = data
            user, pos, neg = user.to(config['device']), pos.to(config['device']), neg.to(config['device'])

            optimizer.zero_grad()
            loss_rec = model(user.squeeze(), pos.squeeze(), neg.squeeze())
            loss_cross_domain_contrastive = model.cross_domain_contrastive_loss(
                user.squeeze(), pos.squeeze(),
                graph_ig.edge_index, graph_kg.edge_index, graph_cf.edge_index
            )
            loss_interest_recon = model.interest_recon_loss(graph.edge_index)
            loss = loss_rec + loss_cross_domain_contrastive + loss_interest_recon
            loss.backward()
            optimizer.step()

            loss_list.append(loss.detach().cpu().numpy())

        sum_loss = np.sum(loss_list) / len(loss_list)

        kg_total_loss = None
        n_kg_batch = None
        if config['use_kge']:
            # train kg
            kg_total_loss = 0.0
            n_kg_batch = config['n_triplets'] // 4096
            for _iter in range(1, n_kg_batch + 1):
                # entity contains interests, but at the transE stage, neg sample space shouldn't contains interests
                kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = generate_kg_batch(
                    loader.kg_dict, 4096, config['entities'] - config['interests'], 0
                )
                kg_batch_head = kg_batch_head.to(config['device'])
                kg_batch_relation = kg_batch_relation.to(config['device'])
                kg_batch_pos_tail = kg_batch_pos_tail.to(config['device'])
                kg_batch_neg_tail = kg_batch_neg_tail.to(config['device'])

                kg_batch_loss = model.get_kg_loss(
                    kg_batch_head, kg_batch_pos_tail, kg_batch_neg_tail, kg_batch_relation
                )

                kg_optimizer.zero_grad()
                kg_batch_loss.backward()
                kg_optimizer.step()
                kg_total_loss += kg_batch_loss.item()

        # eval
        if (epoch + 1) % config['eval_interval'] == 0:
            eval_result = eval_model(model, loader, 'test')
            score = eval_result['recall'][1] + eval_result['ndcg'][1]

            if score > best_score:
                best_performance = []
                best_score = score
                patience = config['patience']

                # ---- save best checkpoint & metrics snapshot (修复 JSON 序列化问题) ----
                # 转换 NumPy ndarray 为 Python 原生 list
                def to_serializable(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.generic):  # 处理 np.int64, np.float32 等标量
                        return obj.item()
                    elif isinstance(obj, (list, tuple)):
                        return [to_serializable(x) for x in obj]
                    elif isinstance(obj, dict):
                        return {k: to_serializable(v) for k, v in obj.items()}
                    else:
                        return obj

                # 保存 checkpoint (torch.save 支持 ndarray，但为统一也转换)
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "best_score": float(best_score),
                        "dataset": dataset,
                        "seed": seed,
                        "run_tag": run_tag,
                        "Ks": to_serializable(Ks),
                        "config": config,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "kg_optimizer_state_dict": (kg_optimizer.state_dict() if kg_optimizer is not None else None),
                        "eval_result": {
                            "recall": to_serializable(eval_result["recall"]),
                            "ndcg": to_serializable(eval_result["ndcg"]),
                            "precision": to_serializable(eval_result["precision"]),
                            "hit_ratio": to_serializable(eval_result["hit_ratio"]),
                        },
                    },
                    ckpt_path
                )

                # 保存纯 JSON 指标文件（必须转换）
                metrics_to_save = {
                    "epoch": epoch + 1,
                    "best_score": float(best_score),
                    "Ks": to_serializable(Ks),
                    "recall": to_serializable(eval_result["recall"]),
                    "ndcg": to_serializable(eval_result["ndcg"]),
                    "precision": to_serializable(eval_result["precision"]),
                    "hit_ratio": to_serializable(eval_result["hit_ratio"]),
                }

                with open(best_metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics_to_save, f, ensure_ascii=False, indent=2)

                print(f"[save] best checkpoint saved: {ckpt_path}")
                print(f"[save] best metrics saved:    {best_metrics_path}")

                for i, k in enumerate(Ks):
                    table = PrettyTable([f'recall@{k}', f'ndcg@{k}', f'precision@{k}', f'hit_ratio@{k}'])
                    table.add_row([
                        round(eval_result['recall'][i], 4),
                        round(eval_result['ndcg'][i], 4),
                        round(eval_result['precision'][i], 4),
                        round(eval_result['hit_ratio'][i], 4)
                    ])
                    print(table)
                    best_performance.append(table)
            else:
                patience -= 1
                print(f'patience: {patience}')

        end = time.time()
        epoch_time_s = end - start

        kg_avg_loss = None
        if config['use_kge']:
            kg_avg_loss = kg_total_loss / (n_kg_batch + 1)
            print(f'kg loss:{round(kg_avg_loss, 4)}')

        print('epoch loss: ', round(sum_loss, 4))
        print('current mask rate:', round(model.gmae_p, 4))

        # one line per epoch (no header)
        recall_str = ndcg_str = precision_str = hit_str = 'NA'
        if eval_result is not None:
            # 日志中直接转换为字符串（避免序列化问题）
            recall_str = ','.join([str(round(float(x), 6)) for x in eval_result['recall']])
            ndcg_str = ','.join([str(round(float(x), 6)) for x in eval_result['ndcg']])
            precision_str = ','.join([str(round(float(x), 6)) for x in eval_result['precision']])
            hit_str = ','.join([str(round(float(x), 6)) for x in eval_result['hit_ratio']])

        with open(log_path, mode='a', encoding='utf-8') as f:
            f.write(
                f"epoch={epoch + 1}\t"
                f"epoch_loss={sum_loss:.6f}\t"
                f"kg_loss={(kg_avg_loss if kg_avg_loss is not None else 'NA')}\t"
                f"mask_rate={float(model.gmae_p):.6f}\t"
                f"eval_score={(score if score is not None else 'NA')}\t"
                f"best_score={best_score}\t"
                f"patience={patience}\t"
                f"time_s={epoch_time_s:.2f}\t"
                f"Ks={Ks}\t"
                f"recall={recall_str}\t"
                f"ndcg={ndcg_str}\t"
                f"precision={precision_str}\t"
                f"hit_ratio={hit_str}"
                "\n"
            )

        if patience <= 0:
            break

        print(f'train time {round(epoch_time_s)}s')
        print('-' * 90)

    print('Best testset performance:')
    for table in best_performance:
        print(table)


if __name__ == "__main__":
    main()
