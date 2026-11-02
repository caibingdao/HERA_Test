# maps/ — 测试地图与划分

| 文件/目录 | 内容 |
|---|---|
| `map_split_seed_42.json` | 划分清单；训练 4530 / 测试 1133；split seed 42；包含 5663 张地图的 SHA256 和分组信息 |
| `test_maps.csv` | 1133 张测试地图的 `map_id`、SHA256、原始来源路径 |
| `test_maps/` | 1133 张测试地图 PNG，文件名与 `test_maps.csv` 的 `map_id` 一致 |

## 校验

```bash
# 在 paper/ 目录执行
python3 - <<'PY'
import csv, hashlib, os
rows=list(csv.DictReader(open('maps/test_maps.csv', newline='')))
bad=[]
for r in rows:
    p=os.path.join('maps/test_maps', r['map_id'])
    if not os.path.exists(p) or hashlib.sha256(open(p,'rb').read()).hexdigest()!=r['sha256']:
        bad.append(r['map_id'])
print('checked', len(rows), 'bad', len(bad), bad[:10])
PY
```

## 关键值

- `split_seed`: 42
- `test_maps`: 1133
- `train_maps`: 4530
- `map_split_seed_42.json` SHA256: `8397d4d1a3b1e6f57fa604b23f59284352a2e6dea6686f55ff62d0f531fcf0c0`

## 未纳入

训练地图 4530 张未复制（约 43.8 MB）；本分支只保留 1133 张测试地图，以便复现评测。需要训练地图时，按 `map_split_seed_42.json` 的 `train_maps` 从上游地图集补齐。
