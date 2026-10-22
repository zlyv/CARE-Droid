# TASA-Droid

这是 TASA-Droid 的第一阶段工程骨架。当前版本只实现 P0 数据入口与预审计，不训练模型，也不会修改、解压或移动 APK。

## 1. 放到服务器

将整个目录复制到服务器：

```bash
projects/tasa-droid
```

进入工程并创建环境：

```bash
cd projects/tasa-droid
bash scripts/bootstrap_server.sh
```

服务器如果存在自签名证书代理，不要强行关闭 TLS 校验。`scripts/bootstrap_server.sh` 默认不升级 pip，也不联网下载依赖；如果环境没有 PyYAML，会自动使用 JSON 配置。确实需要联网升级 pip 时再执行：

```bash
TASA_UPGRADE_PIP=1 bash scripts/bootstrap_server.sh
```

P0 不依赖 editable 安装。如果确实需要安装 `tasa-droid` 命令，再显式执行：

```bash
TASA_INSTALL_EDITABLE=1 bash scripts/bootstrap_server.sh
```

## 2. 配置数据路径

```bash
cp configs/local_paths.example.yaml configs/local_paths.yaml
```

如果服务器没有 PyYAML，则使用：

```bash
cp configs/local_paths.example.json configs/local_paths.json
```

默认示例把数据根目录写为 `/android/AndroZoo`，并登记两个互相隔离的数据源：

- `legacy_packed_small`：旧的小规模混合/加壳样本，用于管线可行性验证；
- `new_nonpacked_2017_2021`：2017—2021 年新未加壳样本，作为后续正式数据候选。

如果服务器实际根目录不同，只修改 `dataset_root`。如果旧样本不在 `apks`，修改对应 `apk_roots`，不要移动数据。

注意：程序不会根据目录名中的 `b200`、`t100` 猜测良恶标签。只有相对路径明确包含配置中的标签词时才做临时推断；正式标签必须在后续 P0 元数据合并阶段从历史标签清单导入。

## 3. 运行 P0 初始扫描

```bash
tasa-droid p0 scan \
  --config configs/local_paths.yaml \
  --output artifacts/p0-inventory \
  --hash-mode sha256 \
  --workers 8
```

输出：

- `inventory.csv`：APK 路径、来源、声明的加壳状态、大小、SHA256、路径推断年份/标签；
- `unresolved.csv`：无法从相对路径判断年份或标签的记录；
- `exact_duplicates.csv`：SHA256 完全重复组；
- `data_manifest.sha256`：清单文件哈希；
- `p0_audit.json`：当前 P0 状态和阻断原因。

当前程序会始终保持 `formal_training_enabled=false`。只有后续补齐历史标签、可信时间、package/certificate 版本链、近重复 cluster、静态解析覆盖并生成 resolved 配置后，才能解除正式训练门禁。

## 4. 快速检查配置

```bash
tasa-droid p0 validate-config --config configs/local_paths.yaml
```

没有 PyYAML 时把配置路径改为 `configs/local_paths.json`。

## 5. 运行测试

```bash
python -m unittest discover -s tests -v
```

## 下一阶段

在本阶段清单确认后，继续实现：

1. P0 历史标签与 first-seen 时间合并；
2. package/certificate/近重复 cluster；
3. P1 DEX 解析与 API 上下文图；
4. A0、A3-MVP、A3-Full；
5. P0 与 resolved 配置通过后再实现 A4/A5 正式训练。
