# 轨迹分类 v2：处理规则与字段说明

本说明对应分类器 `2026-11-09.1`、Prompt `v005-toolcall-scenario-capability-subject` 和输出格式 `trajfoundry-classification-v2`。
分类目录来自[智能体数据分类](https://aicarrier.feishu.cn/wiki/U9qfw9wJzilQNykzjvscCOfWnJn)，本次读取的文档修订号为 130。
代码中的 `src/trajfoundry/classification/taxonomy.json` 是发布时固定的目录快照；运行中不访问飞书，也不会随文档编辑自动改变。

## 一、哪些轨迹会分类

1. 读取标准化字段 `total_tool_calls`。该字段按节点统计，因此也逐层检查 `sub_agent_trajectory` 中的同名字段。
2. 根节点或任意子节点的值大于 0：调用分类模型，生成场景、能力、学科标签。
3. 所有节点的值都为 0：保留完整轨迹，追加 `classification.status="skipped"`，不调用分类模型，不计为失败。
4. 字段缺失、不是非负整数或输入校验不通过：作为输入错误终止作业，不把未知值当作 0。

不扫描消息文本推断调用，不因 `tools` 中有工具定义就认为发生了调用。`server_tool_calls` 是标准化格式中的另一类信息，不额外加入本次基于 `total_tool_calls` 的筛选条件。

筛选范围包含子轨迹，但模型输入仍只使用根轨迹自身经过清洗的用户请求；不发送 assistant、tool、system、developer 或子轨迹消息。输出保留原轨迹内容。旧版输入没有 `metadata.sub_session_id` 时仍按原有命名规则补齐，子轨迹继承该值；已有编号保留。子轨迹不另外添加分类对象。

## 二、每条轨迹的 classification

成功示例（这里只展示追加的对象；真实输出还包含完整标准化轨迹）：

```json
{
  "classification": {
    "status": "accepted",
    "scenario_labels": ["IT互联网"],
    "capability_labels": ["代码与程序化操作", "评估与验证"],
    "subject_labels": ["计算机与人工智能"],
    "classifier_revision": "2026-11-09.1",
    "prompt_version": "v005-toolcall-scenario-capability-subject",
    "taxonomy_sha256": "<64位小写SHA-256>"
  }
}
```

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `status` | 字符串 | `accepted` 分类成功；`skipped` 按规则跳过；`failed` 单条分类失败。 |
| `scenario_labels` | 字符串数组 | 从 90 个场景中选且只选 1 个，名称严格与目录一致。 |
| `capability_labels` | 字符串数组 | 从 10 个能力中选 1–2 个，不重复，以任务的主要能力为先。 |
| `subject_labels` | 字符串数组 | 从 25 个选项中选择一个或多个有直接依据的学科，不重复。没有明确学科时为 `["通用（非学科）"]`，此项不能与其他学科同时出现。文档没有规定具体学科最多选两项，本实现不额外加这个限制。 |
| `classifier_revision` | 字符串 | 分类实现版本，用于区分不同时期的处理规则。只在成功对象中出现。 |
| `prompt_version` | 字符串 | 提示词版本。只在成功对象中出现。 |
| `taxonomy_sha256` | 字符串 | 完整分类目录的内容摘要，包含三类标签、描述及典型任务。只在成功对象中出现。 |
| `skip_reason` | 字符串 | 只在跳过对象中出现；当前唯一值为 `no_toolcall`。 |
| `reason` | 字符串 | 只在失败对象中出现；安全的错误代码，不含模型原始回复、用户内容或密钥。 |

三种对象使用互斥结构。跳过和失败对象都只有两个字段：

```json
{
  "classification": {
    "status": "skipped",
    "skip_reason": "no_toolcall"
  }
}
```

```json
{
  "classification": {
    "status": "failed",
    "reason": "network_retry_exhausted"
  }
}
```

不再重复保存 `model_label`、`harness_label`，使用轨迹顶层的 `model`、`harness`。每条结果也不保存字符数量、上下文截断标记或多组配置指纹；运行配置与汇总信息集中在 manifest 中。

常见单条失败代码：

| `reason` | 含义 |
| --- | --- |
| `invalid_model_output_json` | 重试后仍未得到合法 JSON。 |
| `invalid_model_output_fields` | 回复字段缺失，包含旧版字段或多余字段。 |
| `invalid_model_output_scenario_labels` | 场景名称、数量或类型不符合目录规则。 |
| `invalid_model_output_capability_labels` | 能力名称、数量、类型或重复情况不符合规则。 |
| `invalid_model_output_subject_labels` | 学科名称、数量、类型、重复情况或“通用”互斥规则不符合要求。 |
| `network_retry_exhausted` | 连接、读取或超时错误耗尽重试次数。 |
| `rate_limit_exhausted` | HTTP 429 限流耗尽重试次数。 |
| `service_retry_exhausted` | HTTP 5xx 耗尽重试次数。 |
| `context_limit_exhausted` | 缩减输入后仍然超出网关上下文限制。 |
| `classifier_http_<状态码>` | 其他不适合重试的单条 HTTP 错误，例如 `classifier_http_400`。 |
| `invalid_api_response` | 网关响应结构无法解析或没有可用的文本回复。 |

认证或配置错误（如 401、403、持续 404、非上下文限制的 422）会让作业失败，候选结果不会开始覆盖已经发布的输出；不会为整批记录伪造分类成功。

## 三、manifest.json 的全部字段

manifest 是本次输出的文件清单与运行说明，不是一条轨迹。一个输入分区对应一个 manifest，成功、失败、跳过的完整轨迹均列入同一个文件清单。

### 顶层字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `schema_version` | 字符串 | 固定为 `trajfoundry-classification-v2`，标识本说明中的结构。 |
| `created_at` | 字符串 | 本次 manifest 生成的 UTC 时间，ISO 8601 格式。 |
| `input_root` | 字符串 | 输入标准化分区的 S3 URI，该目录必须直接包含 `manifest.json`。 |
| `input_manifest_sha256` | 字符串 | 输入 manifest 原始字节的 SHA-256，定位本次输入快照。 |
| `sessionless_window_hours` | 整数，可选 | 原样继承输入 manifest 中非零的无 session 分组窗口；未启用时省略。不是分类任务重新切分轨迹的参数。 |
| `classifier_revision` | 字符串 | 本次分类实现版本，对全部输出有效，也覆盖跳过和失败记录。 |
| `prompt_version` | 字符串 | 本次提示词版本。 |
| `taxonomy_sha256` | 字符串 | 三类分类目录快照的 SHA-256；对 JSON 对象的键排序后计算，排版和对象键顺序变化不会改变摘要。 |
| `prompt_sha256` | 字符串 | 实际系统提示词 UTF-8 字节的 SHA-256，用于确认提示词是否完全一致。 |
| `config_hash` | 字符串 | 分类版本、Prompt 版本、目录摘要、系统提示词摘要、分类策略和 `config` 中模型请求设置的组合摘要。用于隔离缓存；不包含密钥、输入 manifest 或 `max_workers`。输入 manifest 的摘要另行参与缓存键。 |
| `classification_policy` | 对象 | 分类筛选及标签选择规则，所有子字段见下表。 |
| `config` | 对象 | 本次请求与并发设置，所有子字段见下表。不保存凭据。 |
| `counts` | 对象 | 轨迹处理结果汇总，所有子字段见下表。 |
| `files` | 对象数组 | 本次有效输出文件列表，按 `path` 排序。所有子字段见下表。 |

相较旧版，不再输出 `catalog_sha256`、`policy_sha256`、`strategy_fingerprint`。目录使用一个 `taxonomy_sha256`，完整配置使用一个 `config_hash` 即可。

### classification_policy 子字段

| 字段 | 当前值 | 含义 |
| --- | --- | --- |
| `eligibility` | `any_node_total_tool_calls_gt_zero` | 根节点或任意子节点 `total_tool_calls > 0` 才分类。 |
| `scenario_count` | `1` | 必须选择一个场景。 |
| `capability_min` | `1` | 最少选择一个能力。 |
| `capability_max` | `2` | 最多选择两个能力。 |
| `subject_min` | `1` | 最少选择一个学科选项；每个选项至多出现一次。 |
| `generic_subject` | `通用（非学科）` | 没有明确学科归属时使用的选项。 |
| `generic_subject_exclusive` | `true` | 通用选项只能单独使用。 |

### config 子字段

| 字段 | 默认/当前设置 | 含义 |
| --- | --- | --- |
| `model` | `deepseek-v4.1-flash` | 执行分类的模型名，与轨迹顶层 `model` 所表示的原始轨迹模型不同。 |
| `api_url` | 调度节点配置的完整 Chat Completions URL | 分类模型服务地址，不包含密钥、查询参数或账户信息。 |
| `max_workers` | `20` | 单个作业最多同时发起分类请求的线程数。多个调度实例同时运行会叠加并发；补数建议串行。 |
| `request_timeout_seconds` | `180.0` | 每次 HTTP 请求/阻塞读取的超时设置，不是整条轨迹或整个作业的总耗时上限。 |
| `max_retries` | `20` | 单次模型调用发生可重试 HTTP/网络错误时的额外重试次数，含首次最多尝试 21 次。 |
| `semantic_retries` | `2` | 回复标签/JSON 校验失败时额外调用模型的次数，含首次最多进行 3 次语义尝试。上下文缩减另有独立预算。 |
| `max_context_chars` | `600000` | 序列化用户请求 JSON 的字符预算，不是 token 数，也不包含系统提示词。 |
| `max_output_tokens` | `4096` | 单次模型回复的 token 上限。 |
| `input_projection` | `user_only_root_v2` | 说明模型读取的是经过清洗的根轨迹用户请求。 |
| `input_projection_version` | `2.0.0` | 用户请求抽取、清洗规则的版本。 |
| `truncation_strategy` | `user_turn_head_tail_v1` | 输入超过字符预算时使用的裁剪策略。 |
| `context_reduction_factor` | `0.75` | 网关报告上下文超限后，下一次字符预算乘以该比例；预算下限为 1024。 |
| `max_context_reductions` | `5` | 网关上下文超限后最多缩减输入的次数。 |
| `temperature` | `0` | 模型采样温度。 |
| `thinking` | `disabled` | 请求关闭模型的思考模式。 |
| `response_format` | `json_object` | 请求模型输出 JSON 对象。 |

本地模拟测试使用假客户端时，`api_url` 可为 `injected-client`；真实调度使用实际配置的 API URL。

### counts 子字段

| 字段 | 含义 |
| --- | --- |
| `input_trajectories` | 输入 manifest 中有效的顶层轨迹数量，包含标准化阶段接受和隔离的完整轨迹；嵌套子轨迹不另计一条。隔离原始记录不是完整轨迹，不参加分类。 |
| `classified` | 本次输出中 `status="accepted"` 的数量，包含缓存命中。 |
| `failed` | 本次输出中 `status="failed"` 的数量；完整轨迹仍写出。 |
| `skipped_no_toolcall` | 本次输出中因没有工具调用而跳过的数量。 |
| `cache_hits` | 复用了同一输入和分类配置下成功结果的数量，是 `classified` 的子集。跳过记录不进入模型结果缓存。 |
| `context_truncated` | 实际输入发生字符预算裁剪或网关上下文缩减的轨迹数量；包括分类失败的记录，命中缓存时保留该结果最初的截断信息。跳过记录不计入。 |

计数应满足：

```text
input_trajectories = classified + failed + skipped_no_toolcall
0 <= cache_hits <= classified
0 <= context_truncated <= classified + failed
```

单条分类失败并不意味着作业没有发布结果。调度日志出现 `validation=True` 表示输入校验、分类输出契约与已上传轨迹/lineage 文件回读校验通过；是否有单条分类失败应查看 `failed`。

### files 数组中每个对象

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `path` | 字符串 | 相对输出根目录的路径。每条顶层轨迹一个 `.jsonl` 文件且只有一行；另有 `lineage.jsonl`。读取时使用清单中的路径。 |
| `sha256` | 字符串 | 输出文件原始字节的 SHA-256，包含结尾换行。 |
| `bytes` | 非负整数 | 文件的字节数。 |
| `kind` | 字符串 | `trajectory` 表示轨迹；`lineage` 表示从标准化输入原样复制的来源关系文件。 |
| `classification_status` | 字符串，仅 trajectory | 对应轨迹的 `accepted`、`skipped` 或 `failed` 状态。lineage 文件没有此字段。 |

manifest 不把自身列入 `files`，也不对自身计算嵌套摘要。完整读取流程是：读取 manifest → 按清单读文件 → 核验字节数和 SHA-256 → 消费轨迹。

## 四、缓存、发布和调度

分类缓存位于 Worker 本地 `workspace_parent/.trajfoundry-label-cache/`，以轨迹内容、输入 manifest 摘要和分类配置摘要共同生成缓存键。只缓存校验成功的标签；失败记录在下次运行重试，跳过记录直接重新检查。截断诊断仅保存在本地缓存中，用于 manifest 汇总，不会重新出现在轨迹分类对象里。更换 Worker 可能没有本地缓存，应按重新调用模型估算耗时。

所有候选行在本地完成后才开始上传；轨迹和 lineage 回读校验后，最后写 manifest。旧输出未列入新清单的顶层 JSONL 会在新 manifest 发布后清理。上传过程中同名文件可能先于新 manifest 改变，读取方应核验文件摘要，并禁止两个作业同时写同一输出分区。输入输出目录不能重叠。

本次新结构使用新的分类输出版本目录，旧分类数据保留。旧标准化 v3 分片输入和 v4 平铺输入继续支持；旧的八字段分类目录数组已替换为三类目录格式，不再用作新分类器的输入。

调度节点模板见 [dolphinscheduler_label_s3_node.py](../examples/dolphinscheduler_label_s3_node.py)。使用 Python 节点，共享 Python 环境，Worker 本地 `/tmp`，20 并发、20 次额外网络重试、180 秒请求超时。模型调用凭据沿用 Worker 环境中的 `CLASSIFIER_API_KEY`；S3 凭据沿用现有受保护文件，节点中不填写密钥。

| 节点参数 | 说明 |
| --- | --- |
| `input_uri` | 一个直接包含标准化 `manifest.json` 的分区 URI，可在平台上用业务日期替换，不要填写没有根 manifest 的跨日期总目录。 |
| `output_uri` | 对应的新分类输出分区 URI，建议使用新的版本目录。 |
| `endpoint_url` | S3 服务地址。 |
| `classifier_api_url` | 沿用之前 DeepSeek 服务的完整 `/chat/completions` 地址。 |

配置完成后先运行一个较小分区，检查模型可用性、三个标签名称、`failed`、`skipped_no_toolcall` 和计数等式，再执行批量补数。没有工具调用的轨迹也会出现在输出清单中。
