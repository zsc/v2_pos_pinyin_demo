# POS+NER-Aware 多音字消歧 → 带声调拼音 转换系统 SPEC

> 目标：把输入文本中的汉字转换为**带声调的拼音（tone marks）**，并用 **LLM 分词 + POS+NER 标注** 与一套 **pos+ner-aware 多音字消歧规则（context-sensitive grammar 子集）** 做消歧；对不确定处进行 **LLM double check**，必要时 **交互式让用户手动选读音**，并把用户标注沉淀为 **overrides.json** 规则以便持续扩展。
> 约束：**中文词内无空格**（但原始文本可能包含空格/换行/混排内容），**不得改写受保护片段内容**（英文/数字/URL/数学符号/标点字符保持原样，顺序不变）。

---

## 0. 非目标与边界

### 非目标

* 不追求覆盖全部汉语读音现象；该系统仅是“多音字消歧规则子集”。
* 不试图处理“故意读错字”“方言读法”“戏谑读法”等非规范用法。
* 不要求生成可逆的对齐标注（但应提供偏移与日志用于调试与后续扩展）。

### 边界与假设

* 输出拼音格式：**带声调（tone marks）**、**词内无空格**（如 `yínháng`）、**词间有空格**（如 `yínháng hángzhǎng`）。
* 对混排：英文单词、数字、网址 URL、数学符号、标点、emoji 等 **原样保留**；必要时在“拼音词”与相邻的非汉字 *word-like* 片段（如 latin/number/url）之间插入空格，避免粘连（如 `xìshuō OpenAI de API`）。
* 依赖 LLM 进行分词 + POS+NER；但系统必须能在 LLM 输出不合法/不一致时降级（fallback）。

---

## 1. 总体流程

```
raw input string
  -> Preprocess: 保护混排片段（URL/latin/number/punct/symbol/space 等），切分为 spans
  -> LLM: 对每个 Han span 做分词 + POS+NER 标注（JSON 协议）
  -> Build token stream with global offsets
  -> Word lookup: 优先匹配 word.json（词级固定拼音词典）
  -> Char base lookup: 未命中词表的字符使用 char_base.json（字基础读音）
  -> Polyphone detection: 结合 polyphone.json 识别多音字
  -> POS+NER disambiguation: 多音字使用 polyphone_disambig.json 进行 pos+ner-aware 消歧
  -> Rule Engine (overrides.json): 应用用户自定义覆盖规则
  -> LLM double check: 对低置信或冲突处进行复核，输出需要人工确认项
  -> (optional interactive) 用户选择读音
  -> 输出最终拼音文本 + 结构化报告
  -> 从用户标注生成 overrides.json 规则（可选：最小化上下文）
```

---

## 2. 输入输出

### 2.1 输入

* 任意 Unicode 文本字符串 `input_text`。
* 可包含：汉字、ASCII/Unicode 字母、数字、URL、标点符号、数学符号、空格/换行、emoji。

### 2.2 输出（主输出）

* `output_text`: 将汉字替换为带声调拼音（tone marks），其余片段原样保留。
  * 对汉字：按分词 token 输出拼音；**token 内不含空格**，**token 与 token 之间用单个空格分隔**。
  * 对非汉字 protected 片段：保持 `span.text` 字符完全一致；必要时在其与相邻“拼音词”之间补一个空格（仅用于分隔，不改写 protected 内容）。
* 示例：

  * 输入：`细说`
  * 输出：`xìshuō`

> 重要：输出文本会插入空格作为“词边界”（汉字 token 之间、以及汉字 token 与相邻的非汉字 word-like 片段之间）。URL/英文/数字等 protected 片段必须字符完全一致；标点与输入中的空白符（space/tab/newline）保持原样。

### 2.3 输出（结构化报告）

输出一个 `report.json`（或返回对象）用于调试与扩展：

* LLM 分词与 POS+NER
* 每个多音字的候选集、最终选择、应用的规则链路
* 冲突/不确定项列表（以及是否已由用户确认）
* overrides 生成记录（如有）

---

## 3. 预处理：混排保护与 span 切分

### 3.1 目标

* 识别并“保护”以下片段，使其在后续流程中不被分词/改写：

  * URL（含 http/https、裸域名可选、带路径参数）
  * 英文单词/缩写（latin 字母序列，允许 `-`、`_`）
  * 数字（含小数、科学计数法可选、百分号、货币符号可选）
  * 标点符号、空白符（space/tab/newline）
  * 数学符号与常见操作符（`+ - * / = < > ≤ ≥ ≠ ∑ √` 等）
  * emoji 与其它非汉字段落（可简单归为 protected）

### 3.2 Span 数据结构（规范）

偏移使用 **0-based**，`end` 为**半开区间**（Python slice 语义）。

```json
{
  "schema_version": 1,
  "text": "原始输入",
  "spans": [
    {
      "span_id": "S0",
      "type": "han",
      "start": 0,
      "end": 6,
      "text": "银行行长"
    },
    {
      "span_id": "S1",
      "type": "protected",
      "kind": "punct",
      "start": 6,
      "end": 7,
      "text": "，"
    }
  ]
}
```

### 3.3 切分策略（可实现为确定性扫描）

* 先用正则/有限状态机优先识别 URL，再识别 email（可选），再识别 latin/number，最后处理剩余字符。
* `han` span 仅包含汉字（Unicode Han script，或以 `\u4e00-\u9fff` 为主，扩展 CJK 区可选）。

---

## 4. LLM 分词 + POS+NER 标注协议

### 4.1 设计原则

* LLM 只处理 `han` span（推荐）；避免它对 URL/英文做“分词改写”。
* 输出必须是**严格 JSON**（无多余文本）。
* 必须满足：每个 span 的 token `text` 串联后 == 原 span `text`。

### 4.2 LLM 请求（建议）

```json
{
  "schema_version": 1,
  "task": "segment_and_tag",
  "tagset": {
    "upos": "UDv2",
    "xpos": "CTB",
    "ner": "CoNLL"
  },
  "spans": [
    { "span_id": "S0", "text": "银行行长重新营业" },
    { "span_id": "S3", "text": "他得去得到答案" }
  ]
}
```

### 4.3 LLM 响应（必须）

```json
{
  "schema_version": 1,
  "spans": [
    {
      "span_id": "S0",
      "tokens": [
        { "text": "银行", "upos": "NOUN", "xpos": "NN", "ner": "O" },
        { "text": "行长", "upos": "NOUN", "xpos": "NN", "ner": "O" },
        { "text": "重新", "upos": "ADV",  "xpos": "AD", "ner": "O" },
        { "text": "营业", "upos": "VERB", "xpos": "VV", "ner": "O" }
      ]
    }
  ],
  "warnings": []
}
```

### 4.4 合法性校验与降级

若出现以下任一问题：

* token 串联 != span text
* POS 或 NER 缺失/非法（不在允许集合内）
* span_id 缺失或 tokens 为空

则降级策略：

1. 回退为“确定性分词”：优先用 `word.json` 做最长匹配切分（forward maximum matching）；未命中再按字切分。
2. 降级产生的 token：`upos="X"`, `xpos="UNK"`, `ner="O"`。
3. 在 report 中记录 `llm_invalid=true`，方便后续排查。

---

## 5. 拼音查找与多音字消歧流程

### 5.1 三级查找策略

系统采用三级查找策略确定每个汉字的拼音：

```
Token("银行") 
  -> 1. word.json 词表匹配: "银行" -> "yínháng" (命中，直接使用；词内无空格)
  -> 2. 若未命中: char_base.json 字表: "银"->"yín", "行"->(多音字候选)
  -> 3. 若为多音字: polyphone_disambig.json POS+NER 消歧
```

### 5.2 word.json（词级固定拼音词典）

优先匹配整词，覆盖常见词汇的固定读音（尤其是多音字组成的词）。

示例（片段）：

```json
[
  { "word": "银行", "pinyin": "yín háng" },
  { "word": "行长", "pinyin": "háng zhǎng" },
  { "word": "重新", "pinyin": "chóng xīn" }
]
```

匹配规则：
* 完全匹配 token text，直接返回词级拼音
* 词表中的多音字已按该词语境确定读音
* 优先级最高，跳过后续消歧步骤

> 规范化：`word.json` 的 `pinyin` 可能用空格分隔音节（如 `yín háng`）。实现时应将其规范化为“词内无空格”的形式（如 `yínháng`），以满足输出格式要求。

### 5.3 char_base.json（字基础读音表）

当词表未命中时，按字查找基础读音。

```json
{ "index": 1, "char": "一", "pinyin": ["yī"], "strokes": 1, "radicals": "一" }
```

* `pinyin` 为候选列表；长度为 1 时可视为单音字。
* 长度 > 1 时为多音字，进入 polyphone_disambig.json 处理（或与 `polyphone.json` 交叉校验）。
* 文件实现提示：本仓库的 `char_base.json` 可能不是严格 JSON 数组（更像“对象流/行式 JSON”）；加载时建议用流式解析/预处理包装为数组，避免一次性读入和严格 `json.load` 失败。

### 5.4 polyphone.json（多音字候选表）

仅用于识别哪些字是多音字，以及有哪些候选读音。

```json
[
  { "index": 6, "char": "厂", "pinyin": ["chǎng", "ān", "hàn"] },
  { "index": 12, "char": "儿", "pinyin": ["ér", "er"] }
]
```

### 5.5 polyphone_disambig.json（POS+NER 消歧表）

多音字消歧的核心数据，基于 POS+NER 上下文选择最佳读音。

结构见外部文件 `polyphone_disambig.json`，关键字段：
* `items[]`：每个元素对应一个多音字
  * `char`: 多音字
  * `candidates`: 候选读音列表（tone marks）
  * `default`: 默认读音（无法消歧时的 fallback）
  * `contexts`: 基于 `pos|ner` 组合的消歧统计表（字典；key 形如 `pos=NOUN|ner=O`，value 形如 `{best,p,p2,n}`）

示例逻辑：
```
char="长" + pos=NOUN|ner=O -> 概率最高: "zhǎng" (如"校长")
char="长" + pos=ADJ|ner=O  -> 概率最高: "cháng" (如"很长")
```


---

## 6. 规则系统：overrides.json 用户覆盖规则

### 6.1 定位与流程

`overrides.json` 是在 `polyphone_disambig.json` 消歧之后的**用户自定义覆盖层**，用于：

* 修正自动消歧的错误
* 添加自定义业务场景读音
* 沉淀用户交互标注的结果

处理流程：
```
LLM 分词 + POS+NER
  -> word.json 词表匹配
  -> char_base.json 单音字
  -> polyphone_disambig.json POS+NER 消歧
  -> 【overrides.json 用户覆盖规则】（当前层）
  -> LLM double check
  -> 用户确认
```

### 6.2 规则总体原则

* 规则只解决“典型语法/词法环境下的多音字”，不追求全覆盖。
* 规则可基于：

  * token 的 `text`
  * token 的 `upos/xpos/ner`
  * 左右邻接 token（以及它们的 POS/NER/text）
  * 词内位置（多音字在 token 的第几个字）
* 规则执行需要：

  * **优先级 priority**（大者先）
  * **可解释性**（记录 rule_id、匹配上下文、动作）
  * **冲突检测**（多个规则给同一字不同读音）

### 6.3 Rule 数据结构（base_rules.json 与 overrides.json 共用）

```json
{
  "schema_version": 1,
  "rules": [
    {
      "id": "xing_vs_hang_in_bank",
      "priority": 1000,
      "description": "‘银行’中的‘行’读 háng",
      "match": {
        "self": { "text": "银行" }
      },
      "target": {
        "char": "行",
        "occurrence": 1
      },
      "choose": "háng"
    }
  ]
}
```

#### 字段说明

* `id`: 全局唯一字符串（建议可读、可追溯）。
* `priority`: 整数，越大越优先；`overrides.json` 默认应高于基础规则。
* `match`: 匹配条件（见 6.3）。
* `target`:

  * `char`: 需要消歧的汉字
  * `occurrence`: 在 token `text` 中第几次出现（1-based）；可选 `"all"`
* `choose`: 选定读音（tone marks；如 `háng`；轻声用不带音标的形式，如 `de`）。
* （可选）`confidence`: 0~1，用于 double check 阈值；未提供默认 0.8。

### 6.4 Match 语法（可扩展但先做最小可用集）

`match` 支持以下 key（全部为 AND 关系；内部数组默认为 OR）：

```json
"match": {
  "self": {
    "text": "行长",
    "text_in": ["行长", "校长"],
    "regex": ".*行长.*",
    "upos_in": ["NOUN", "PROPN"],
    "xpos_in": ["NN", "NR"],
    "ner_in": ["O", "LOC"],
    "contains": ["行"]
  },
  "prev": {
    "text_in": ["银行", "本行"],
    "upos_in": ["NOUN"],
    "ner_in": ["O"]
  },
  "next": {
    "upos_in": ["NOUN", "VERB"],
    "ner_in": ["O", "PER"]
  }
}
```

约定：

* `self` 指当前 token（包含 target char 的 token）。
* `prev/next` 是相邻 token；不存在时视为不匹配（或允许配置 `allow_missing`，默认不允许）。
* `text` 是精确匹配；`text_in` 为列表；`regex` 是正则字符串（实现语言自行选择正则引擎）。
* `upos_in/xpos_in/ner_in` 支持集合匹配。
* `contains`：token text 包含某些字（用于弱词典场景）。

---

## 7. 规则执行语义（Grammar Reduction）

### 7.1 执行顺序与冲突

1. 加载规则集：`overrides.json`（最高优先） + `base_rules.json`。
2. 按 `priority DESC, id ASC` 排序。
3. 扫描 token 流：

   * 找到包含 `target.char` 的 token
   * 评估 `match`（含 prev/next）
   * 命中则产生动作：对指定 occurrence 的该字符节点 `set_reading(choose)`
4. 若同一字符节点：

   * 已被设定读音，再遇到同读音规则：记录 provenance（可选）
   * 再遇到不同读音规则：标记为 `conflict`，加入 `needs_review`

### 7.2 未命中规则的策略（baseline）

* 若候选读音集合仅 1：直接 resolved。
* 若 >1：

  * 先用“词级词典”快捷规则（如果 token text 在词典里有固定拼音，优先）
  * 否则标记 `needs_review`，交给 LLM double check + 用户确认流程

> 提示：可以实现一个 `lexicon.json`（token->pinyin）作为补充，但不是必需交付。

---

## 8. polyphone_disambig.json 消歧示例（参考）

> 下列规则是“系统应支持表达的规则类型”，不是要求一次性覆盖所有多音字。
> 实装时建议先落地 20~50 条最常见规则，并确保 overrides 能补齐长尾。

### 8.1 “得” (de / děi / dé)

* 结构助词（轻声）：`upos=PART` 或 `xpos=DEC/DEG/DER`（若使用 CTB 细分）→ `de`

  * 例：`跑得快` → `pǎo de kuài`
* 动词“得到”（dé）：token text 精确等于 `得到/得出/得知/...` 或 `upos=VERB` 且后接宾语

  * 例：`得到答案` → `dédào dáàn`
* 情态/必须（děi）：`得去/得做/得赶紧`，常见为 `upos=AUX` 或 `self.text` 匹配 `得(去|做|赶紧|马上)`

  * 例：`我得去` → `wǒ děiqù`

### 8.2 “的” (de / dí / dì)

* 结构助词（轻声）：`upos=PART` → `de`
* `目的`：token text = `目的` → `dì`
* `的确`：token text = `的确` → `dí`

### 8.3 “重” (zhòng / chóng)

* `重新/重复/重来/重启/重做`：表示“再次”倾向 `chóng`

  * match: `self.text_in=["重新","重复","重来","重启","重做"]` → `chóng`
* 形容词“重要/沉重/重视/重心/重负”倾向 `zhòng`

  * match: `self.upos_in=["ADJ","VERB","NOUN"]` 且 `self.contains=["重"]` 并命中词表 → `zhòng`

### 8.4 “行” (xíng / háng)

* 行业/银行/同行/本行/支行：`háng`

  * match: `self.text_in=["银行","行业","同行","本行","支行"]`
* 动词“行走/行使/行不行/可行”：`xíng`

  * match: `self.upos_in=["VERB","AUX"]` 或 `self.text_in=["可行","行走","行使"]`

### 8.5 “长” (cháng / zhǎng)

* 形容词“长短/很长/延长/长度”：`cháng`

  * match: `self.upos_in=["ADJ"]` 或 `prev.upos_in=["ADV"]`（很/更/太）且 self 含 “长”
* 名词/职位“校长/队长/行长/部长/家长”：`zhǎng`

  * match: `self.text_in=[...常见职位词表...]` 且 `self.upos_in=["NOUN","PROPN"]`

### 8.6 “乐” (lè / yuè)

* 音乐相关名词：`yuè`

  * match: `self.text_in=["音乐","乐器","乐曲","乐队"]` 或 next/prev 含 `曲/器/队/团`
* 快乐/乐意：`lè`

  * match: `self.text_in=["快乐","乐意","乐观"]`

> 规则文件中只需写出这些“高精度、可解释”的规则；剩余场景依赖 double check + overrides 迭代。

### 8.7 NER 辅助消歧示例

* 人名中的多音字（如"曾"、"查"）：`ner=PER` 时优先人名读音

  * match: `self.ner_in=["PER"]` 且命中人名词表 → 取人名特定读音
* 地名中的特殊读音（如"厦"在"厦门"）：`ner=LOC` 辅助判断

  * match: `self.ner_in=["LOC"]` 且 `self.contains=["厦"]` → `xià`

---

## 9. 拼音生成与 tone marks

### 9.1 内部表示

* 本系统对外与规则层统一使用 **tone marks**（如 `háng`、`de`（轻声无音标））。
* 允许实现内部用其它表示（如数字声调），但必须在输出/报告/规则匹配时统一为 tone marks。

### 9.2 词内拼音规范化（实现要求）

实现一个 `normalize_word_pinyin(pinyin: str) -> str`：

* 将音节分隔空格移除：`"yín háng" -> "yínháng"`（满足“词内无空格”）。
* 保持 tone marks；轻声以无音标形式表示（如 `"de"`）。
* `ü` 规范化：内部如用 `v` 表示 `ü`，最终输出必须为 `ü`（含声调时为 `ǖǘǚǜ`）。

---

## 10. LLM Double Check（复核协议）

### 10.1 触发条件

满足任一则进入 double check：

* `needs_review`（未消歧）
* `conflict`（规则冲突）
* `confidence < threshold`（阈值默认 0.85，可配置）

### 10.2 LLM 输入（建议）

给 LLM：

* 原始 `input_text`
* spans + tokens + POS+NER
* 多音字候选与当前选择
* 需要复核的项列表（含上下文窗口）

### 10.3 LLM 输出（必须 JSON）

```json
{
  "schema_version": 1,
  "verdict": "ok|needs_user",
  "items": [
    {
      "span_id": "S0",
      "token_index": 1,
      "char_offset_in_token": 0,
      "char": "行",
      "candidates": ["háng", "xíng"],
      "recommended": "háng",
      "reason": "‘行长’职位名词",
      "needs_user": false
    }
  ]
}
```

> 索引约定：`token_index` 与 `char_offset_in_token` 均为 **0-based**。

### 10.4 复核结果整合

* 若 LLM 给出 `recommended` 且 `needs_user=false`：可直接采纳（但在 report 中记录来源 `llm_double_check`）。
* 若 `needs_user=true`：进入交互式用户确认（若 `--interactive`），否则在 report 中保留未解决项并采用保守默认（见 11.3）。

---

## 11. 用户交互式标注与 overrides 提取

### 11.1 交互方式（CLI/UI 任选）

对每个 `needs_user` 项输出：

* 原文片段（含左右上下文窗口）
* 候选拼音（tone marks 或数字声调均可，但建议展示 tone marks）
* 让用户选择 1/2/3…

### 11.2 用户选择写入决策

用户确认后：

* 该字符节点 `resolved_by="user"`
* 写入 report：`user_choice`

### 11.3 非交互模式策略（必须定义）

当存在 `needs_user` 但未开启交互：

* 采用“默认读音”策略（例如候选列表第一个，或字典常用读音）
* 同时在 report 中记录 `unresolved_fallback=true`，并保留候选列表供后续人工处理

### 11.4 overrides.json 生成（核心可扩展性）

当用户对某个 token 中的某个多音字做了选择，系统应尝试生成一条**尽量稳健且不过拟合**的 override 规则。

#### 生成策略（推荐：由强到弱）

1. 若 `self.text`（token 文本）长度 ≥ 2：生成 **精确 token 文本**规则

   * 优点：高精度；缺点：覆盖面窄
2. 若 token 长度 = 1（单字词/虚词）：引入 `prev/next` 的 `text_in` 或 `upos_in/ner_in` 作为上下文
3. 始终保留 `self.upos_in / xpos_in / ner_in`（如果可用）以满足“pos+ner aware”

#### overrides.json 格式

```json
{
  "schema_version": 1,
  "rules": [
    {
      "id": "override_2026-02-06_0001",
      "priority": 100000,
      "description": "user override: 行(行长)=háng",
      "match": {
        "self": { "text": "行长", "upos_in": ["NOUN","PROPN"], "ner_in": ["O"] }
      },
      "target": { "char": "行", "occurrence": 1 },
      "choose": "háng",
      "meta": {
        "created_at": "2026-02-06",
        "source": "user",
        "example": "银行行长"
      }
    }
  ]
}
```

---

## 12. 输出拼接规则（保持结构）

### 12.1 拼接原则

* 对 `protected` span：原样输出 `span.text`（字符完全一致）
* 对 `han` span：按 token 顺序输出其拼音，**token 内无空格**，**token 间以单空格分隔**
* 额外空格规则（用于避免粘连）：
  * 若某个 `han` token 的拼音与相邻的 protected *word-like* span（如 `latin|number|url`）在输出中直接相邻，则在它们之间插入单空格
  * 不在标点前强行插空格；不强制在标点后补空格（除非输入中已有）
* 保持原始 spans 的顺序与所有 protected 内容不变

### 12.2 示例

输入：

```
银行行长重新营业，OpenAI API v2.0：https://openai.com
```

可能输出：

```
yínháng hángzhǎng chóngxīn yíngyè，OpenAI API v2.0：https://openai.com
```

---

## 13. 工程交付物与目录建议

### 13.1 必须交付

数据文件：
* `word.json`：词级固定拼音词典（优先匹配，覆盖常见多音词）
* `char_base.json`：字基础读音表（每字提供 `pinyin[]` 候选；长度为 1 可直接返回；长度 > 1 进入消歧）
* `polyphone.json`：多音字候选表（记录哪些字是多音字及其候选读音）
* `polyphone_disambig.json`：POS+NER 消歧表（核心消歧数据，按 pos|ner 组合给出最佳读音）
* `overrides.json`：用户覆盖规则（可不存在；程序运行时自动创建空模板）

程序：
* 核心库 + CLI（或等价入口）：

  * `pinyinize(text, options) -> { output_text, report }`
* 单元测试与样例集（至少覆盖：混排、URL、常见多音字、冲突处理）

### 13.2 可选增强

* `base_rules.json`：额外的自定义规则（用于复杂上下文模式，非必需）
* `report.html`：可视化 debug（非必需）

---

## 14. 验收标准（Acceptance Criteria）

### 14.1 功能正确性

* 输入 `细说` 输出必须是 `xìshuō`
* 混排不破坏：

  * 输入包含 URL：URL 子串必须原样保留（字符完全一致）
  * 英文与数字子串必须原样保留（字符完全一致）
* offsets 校验：

  * LLM tokens 串联必须等于各自 han span text；否则必须触发 fallback 且流程不中断
* 冲突处理：

  * 同一字符被不同规则赋值时必须标记 conflict 并进入 double check

### 14.2 可扩展性

* 增加 overrides.json 规则后，系统必须在不改代码情况下生效
* overrides 的优先级必须高于 base_rules

### 14.3 可追溯性

* report 中必须能看到：

  * 每个多音字最终读音来自：base_rule / override / llm_double_check / user / fallback
  * 命中的 rule_id（如适用）
  * 未解决项列表（如适用）

---

## 15. 最小测试用例集合（建议）

1. 基础：

* in: `细说`
  out: `xìshuō`

2. 行/长/重：

* in: `银行行长重新营业`
  out: `yínháng hángzhǎng chóngxīn yíngyè`

3. 得：

* in: `他得去得到答案`
  out: `tā děiqù dédào dáàn`

4. 混排：

* in: `细说OpenAI的API v2.0：https://openai.com`
  out: `xìshuō OpenAI de API v2.0：https://openai.com`

5. 冲突触发（构造一个规则冲突的例子，验证 report.conflicts 存在）：

* in: `行行好`（可能需要用户选择/或规则）
  out: 允许 fallback，但 report 必须记录 unresolved/conflict

---

## 16. 实现提示（给 codex/gemini-cli 的工程化注意点）

* 强烈建议把 LLM 调用封装为 `LLMAdapter` 接口，便于：

  * mock 测试（无 key 也能跑）
  * 切换 gemini/openai/本地模型
* 规则引擎必须是纯函数式/可测试的：

  * 输入：tokens + candidates + rules
  * 输出：decisions + conflicts + log
* tone marks 转换建议单独模块并写足单测（`ü`、轻声、`iu/ui` 等）

## Agents Added Memories
- 用 python 不用 python3
- The user's operating system is: darwin
- may use Apple Silicon MPS for acceleration
- [a meta request bout the working process] use `echo $'\a'` to notify the user when waiting for input
- 可以调 ollama gemma3:1b 来作为 LLM
