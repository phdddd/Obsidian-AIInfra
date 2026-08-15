# xLLM Prefix Cache：从请求前缀到可恢复状态的完整源码链路

> 本文是 xLLM code_wiki 第 08 篇。目标不是介绍一个“把 prompt 存起来”的小技巧，而是把 Prefix Cache 从 token 身份、物理 block 生命周期、Scheduler 准入、Batch 元数据、模型侧 attention、设备后端、PD/CP 通信，一直追到测试和 Kimi K3 的当前边界。
>
> 代码基线：工作区当前 xllm/ 源码。源码链接均指向本地绝对路径；链接旁的行号是阅读入口，不把接口存在误写成目标模型已经验证。

## 1. 先给结论

Prefix Cache 的一句话模型是：

> 对一段已经完成 prefill 的公共 token 前缀，保存它对应的物理 KV/state block 和可验证的链式身份；新请求命中后，把这些 block 直接挂到自己的 sequence，只计算第一个未命中 token 之后的 suffix。

这句话包含四个必须同时成立的条件：

1. 身份相同：请求 token、位置相关多模态语义以及哈希 seed 必须产生同一条 hash chain。
2. 内容已完成：block 中的数据必须已经由模型 forward 写完，不能只因为 Scheduler 分配过就拿来读。
3. 状态可恢复：普通注意力恢复 KV block；线性注意力还要恢复递推 state；混合注意力必须取所有参与叶子的安全交集。
4. 执行输入一致：Batch 必须把命中 token 数、block table、new cache slots、位置和 mask 传给后端，模型不能把命中长度当成普通新 prompt。

当前实现的关键事实：

- 文本 hash 的实际实现是 XXH3-128 with seed，旧 docs 里写的 murmur_hash 是陈旧描述。
- 默认 block_size=128，默认 enable_prefix_cache=true，默认同 batch 共享开关为 false。
- 普通 KV 使用固体前缀；SWA 使用允许中间 invalid placeholder 的 gap-tolerant cache；LINEAR 保存 chunk-end state checkpoint。
- CompositeBlockManager 根据模型叶子组合选择 FLAT_KV、FLAT_KV_LINEAR 或 SWA_COMPRESSED。
- 本地 Prefix Cache 不新增 collective；它通过减少 prefill token，间接减少 TP/EP/CP 计算和通信。
- 当前 Kimi K3 的服务已完成短请求启动冒烟，但 Prefix Cache 尚未有请求级等价性、长多 chunk、跨页和 KDA+MLA 联合命中验证。

## 2. 它解决的真实问题

### 2.1 普通 prefill 的重复计算

自回归模型在 prefill 阶段需要让每个 token 读取此前 token 的 K/V。若每个请求都以完整 prompt 开始，公共 system prompt 会被重复计算：

~~~text
请求 A: [system, rules, user_a]
请求 B: [system, rules, user_b]
请求 C: [system, rules, user_c]
~~~

没有 Prefix Cache 时，三个请求都计算 [system, rules]。有 Prefix Cache 时，A 先生成公共 block，B/C 复用公共 block，只计算各自 suffix。

这节约的是：

- prefill attention 的 Q/K/V 投影与 attention 计算；
- prefill MLP/MoE 的矩阵乘和路由；
- 由这些层产生的 TP AllReduce、AllGather、MoE AllToAll；
- 生成 KV 或线性 state 的设备写入；
- 在 PD 中需要传输的新增 KV 字节。

它不直接节约：

- 新 suffix 的计算；
- decode 每一步读取历史 KV 的带宽；
- tokenizer、HTTP、采样和输出序列化；
- 首次出现公共 prompt 的计算。

### 2.2 为什么不能只做 response cache

response cache 以完整请求和采样参数为 key，命中时直接返回文本；Prefix Cache 的目标是保留模型内部状态，让不同 suffix、温度和最大生成长度仍能基于同一公共上下文继续生成。

因此 Prefix Cache 的 key 只需要描述“状态是否可复用”，而不是“完整答案是否相同”。这也是它要和 Scheduler、Block Manager、模型 forward 共同工作的原因。

### 2.3 它的非目标

- 它不是 tokenizer cache；tokenizer 仍需要完成请求文本到 token 的转换。
- 它不是权重 cache；模型权重在 worker/context 中管理。
- 它不是跨模型安全隔离机制；当前 key 中没有显式 model/revision/adapter 字段，命名空间通常依赖每个 Engine 独立的 cache 实例。
- 它不是跨节点自动同步；远端 KV transfer 是另外的 PD、host/offload 或 global-store 路径。
- 它不是近似推理算法；正确命中时应复用原始 KV/state。

## 3. 一张总图看懂完整链路

~~~text
HTTP/OpenAI request
    |
    v
tokenizer + multimodal scheduler_data
    |
    v
Request -> Sequence(tokens, mm_data, block_hashes, linear_state_hashes)
    |
    v
SchedulerPolicy::allocate_for_prefill
    |
    +--> CompositeBlockManager::allocate_shared_for_sequence
    |       |
    |       +--> BlockHasher -> XXH3 chained keys
    |       +--> PrefixCache::match / LinearStatePrefixCache::match
    |       +--> trim shape-specific hits
    |       +--> mount shared Block handles / restore source
    |
    +--> allocate suffix blocks and admission budget
    |
    v
BatchInputBuilder
    +--> token_ids = tokens[cached_tokens:]
    +--> kv_cache_tokens_nums = cached_tokens per sequence
    +--> block_tables / multi_block_tables
    +--> new_cache_slots / paged metadata
    +--> linear state restore/save metadata
    |
    v
Worker -> Model forward -> Attention backend
    +--> read shared KV/state blocks
    +--> write only new_cache_slots
    |
    v
Sequence cursor advances; complete blocks become cacheable
    |
    v
BlockManager::cache / PrefixCache::insert / LRU eviction
~~~

这张图里有两个时间点容易混淆：allocate_shared 是“尝试复用已经存在的状态”，cache 是“把本次已经完成的状态发布到未来请求”。中间的 forward 才是“把新 block 写成有效内容”的动作。

## 4. 源码地图：从哪里开始读

建议按以下顺序阅读：

|层次|源码|回答的问题|
|---|---|---|
|配置|[kv_cache_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.h:42)|开关、block size、seed 是什么|
|哈希|[block_hasher.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/block_hasher.h:38)|hash chain 的抽象和 MM 扩展|
|哈希实现|[block_hasher.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/block_hasher.cpp:25)|XXH3、seed、图片 schedule key 如何加入|
|缓存|[prefix_cache.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/prefix_cache.h:44)|LRU map 与 API|
|缓存实现|[prefix_cache.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/prefix_cache.cpp:29)|match、insert、evict 的确切边界|
|状态|[sequence_kv_state.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence_kv_state.h:30)|shared cursor、cached cursor、capacity|
|请求|[sequence.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence.h:55)|stage、token 数、hash 缓存|
|组合|[composite_block_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/composite_block_manager.cpp:44)|KV、LINEAR、SWA 叶子如何组合|
|调度|[scheduler_policy.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/scheduler_policy.cpp:363)|prefill 准入和 chunk rematch|
|批输入|[batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:680)|只发 suffix 的实现|
|执行|[worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:214)|linear 初始状态与 MLA metadata|
|测试|[prefix_cache_test.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/tests/core/framework/prefix_cache/prefix_cache_test.cpp:98)|基本命中、插入、驱逐与预计算 hash|

## 5. 配置入口和优先级

### 5.1 KVCacheConfig 的默认值

[KVCacheConfig](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.h:42) 把 Prefix Cache 归在 KV cache 配置中。当前关键字段是：

|字段|当前默认/含义|源码入口|
|---|---|---|
|block_size|128 token|[kv_cache_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.h:59)|
|enable_prefix_cache|true|[kv_cache_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.h:69)|
|enable_in_batch_prefix_cache|false|[kv_cache_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.h:71)|
|max_linear_state_cache_slots|0|[kv_cache_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.h:73)|
|xxh3_128bits_seed|1024|[kv_cache_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.h:75)|
|enable_xtensor|false|[kv_cache_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.h:77)|

### 5.2 flags、JSON 和运行时 Options

gflag 默认和解析逻辑在 [kv_cache_config.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.cpp:51)。JSON 初始化会在同一文件的 initialize 逻辑中覆盖对应字段。

在线服务还会把配置复制到 runtime Options：[runtime/options.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/options.h:47)。Master 创建 Engine 时把 enable_prefix_cache、block size、linear slots 传给 BlockManagerPool，入口可看 [master.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/master.cpp:385)。

实际排查时要同时检查三处：

1. 命令行/JSON 是否设置了全局 flag；
2. Python LLM/VLM Options 是否覆盖了 flag；
3. Engine 初始化时是否按模型、角色、平台再次关掉或限制 Prefix Cache。

只看命令行不够，因为 DeepSeek、XTensor、PD decode role 等路径可能在更后面做组合级 gating。

## 6. block_size 如何影响收益

Prefix Cache 只匹配完整 block。代码在 PrefixCache::match 中 round_down token 数，再计算 n_blocks，见 [prefix_cache.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/prefix_cache.cpp:33)。

~~~text
matched_tokens = floor(P / B) * B
uncached_tokens = prompt_tokens - matched_tokens
~~~

大 block 的优点：

- hash map entry、LRU node 和引用计数的元数据更少；
- block table 更短；
- 一次插入/驱逐管理更多 token，CPU 开销低。

大 block 的缺点：

- 前缀必须更长才有首个命中；
- 尾部变化会让整个最后 block 不可复用；
- 线性 state、SWA 压缩叶子需要更严格的对齐。

小 block 的优点相反，但增加 block table、hash lookup、碎片和 allocator 操作。选择 B 时应该看真实 prompt 的公共长度分布，而不是只看单请求最大长度。

## 7. Prefix Cache 的身份：链式 XXH3-128

### 7.1 实际算法

当前实现位于 xxh3_128bits_hash：[block_hasher.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/block_hasher.cpp:25)。

~~~text
H0 = XXH3_128(tokens[0:B] as int32 bytes, seed)
Hn = XXH3_128(H(n-1) || tokens[nB:(n+1)B] as int32 bytes, seed)
~~~

每个 Hn 是一个 16-byte XXH3Key。第二块不会只依赖自己的 token，因此它携带了此前整个 prefix 的身份。

### 7.2 为什么是链而不是整段 hash

整段 prompt hash 只能回答整段是否相同，不能在第 2 块命中时复用前两块，也不能在 chunked prefill 中增量推进。链式 hash 允许：

- 一次只扩展新到达的 block；
- 在第一个 miss 处停止；
- 把每个 block 作为独立 LRU entry；
- 同一个公共前缀的不同 suffix 共享同一组前缀 block。

### 7.3 seed 的作用

seed 来自 KVCacheConfig::xxh3_128bits_seed，不是随机化每个请求。相同 seed 才能让同一进程内的预计算 hash 与 cache hash 一致；切换 seed 会使旧 cache 全部逻辑 miss。

seed 不是模型隔离字段。模型、revision、LoRA、tokenizer 或位置配置变了，调用层必须重建 Engine/cache 或增加命名空间；不能因为 token id 相同就假定 state 相同。

## 8. 多模态 hash：token 相同不代表语义相同

MM 路径在 mm_xxh3_128bits_hash：[block_hasher.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/block_hasher.cpp:54) 先算 token chain，再把覆盖当前 block 的多模态 schedule_data.key 追加到 hash 输入。

MMBlockHasher 根据 token position 找到重叠的图片/视频 item：[block_hasher.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/block_hasher.cpp:173)。一个图像跨越多个 block 时，每个受影响 block 都会带上该 item key。

这避免了：

~~~text
placeholder token 序列相同
但 image_a != image_b
=> 视觉 encoder/KV 不同
=> 不能共享
~~~

当前多模态能力还取决于 VLM batch builder、MM item schedule key、cross-attention cache 和模型构造；以具体 backend 的 leaf predicate 和模型 CHECK 为准。

## 9. Hash 的增量更新与 precomputed fast path

extend_prefix_hashes 在 [block_hasher.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/block_hasher.cpp:82) 从已有 hashes.size() 继续生成到 boundary。它保存稳定 previous_hash，避免 vector 扩容后内部指针失效。

Sequence 会缓存 block hash，见 [sequence.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence.cpp:770)。Prefix Cache match/insert 若收到足够的 precomputed hashes，就只做 map lookup，不重复切 token 和拼接 hash，见 [prefix_cache.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/prefix_cache.cpp:63)。

预计算不改变语义，改变的是 CPU 路径：

- 首次请求仍需生成 hash；
- chunked prefill 后续阶段只扩展新增边界；
- 多个 leaf 可以复用同一组 token hash；
- MM leaf 仍要带自己的 hasher type 和 metadata；
- hash 预计算不能跳过内容已经完成的前提。

## 10. Token 改写为什么必须失效 hash

Sequence 在追加生成 token 或 beam 分支发生 token 替换时，会清除受影响 block 的 hash 和 linear hash，见 [sequence.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence.cpp:416)。

链式 hash 有前向依赖：第 k block token 改变，Hk 和后续所有 H 都改变。继续使用旧 hash 会导致 cache lookup 误命中，属于逻辑正确性 bug，不是普通 cache miss。

beam/best-of 复制也要遵守规则。Sequence copy 会共享可复用 KV，但移除 EMBEDDING/LINEAR 的私有 live slot，见 [sequence.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence.cpp:276)。新分支必须重新取得自己的线性 live slot，不能让多个 beam 写同一个 recurrent state。

## 11. PrefixCache 的内部数据结构

PrefixCache 维护三类数据：

1. cached_blocks_：XXH3Key 到 Node 的哈希表；
2. lru_lst_：按最近使用顺序排列的双向链表；
3. Node 内的 Block handle：真正持有物理 block 的引用。

声明见 [prefix_cache.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/prefix_cache.h:127)。Block 本身有 manager、id、hash、valid、shared 和原子 refcount 字段，见 [block.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block.h:101)。

重要区分：

- cache map entry 是逻辑索引；
- Node 是 LRU 所有权；
- Block handle 是物理资源引用；
- Sequence 里另有一个 Block handle，命中后会增加 refcount；
- is_shared 表示该 block 仍被 cache 或其他 sequence 持有，不能简单回收到 free list。

## 12. Block 的 RAII 和引用计数

Block 构造时 refcount 从 1 开始，copy 会增加引用，析构会减少引用，见 [block.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block.cpp:27)。最后一个引用释放时，BlockManager 才真正回收物理 id，见 [block.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block.cpp:108)。

一次命中后的引用关系：

~~~text
free-list owner(0)
       |
       +-- sequence handle
       +-- prefix-cache Node handle
       +-- temporary probe handle
~~~

probe 临时 vector 必须及时 clear。Composite manager 的 release_probes 先交还 leaf，再清空 alias，避免 refcount 过高导致 used-block accounting 不减，见 [composite_block_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/composite_block_manager.cpp:661)。

## 13. PrefixCache::match 的逐行语义

### 13.1 只接受完整 block

match 首先向下取整 token 数，再计算 n_blocks，见 [prefix_cache.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/prefix_cache.cpp:33)。尾部不完整 block 不会被匹配，也不会被当成可共享 cache。

### 13.2 先保留已共享块

Composite 场景下 Sequence 可能已经有共享块；match 先把 existed_shared_blocks 放入结果，从它的下一个 block 开始 lookup，见 [prefix_cache.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/prefix_cache.cpp:42)。

### 13.3 首个 miss 立即停止

每个命中 block 会被复制到结果并把 LRU node 暂移到工作列表；第一个 hash miss 后 break，见 [prefix_cache.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/prefix_cache.cpp:50)。

~~~text
H0 hit, H1 hit, H2 miss, H3 entry exists
=> 实际命中 [H0, H1]
~~~

### 13.4 LRU 触碰和指标

lookup 命中的节点按原顺序放回 LRU 工作链，并记录 prefix_cache_block_matched_rate、prefix_cache_block_matched_num，见 [prefix_cache.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/prefix_cache.cpp:90)。

命中不仅是查询结果，也改变未来 eviction 顺序；压测不能只记录命中率，还应观察 LRU 生存时间和驱逐量。

## 14. PrefixCache::insert 的逐行语义

insert 先取 min(token_ids.size()/block_size, blocks.size())，见 [prefix_cache.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/prefix_cache.cpp:106)。它只发布完整 token/block 对。

对每个 block：

1. 从上一个 block hash 继续计算或使用 precomputed hash；
2. 把 hash 写入 block 的 immutable hash 字段；
3. 若 map 已有相同 hash，释放新 block alias，继续触碰旧 entry；
4. 若没有，则创建 LRU node 并把 block 放入 cache；
5. 返回已经覆盖的完整 token span。

核心实现位于 [PrefixCache::insert](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/prefix_cache.cpp:124)。这个流程使重复 insert 幂等，避免同一公共 prefix 消耗两份物理 block。

## 15. LRU eviction 的边界

evict(num_blocks) 从最老节点开始，遇到 block.is_shared() 的 pinned block 会跳过，只有不再被请求、beam 或 probe 持有的 entry 才能删除，见 [prefix_cache.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/prefix_cache.cpp:243)。

结果是：

- cache entry 数量达到上限不等于立刻能回收同样数量的物理 block；
- 活跃请求长时间持有 shared block 时，LRU 可能找不到足够可驱逐对象；
- Scheduler 需要延迟 admission、抢占其他序列或退化到 miss。

BlockManager 在容量压力下会先算需要驱逐的 block 数，再调用 Prefix Cache，见 [block_manager_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_impl.cpp:131)。

## 16. BlockManager 如何把命中挂到 Sequence

### 16.1 创建缓存实例

Block manager 构造时，如果启用 Prefix Cache，就根据 block type 创建对应 cache；同时保留 id 0 作为 padding block，见 [block_manager_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_impl.cpp:50)。

### 16.2 allocate_shared

leaf manager 的 allocate_shared 调用 Prefix Cache match、记录 hit 指标并更新有效使用量，见 [block_manager_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/block_manager_impl.cpp:160)。它返回的是 Block handle，不是复制 KV 数据。

### 16.3 Sequence capacity

Sequence 的 KVCacheState 记录 shared block 数、总 block 数、cached token cursor 和当前最大容量。current_max_tokens_capacity 会根据 KV 或 C4/C128 的最小覆盖量判断 batch builder 是否有足够空间，见 [sequence_kv_state.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence_kv_state.cpp:275)。

### 16.4 反向释放

请求结束时 CompositeBlockManager::deallocate_for_sequence 先 cache 再释放 sequence blocks，见 [composite_block_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/composite_block_manager.cpp:440)。顺序很关键：先发布已经完成的 block，再让 sequence owner 消失。

## 17. Sequence 的三个 token 游标

Prefix Cache 代码中至少要区分三个概念：

|游标|含义|用途|
|---|---|---|
|num_tokens()|Sequence 当前 token 总数|prompt 加已生成 token|
|kv_cache_tokens_num()|当前 forward 前已经可读的 cached/forwarded token|决定 q 从哪里开始|
|cached_tokens()|已经达到 cache 发布边界的 token 数|决定 insert flush 到哪里|

Sequence stage 根据 cached count 与 prompt length 推导 PREFILL、CHUNKED_PREFILL、DECODE，见 [sequence.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence.h:134)。

num_prefix_cache_tokens 取 device/host state 中最大的 shared token 数，见 [sequence.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence.cpp:745)。它用于 usage 观测，不等于本步实际执行的全部 token。

## 18. 一个冷请求的完整生命周期

假设 block size=4，prompt token 为 [a,b,c,d,e,f,g,h,i]，cache 为空。

### 18.1 入队和首次匹配

Scheduler 把 request 变成 Sequence，allocate_shared_for_sequence 对完整的 [a,b,c,d]、[e,f,g,h] 做 hash lookup，得到 0 个命中。

### 18.2 物理准入

BlockManager 为 prompt 需要的 KV/其他 cache leaf 分配 block。Composite 的 allocate 是 fan-out：每个 leaf 先 stage，所有 leaf 足够才 commit，失败则回滚，见 [composite_block_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/composite_block_manager.cpp:366)。

### 18.3 Batch 构造

n_kv_cache_tokens=0，所以 q_seq_len=9。Batch builder 将全部 token、位置、block table 和新 slot 写入 forward input，见 [batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:680)。

### 18.4 forward 完成

Attention 把每个 token 的新 KV 写到 new_cache_slots；kv_cache_tokens_num 在 builder 中按本次 q 长度前移，见 [batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:972)。

### 18.5 发布完整 block

只有前两个完整 block 能进入 Prefix Cache，尾部 [i] 仍是 partial block。释放时 cache_for_sequence 按 cached_tokens 截断并调用 leaf cache，见 [composite_block_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/composite_block_manager.cpp:736)。

## 19. 一个热请求的完整生命周期

第二个请求 prompt 为 [a,b,c,d,e,f,g,h,x]。

1. hash H0、H1 与 cache 中 entry 相同，H2 因 x != i miss。
2. Sequence 挂载前两个 shared block，kv_cache_tokens_num=8。
3. Scheduler 只为 1 个 suffix token 以及后续 decode 预算申请新增容量。
4. Batch builder 设置 q_seq_len=1、logical_seq_len=9，token input 只包含 x。
5. backend 通过 block table 读取前 8 token 的历史 KV，并把 x 的 KV 写到新 slot。
6. 未来请求若匹配相同 [a..h,x]，完整 block 边界仍只会发布到 8；最后一个 block 只有部分内容。

因此“命中 8 token”不仅是统计数字，它同时改变输入 token 数、位置、mask、block table、KV 读范围、容量估计、usage 和后续 cache cursor。

## 20. 精确重复请求为什么保留一个 token 计算

Sequence add_shared_blocks 对整段 prompt 完全命中的情况会弹出最后一个完整 block，使至少还有一段模型工作，见 [sequence_kv_state.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/request/sequence_kv_state.cpp:184)。

原因不是哈希不准，而是 forward 仍需要产出：

- 下一 token 的 hidden/logits；
- 采样器可用的输出分布；
- 某些 backend 的 q length、mask 和 graph shape。

所以精确重复 prompt 不代表 TTFT 变成零。它通常会回退一个 block 的 token 到本次 prefill，之后 decode 从正确的位置继续。

## 21. Scheduler：Prefix Cache 何时匹配

Continuous Scheduler 初始化时读取全局 Prefix Cache 和 in-batch flags，见 [continuous_scheduler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/continuous_scheduler.cpp:96)。

调度 prefill 的核心顺序是：

~~~text
request candidate
  -> allocate_for_prefill
  -> allocate_shared / rematch
  -> calculate actual tokens after hit
  -> capacity guard
  -> admit into batch
~~~

allocate_for_prefill 的 full-prefill 分支先挂已有 prefix，再把实际 token 数设为总 prompt减 cached，见 [scheduler_policy.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/scheduler_policy.cpp:363)。

## 22. Scheduler：Chunked Prefill 下的重复匹配

chunked prefill 把长 prompt 分成多个调度 step。Prefix Cache 不能只在请求进入队列时匹配一次，因为前一 chunk 结束后新产生的完整 block 可能成为下一 chunk 的命中；chunk budget 也可能因为 decode-first 预算改变。

连续 Scheduler 对普通 cache 依据 chunked_match_frequency 决定 rematch 间隔，对 LINEAR 则每个 chunk 重新匹配，见 [scheduler_policy.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/scheduler_policy.cpp:416)。

LINEAR 还要求 chunk stride 与 block size 对齐；Engine 在构造 BlockManager 前做 CHECK，见 [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:516)。

## 23. Scheduler：统一策略和请求排序

UnifiedPolicy 会在排序/预算选择前主动 probe 请求的 prefix，记录命中 token，并在满足配置时发布同 batch prefix，见 [unified_policy.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/unified_policy.cpp:259)。

这影响调度公平性：

- 热请求的实际 token budget 较小，可能更容易被纳入 batch；
- 冷长请求仍占用容量，即使最终部分 token 被跳过；
- 若只按 request count 排队，会低估不同命中率请求的真实 compute cost；
- SLO 调度应把 cached_tokens 从 TTFT 预算中扣除，但把 lookup/restore 的 CPU/网络成本算回去。

## 24. In-Batch Prefix Cache 到底做了什么

enable_in_batch_prefix_cache 的语义是：已准入的 prefill full block 可在同一个 batch 的后续 request admission 中被共享。Continuous Scheduler 的实现位于 [scheduler_policy.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/scheduler_policy.cpp:820)。

必须区分：

1. block 已分配；
2. block 已被当前 batch 的 model forward 写完；
3. block 已登记到 Prefix Cache map。

同 batch 模式让第 3 步提前发生，以便后续请求 lookup；正确性依赖 batch 中生产者对共享 block 的写入先于消费者读取。现有单测主要证明调度和 block 共享逻辑，不是跨设备数值等价证明。

风险控制：

- 只发布完整 block；
- 禁止 partial block 进入公共 cache；
- consumer 必须仍带正确 kv_cache_tokens_num；
- backend 若使用独立 stream，需要额外 event/分批执行验证。

## 25. Block Manager 的组合模型

CompositeBlockManager 不是简单的 KV map。它根据模型和 runtime 生成多个 leaf，角色判断位于 [composite_block_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/composite_block_manager.cpp:44)。

|BlockType|保存内容|是否天然参与 Prefix Cache|
|---|---|---|
|KV|full attention 的 K/V|PREFILL/MIX 通常参与|
|C4/C128|DSV4 压缩 attention 结构|PREFILL/MIX 参与，需成对|
|SWA|sliding-window K/V|PREFILL/MIX gap-tolerant|
|LINEAR|KDA/GDN/线性 state 槽|PREFILL/MIX 以 chunk checkpoint 参与|
|EMBEDDING|跨模态 embedding cache|不作为 token Prefix Cache|
|XTensor leaf|XTensor 自有存储|当前不参与|

DECODE role 会跳过 SWA、LINEAR、EMBEDDING 等叶子，只保留 KV/C4/C128，源码 predicate 见 [composite_block_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/composite_block_manager.cpp:44)。

## 26. 三种叶子组合

组合分类在 [composite_block_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/composite_block_manager.cpp:269)：

### 26.1 FLAT_KV

只有普通 KV leaf。命中长度由 KV block 数决定，Sequence 直接 mount shared blocks。

### 26.2 FLAT_KV_LINEAR

同时有 KV 和 LINEAR。KV 命中必须被 LINEAR 最深可恢复 checkpoint 截断，否则 attention 已跳过 token 但 recurrent state 没有对应状态。

### 26.3 SWA_COMPRESSED

同时有 SWA、C4、C128。三者命中取最小值，再按 C128 对齐；SWA 尾窗口还要检查 invalid placeholder，最后各 leaf 以同一安全 token 边界 mount。

不满足组合约束时返回 UNSUPPORTED，而不是悄悄使用错误的部分 cache。

## 27. Composite 的 probe、trim、mount 三阶段

### 27.1 Probe

probe_prefix_leaves 遍历支持 Prefix Cache 的 leaf：LINEAR 使用 linear_state_hashes，其他 leaf 使用 block_hashes 和已有 shared slice，见 [composite_block_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/composite_block_manager.cpp:475)。

### 27.2 Trim

不同组合使用不同 trim：

- FLAT_KV 不裁剪；
- FLAT_KV_LINEAR 以 LINEAR checkpoint 可恢复 token 数裁 KV；
- SWA_COMPRESSED 取跨叶子最小命中、C128 对齐、检查 SWA tail，再处理 exact-repeat。

### 27.3 Mount

trim 后的 block 交给 Sequence::add_shared_blocks 或 KVCacheState::mount_composite_shared，并设置共同 cached token cursor，见 [composite_block_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/composite_block_manager.cpp:678)。

## 28. LINEAR：缓存的不是 KV，而是递推状态

线性注意力可抽象为：

~~~text
S_t = update(S_(t-1), k_t, v_t)
y_t = read(S_t, q_t)
~~~

普通 attention 可以按 token 读取历史 K/V；线性 attention 要从某个 S_t checkpoint 继续递推。因此 LINEAR Prefix Cache 的 entry 是 hash 到 state slot，而不是 hash 到一组 token blocks。

LinearStatePrefixCache 的声明在 [linear_state_prefix_cache.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/linear_state_prefix_cache.h:29)。

它允许 invalid placeholder，匹配可跨过状态暂未保存的 chunk，并在最后只保留最深有效 checkpoint，见 [linear_state_prefix_cache.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/prefix_cache/linear_state_prefix_cache.cpp:27)。

## 29. LINEAR 的 deferred save/restore

线性 state 的 live slot 正在被当前请求使用时，不能立即把它同时作为公共 checkpoint。LinearStateBlockManager::allocate_for_sequence 采用 deferred save：

1. 当前 step 记录上一个 checkpoint hash；
2. forward 使用 live state slot；
3. 下一个 allocation 轮转 live slot；
4. 把上一轮完成的 state 以记录的 hash 插入 cache；
5. 为下一段请求取得新的 live slot，并设置 restore source。

源码在 [linear_state_block_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/linear_state_block_manager.cpp:56)。这是“内容完成后才公开”的具体实现。

如果 pinned checkpoint 没有可驱逐槽，插入会 best-effort 失败；这应造成 cache miss 或调度延迟，而不是覆盖活跃 state。相关 pin/LRU 单测见 [linear_state_prefix_cache_test.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/tests/core/framework/prefix_cache/linear_state_prefix_cache_test.cpp:222)。

## 30. LINEAR 的 chunk stride 约束

LINEAR state hash 的一个 entry 对应一个 chunk-end checkpoint。为了让 token cursor、KV block 和 state slot 对齐：

~~~text
max_tokens_per_chunk_for_prefill % block_size == 0
~~~

Engine 对启用 Prefix Cache 加 linear attention 的 PREFILL/MIX 实例做 CHECK，见 [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:527)。

最后一个不足一个 chunk 的 prompt 可以作为最终计算段，但不能把未到边界的中间 state 当成跨请求 checkpoint。

## 31. FLAT_KV_LINEAR 的安全边界

设 KV 命中 1024 token，LINEAR 只有 768 token 有效 checkpoint，prompt 长度 1200。Composite trim 会把 KV 命中从 1024 截到 768，把 768 对应的 LINEAR slot 作为 restore source。

Batch 只能跳过 768 token，剩余 432 token 由模型递推并写入新的 KV/state。

如果只看 KV 命中，就会出现 full attention 从 1024 开始、KDA 却从 768 状态开始的跨层不一致，最终 hidden/logits 错误。

## 32. SWA 与压缩 attention 的安全边界

SWA 只有最近窗口有效；历史滑出后，vector 中可能保留 invalid placeholder。C4/C128 保存压缩结构，需要两者同时存在。

trim_swa_compressed 先求所有叶子命中覆盖量的最小值，再按 C128 block 对齐；如果 SWA 尾窗口中有 invalid entry，就按 C128 步长回退，见 [composite_block_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/composite_block_manager.cpp:575)。

这解释了一个现象：hash 命中 20 个 block，最终 kv_cache_tokens_num 可能只有 16 个 block 的 token。少跳过一些是为了保证窗口尾部和压缩状态完整。

## 33. BatchInputBuilder 如何真正跳过 token

process_single_sequence 读出 sequence token 总数和 n_kv_cache_tokens，计算：

~~~text
q_seq_len = min(num_tokens - n_kv_cache_tokens, allowed_max_tokens)
logical_seq_len = q_seq_len + n_kv_cache_tokens
~~~

源码入口为 [batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:680)。extract_tokens_and_positions 只切取 cached 之后的 token/position。

state_to_forward_input 在 [batch_input_builder.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/batch/batch_input_builder.cpp:1190) 输出：

- token_ids：suffix token；
- q_seq_lens：本步 query 长度；
- kv_seq_lens：历史加本步长度；
- kv_cache_tokens_nums：每个 request 的历史 cache 长度；
- new_cache_slots：本步写入位置；
- block_tables：包括 shared 和新 block 的物理映射；
- paged_kv_indptr/indices/last_page_len：paged backend 页元数据。

Prefix Cache 若只修改 BlockManager 而不修改这些字段，模型仍会重复计算或读取错误位置。

## 34. Attention backend 如何消费命中元数据

普通 backend 使用 block_tables 找到历史 KV，并用 kv_cache_tokens_nums 区分历史长度和本步 Q。Forward 参数结构在 [model_input_params.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/model/model_input_params.h:355)。

设备侧抽象为：

~~~text
history KV: block_tables[:cached_blocks]
new KV:     new_cache_slots for suffix tokens
query:      q_seq_lens
key length: kv_cache_tokens_nums + q_seq_lens
~~~

FlashInfer 的 paged metadata 还要满足 paged_kv_indptr 的 batch 前缀和、paged_kv_indices 的物理页 id、last_page_len 的尾页长度。Prefix hit 改变的是历史页数量和 query 长度，不能只替换 block table。

Worker 对 linear attention 额外准备 has_initial_state，要求 host/device kv_cache_tokens_nums 的 batch 维度一致或可整除，见 [worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:214)。

## 35. MLA 场景：对象是 latent/rope cache

MLA 不一定保存传统 [K,V] 两个完整 head tensor。Kimi K3/DeepSeek 风格路径会把压缩 latent KV 和 positional/rope 部分分别分页保存。

Worker 在 Prefix Cache metadata 存在时为 MLA 准备历史压缩 KV、历史 rope key、ring sequence length，见 [worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:1785)。

这意味着：

- 命中 token 数必须和 latent KV、rope cache 的页表一致；
- TP 分片后的本地 query head 数不能反过来改变 cache geometry；
- reshape_paged_cache、FIA/FlashInfer metadata 和 Prefix Cache mount 必须联合验证；
- 短请求 HTTP 200 不能证明多页 MLA prefix hit 正确。

当前 Kimi K3 离线逐层 dump 在第 4 层 Full MLA 的 reshape_paged_cache 报 NPU Invalid argument，Prefix Cache 尚未进入 E4 数值验证。

## 36. 模型侧显式分支

模型实现有时需要根据 Prefix Cache 调整 mask 或 cache layout。例如 NPU DeepSeek V2 在启用 Prefix Cache 且非 decode 时选择更大的 attention mask，见 [deepseek_v2.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/models/llm/npu/deepseek_v2.h:175)。

这不是 Prefix Cache 改了模型数学，而是模型要知道 query 是 append 到已有 history：

- causal mask 形状由 q_len 和 kv_len 决定；
- prefix 命中后 query 不再从 position 0 开始；
- 某些融合 kernel 对 prefix path 使用专用 mask/bucket；
- MTP、speculative、CP 可能需要另一个 block table 视图。

相反，DeepSeek V3 在非 DCU 平台构造函数中直接 CHECK 禁用 Prefix Cache 与 chunked prefill，见 [deepseek_v3.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/models/llm/deepseek_v3.h:29)。

## 37. XTensor、VLM 和 embedding 边界

LLM Engine 在启用 XTensor 时把 BlockManager 的 Prefix Cache 强制关闭，见 [llm_engine.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/llm_engine.cpp:554)。XTensor 有自己的 cache/图布局，普通 block map 不能安全接管。

VLM 组合还需要同时考虑：

- text token hash；
- MM schedule key；
- embedding/cache leaf；
- cross-attention block table；
- 图片预处理是否确定性。

EMBEDDING leaf 不等于 token Prefix Cache，也不应因为 embedding slot 命中就增加 kv_cache_tokens_num。

## 38. Prefix Cache 与 beam/best-of

best-of/beam 的公共 prompt 适合 Prefix Cache，但分支之后的状态不能互相写入。Sequence copy 会共享已有 KV handle，并为每个分支重新管理需要写入的 block/state。

PD decode 端对 best_of>1 有明确要求：只从 P 拉取第一条 sequence 的 prompt KV，剩余分支依赖 decode 实例 Prefix Cache 本地展开；若关闭 Prefix Cache，服务直接拒绝该请求，见 [disagg_pd_service_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/distributed_runtime/disagg_pd_service_impl.cpp:33)。

这说明在某些 PD + best-of 组合中 Prefix Cache 是资源协议的一部分，而不仅是性能开关。

## 39. Prefix Cache 与抢占/Zero-Eviction

抢占可能释放 sequence owner，但不一定立即删除 cache owner。Zero-Eviction scheduler 在候选请求上做 prefix probe，并把候选 block 纳入 capacity simulation，见 [zero_eviction_scheduler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/zero_eviction_scheduler.cpp:121)。

当前容量 guard 相对保守：Prefix Cache 开启时按 reserved block 数计算，而不是简单把命中 block 当成永远免费。这避免模拟阶段过度乐观，也意味着高命中请求仍可能因保守容量被延迟。

排查 admission 失败时要看：

- prefix hit 后仍需多少 suffix block；
- shared block 是否被其他 sequence pin；
- cache eviction 能否释放物理 block；
- Zero-Eviction guard 是否把 reserved prefix block 计入；
- linear live slot 是否比 KV block 更早耗尽。

## 40. Prefix Cache 与 DP/TP/EP/CP 通信

### 40.1 TP

每个 TP rank 都需要拥有与本地 KV 分片对应的物理 block。lookup 元数据通常在各 rank 本地执行，不增加 AllReduce；命中后减少每个 rank 的 prefill GEMM 和 attention，间接减少 TP collective。

### 40.2 EP

命中减少 MoE token，可能减少 expert dispatch 的 AllToAll payload；剩余 suffix 的 token 路由仍需完整执行。Prefix Cache 不改变 token 到 expert 的路由结果。

### 40.3 DP

不同 DP rank 的 Prefix Cache 通常独立。当前源码搜索不到 enable_prefix_cache_aware_dp_routing；旧 docs 关于 prefix-aware DP routing 的段落属于残留/陈旧路径，不能作为当前支持证据。

### 40.4 CP

NPU CP 会生成 prefix_cache_slots metadata，见 [npu_cp_plan.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/parallel_state/npu_cp_plan.cpp:242)。它把逻辑页映射到 CP/KV split 计划中，避免 rank 使用不一致的 local offset。

## 41. Prefix Cache 与 PD 分离

PD 的关键问题是 P 实例产生的 KV 必须在 D 实例可读。Prefix Cache 有三种位置：

1. P 本地命中：P 少算公共 prompt，再发送更少的新 KV 或状态。
2. D 本地命中：D 收到请求后直接复用已有 prefix，减少本地前置工作。
3. 跨实例传输：P/D 不共享 HBM，需要 transfer protocol 搬运 block/state。

Disagg scheduler 在请求 metadata 中发送 num_cached_tokens，见 [disagg_pd_scheduler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/disagg_pd_scheduler.cpp:675)。D 端接收后记录 usage，见 [disagg_pd_scheduler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/disagg_pd_scheduler.cpp:895)。

通信成本：

~~~text
P->D bytes = uncached_prefill_KV + required_state_transfer
~~~

若本地重算时间小于远端传输时间，跨实例 Prefix Cache 反而可能增加 TTFT。

## 42. Host offload 和 global cache 的区别

host_blocks_factor>1 会建立分层 host KV transfer，但当前 Worker 初始化时把 enable_kvcache_store(false) 写死，见 [worker_impl.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/runtime/worker_impl.cpp:1785)。

|类型|内容|主要成本|
|---|---|---|
|HBM Prefix Cache|设备 block handle|占 HBM，命中快|
|Host offload|KV 在 CPU/NUMA 内存|PCIe/HCCS/NPU copy，需同步|
|Global/remote store|跨实例或持久服务|网络、序列化、版本/一致性协议|

多级 cache 只能在具体 transfer 初始化和实际 read/write 路径存在时标为条件支持。

## 43. 通信、overlap 与依赖关系

### 43.1 本地路径

CPU hash、unordered_map lookup、LRU 操作与设备上一轮 decode 理论上可并行；但当前代码没有把每个 Prefix Cache API 自动放入独立异步线程作为稳定契约，是否 overlap 取决于 Scheduler tick、线程池和 batch builder。

### 43.2 Host restore

host-to-device copy 可放在 prepare stream，与权重/metadata 准备重叠；Attention 消费 block 前必须由 event 或 stream dependency 保证完成。非阻塞 copy 不能代替数据依赖。

### 43.3 PD transfer

P/D 之间可以异步发送 KV block，D 在收到最后一个所需 block 后才允许把 sequence 交给 decode batch。Overlap 的证据应包含 transfer start/end、forward start/end 和依赖 event；接口存在不代表已经隐藏传输延迟。

### 43.4 CP/TP

Prefix hit 减少 token，间接减少 collective payload；它不自动把剩余 token 的 AllGather/AllReduce 变成异步。Graph capture 时 block table/prefix slots 还可能必须固定 shape，动态命中率高时会增加 graph bucket miss。

## 44. 内存模型：省计算换驻留

对一个 KV leaf，粗略 resident bytes：

~~~text
bytes_per_block
  = block_size
  * n_layers_in_leaf
  * (K_elements_per_token + V_elements_per_token)
  * bytes(dtype)
  / tensor_parallel_sharding_factor
~~~

MLA、量化、C4/C128 会改变每 token 元素数，不能直接套普通 MHA 的 2*heads*head_dim。

Prefix Cache 总成本还包括 HBM block、host block、linear state slots、hash map/LRU node、block table、pinned history buffer、graph persistent buffer 和不同 leaf 对齐造成的碎片。

max_linear_state_cache_slots 只限制线性 checkpoint 槽，不等价于 KV block 数限制，见 [kv_cache_config.h](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/config/kv_cache_config.h:69)。

## 45. 计算收益模型

设请求 prompt 长度为 L，命中 H，suffix 为 S=L-H，单 token prefill 成本为 Cp，lookup/restore 成本为 Cl：

~~~text
无缓存成本   ≈ L * Cp
本地命中成本 ≈ S * Cp + Cl
节省比例     ≈ H/L - Cl/(L*Cp)
~~~

当 H 很小、prompt 很短时，hash 和调度开销可能超过收益；当 H 很大且 suffix 仍需至少一个 token 时，TTFT 才明显下降。

吞吐还要扣除 cache 占用造成的 active sequence 容量变化：

~~~text
effective throughput = compute_saved - preemption_penalty - transfer_penalty
~~~

benchmark 必须同时报告 token hit ratio、HBM used、active batch size 和 tail latency。

## 46. 精度分类和风险矩阵

### 46.1 布局/地址错误

block table、slot mapping、CP prefix slots 错位会读取另一段 KV，属于逻辑错误，通常输出完全错误。

### 46.2 状态边界错误

FLAT_KV_LINEAR 没按 LINEAR checkpoint 截断，或 SWA tail 含 invalid block，会造成跨层状态不一致，属于逻辑正确性错误。

### 46.3 dtype/量化差异

复用同一低精度 KV 不会新增误差，但和每次重算后转换的路径相比可能有不同 rounding。INT8/FP8 scale、MLA latent dtype、C4/C128 layout 必须纳入 cache namespace 或确保 Engine 不混用。

### 46.4 hash 碰撞

XXH3-128 碰撞概率极低，但系统正确性不应只依赖概率。更常见的风险是 key 缺少 adapter、模型 revision、位置或多模态 schedule key。

### 46.5 测试判定

逐 token greedy 对齐可证明逻辑一致；随机 sampling 还需比较完整 logits 分布。只比较 HTTP 200、token 数或命中率不能证明精度。

## 47. 模型/特性兼容矩阵

|特性|Prefix Cache 状态|证据与注意事项|
|---|---|---|
|普通 full KV|已支持|PrefixCache、BlockManager、Batch builder 可达|
|Chunked Prefill|条件支持|普通 KV 可 rematch；长 chunk 需验证|
|KDA/LINEAR|条件支持|需 chunk 对齐、state slot 与 restore 验证|
|MLA latent KV|条件支持|需验证 latent/rope paged metadata；K3 当前未完成|
|SWA/C4/C128|条件支持|Composite trim 有安全边界，模型组合需测|
|VLM/MM|条件支持|需 MM schedule key 和 cross-attention cache|
|XTensor|已禁用|Engine 强制关闭普通 Prefix Cache|
|DeepSeek V3 非 DCU|已禁用|模型构造 CHECK|
|PD best-of>1 decode|条件支持|D 端 Prefix Cache 必须开启|
|跨 DP cache-aware routing|未确认|当前源码未找到 docs 宣称的 flag|
|远端 global store|残留/陈旧路径|当前 worker 将 store 设为 false|

## 48. 旧文档与当前源码的冲突

### 48.1 Murmur hash

官方 feature 文档仍写 mermer_hash/murmur_hash，见 [prefix_cache.md](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/docs/src/content/docs/zh/features/prefix_cache.md:7)。当前 block_hasher.cpp 明确调用 XXH3_128bits_withSeed，所以知识库以源码为准。

### 48.2 Prefix-aware DP routing

文档列出 enable_prefix_cache_aware_dp_routing、match threshold、imbalance threshold，但当前分支 rg 不存在这些符号。不能把这些参数作为当前部署开关。

### 48.3 Qwen3 10% 声明

文档中的 E2E 下降 10% 缺少当前 commit、硬件、流量分布和基线，不应当作 Kimi K3 或当前环境的 benchmark 结果。

## 49. 观测指标和 Request usage

Request 在 record_num_prefix_cache_tokens 时记录命中 token，并采用 max 语义，见 [request_prefix_cache_tokens_test.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/tests/core/framework/request/request_prefix_cache_tokens_test.cpp:110)。即使之后 KV block 被释放，Request 已记录的 usage 仍保持。

建议监控：

- prefix_cache_block_matched_rate；
- prefix_cache_block_matched_num；
- request token hit rate 和 token hit ratio；
- resident block、eviction、pinned skip；
- linear checkpoint hit/deepest restore；
- host/PD transfer bytes 和等待时长；
- hit/miss TTFT、TPOT、TTLT、P99；
- prefix 占用导致的 preemption、allocation rollback、graph fallback。

请求命中率高但 token hit ratio 低，说明多数请求只命中一个短 block；block 数增加但 TTFT 不降，可能是后端仍发送了全 prompt。

## 50. 测试证据层级

### E0：接口存在

配置、PrefixCache API、Batch metadata 字段存在。

### E1：可达路径

Engine 构造 BlockManager，Scheduler 调用 allocate_shared，Batch builder 输出 hit metadata。

### E2：本地正确性

PrefixCache 基本 insert/match/evict、precomputed hash 与计算 hash 等价测试在 [prefix_cache_test.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/tests/core/framework/prefix_cache/prefix_cache_test.cpp:429)。

### E3：请求 smoke

在线服务 HTTP 200、usage 中 cached tokens、重复 prompt 能减少 q token；这还不是数值等价。

### E4：数值正确性

关闭/开启 Prefix Cache 的同一 prompt/suffix，逐层 hidden、logits、输出 token 在容差内一致；LINEAR、MLA、SWA 要覆盖 state restore 和多页。

### E5：性能稳定性

固定流量、长时间运行，比较 TTFT、TPOT、吞吐、P99、HBM、eviction、preemption，并验证 graph、PD、host transfer 没有退化。

## 51. 单元测试覆盖了什么

PrefixCache 单测覆盖空 cache miss、插入和独立匹配、LRU eviction、重复 insert 幂等、预计算 hash 等价、binary/linear probe 边界。

LinearStatePrefixCache 单测覆盖 padding slot 0、live slot 释放、checkpoint dedupe、LRU reclaim、pinned checkpoint 保护、全 pinned 时 insert 失败。

Request test 证明 usage 在 block release 后保留，并且 Prefix Cache disabled 时 cached tokens 为 0，见 [request_prefix_cache_tokens_test.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/tests/core/framework/request/request_prefix_cache_tokens_test.cpp:147)。

## 52. 还缺什么测试

当前单测没有完整覆盖：

1. MLA latent KV 加 rope cache 的关闭/开启等价；
2. KDA LINEAR 跨两个以上 chunk 的恢复等价；
3. SWA placeholder 与 C128 trim 的数值结果；
4. in-batch producer/consumer 在真实 NPU stream 上的读写依赖；
5. DP 多 rank 独立 cache 的命中/负载行为；
6. PD P 到 D 传输与 D 本地命中共同存在；
7. LoRA、adapter、model revision 切换后的隔离；
8. graph capture 在不同 prefix hit length 下的 fallback；
9. 长 prompt、多页、prefill 到 decode 的长期稳定性。

## 53. 数值例子：普通 KV + chunked prefill

设 block size=128、prompt=640 token、第一请求冷启动、chunk budget=256、第二请求公共 prefix=512。

第一请求：

~~~text
step 1: q=256, cached=0,   申请 2 blocks
step 2: q=256, cached=256, 申请 2 blocks
step 3: q=128, cached=512, 申请 1 block
发布: 5 个完整 block
~~~

第二请求：

~~~text
match: H0..H3 hit = 512 token
step 1: q=128, cached=512
~~~

第二请求少算 512 token，但仍需读取 512 token 历史 KV、写第 5 个 block，并计算 logits。若 chunked_match_frequency=2，普通 leaf 可能在中间 step 不重新 hash；LINEAR leaf 则必须在 checkpoint 边界重新确认。

## 54. 数值例子：KV 命中超过 LINEAR checkpoint

设 block size=128，KV 命中 1024 token，LINEAR hashes 只有 256、512、768 三个有效 checkpoint。

~~~text
KV candidate       = 1024
LINEAR recoverable = 768
safe_hit           = min(1024, 768) = 768
~~~

Batch 设置 kv_cache_tokens_num=768，q_seq_len=prompt_len-768，linear restore source 为 hash@768 的 slot。

若 source slot 被 pinned，仍可挂载；若被驱逐，LINEAR 命中退化，KV 也必须被裁到可恢复边界。

## 55. 时序图：一次命中请求

~~~text
Scheduler                 BlockManager                 Batch/Worker              Device
    |                           |                            |                      |
    |-- probe tokens ---------->|                            |                      |
    |                           |-- hash/map/LRU ------------|                      |
    |<-- shared handles --------|                            |                      |
    |-- allocate suffix ------->|                            |                      |
    |------------------------------------------------------->| build q/cache/table  |
    |                           |                            |-- copy metadata ----->|
    |                           |                            |-- model forward ----->|
    |                           |<---------------------------| new blocks complete   |
    |-- cache full blocks ----->|                            |                      |
    |                           |-- insert/LRU/refcount ---->|                      |
~~~

任何一步缺失都会出现不同故障：没有 shared handles 就是重复计算；没有 q/cache metadata 就是 position 错；提前 insert 就是读未完成数据；没有 refcount 就是 use-after-free。

## 56. 常见错误定位

### 56.1 命中数为零

检查 tokenizer/chat template、block size、seed、Sequence hash 是否更新、cache 是否被模型/角色 gating 关闭，以及第一个完整 block 是否存在。

### 56.2 命中但 q token 仍等于完整 prompt

检查 kv_cache_tokens_num 是否在 Sequence mount 后更新，Batch builder 是否使用 tokens[cached:]，以及模型是否覆盖了自己的 input builder。

### 56.3 命中后输出错误

优先检查 block table 与 slot mapping、KV layout/dtype、MLA latent/rope 两套 cache、LINEAR restore source、SWA tail、CP remap。

### 56.4 cache 越用越满且无法驱逐

查看 Block refcount、pinned shared handle、beam 分支、host alias、probe 临时 vector 是否及时释放。

### 56.5 开启后性能更差

检查短 prompt、低命中率、hash CPU 开销、host/PD transfer、保守 capacity guard、graph bucket miss 和 eviction 抖动。

## 57. 参数调优方法

### 57.1 先固定正确性

使用小 block、单 rank、关闭 graph/PD/CP，跑 cold/warm 对照；确认 token、hidden、logits 一致后再放大并发。

### 57.2 再选择 block size

用生产流量统计公共 prefix 长度 P50/P90/P99，比较 floor(P/B)*B 的 hit token 与 block metadata 开销。不要只按 KV block 最大容量选择 B。

### 57.3 最后调 cache 容量

观察 HBM headroom、active sequence 数和 eviction rate。Prefix Cache 占满时保留热点长前缀；如果它导致频繁抢占，降低 cache 容量可能提高总吞吐。

### 57.4 线性模型单独调槽

max_linear_state_cache_slots 太小会让 state checkpoint 经常 miss；太大则挤压 KV/HBM。按活跃 live slots、热 checkpoint slots 和 padding slot 估算。

## 58. 生产验收清单

- [ ] 配置实际生效：日志打印 block size、prefix flag、linear slots、hasher type。
- [ ] 冷请求发布完整 block，尾 partial 不进入 cache。
- [ ] 第二请求命中后 q_seq_len 只包含 suffix。
- [ ] block table、paged indices、new slots 与 kv_cache_tokens_num 一致。
- [ ] exact-repeat 保留至少一个 token 计算。
- [ ] chunked prefill 在两个以上 chunk 后仍能命中并继续。
- [ ] LINEAR restore source 与 KV safe boundary 一致。
- [ ] SWA/C4/C128 组合在滑窗边界不读取 invalid block。
- [ ] beam/best-of 分支不共享可写 live state。
- [ ] in-batch 模式经过真实设备依赖测试后再打开。
- [ ] PD/host transfer 的字节数、等待时长和错误回滚有指标。
- [ ] model/adapter/revision 切换会 reset 或隔离 cache。
- [ ] 关闭/开启 Prefix Cache 的逐层数值对照通过。
- [ ] 长时间压测没有 refcount 泄漏、HBM 碎片和 P99 回退。

## 59. Kimi K3 当前状态

Kimi K3 同时包含 KDA LINEAR 层和 Full MLA 层。根据当前 [research/PROJECT.md](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/research/PROJECT.md:1) 的已验证记录：

- 五层 TP4 在线服务 health、models、max_tokens=1/2 请求通过；
- KDA cache、MLA latent/rope cache 均完成分配；
- 离线逐层 dump 在第 4 层 Full MLA 的 reshape_paged_cache 报 NPU Invalid argument；
- 尚未完成 Full MLA 修复后的 Prefix Cache warm-hit/cold-miss 对照；
- 尚未完成多 chunk、长 decode、PD、CP、in-batch Prefix Cache 证明。

因此 Kimi K3 的 Prefix Cache 状态应标为：

> 条件支持，代码可达、请求级专项未验证。

下一步应先修复 MLA paged metadata/reshape，再做：

1. Prefix Cache off/on 的逐层 hidden/logits 对齐；
2. 4 个 KDA chunk 加 1 个 MLA chunk 的共同前缀命中；
3. exact-repeat、suffix 分叉和 decode 连续运行；
4. TP4 HBM、TTFT、命中 token、P99 对照。

## 60. 读者自测问题

1. 为什么 H2 命中而 H1 miss 时不能使用 H2？
2. 为什么 Prefix Cache 只能匹配完整 block？
3. kv_cache_tokens_num 和 cached_tokens 有什么不同？
4. exact-repeat 为什么还要算至少一个 token？
5. 为什么 LINEAR state 必须和 KV 命中取安全交集？
6. SWA 的 invalid placeholder 为什么不能直接当 cache miss 后继续读后面的 block？
7. in-batch Prefix Cache 的真正设备依赖是什么？
8. 为什么命中 Prefix Cache 会间接减少 TP/EP 通信，却没有新增 collective？
9. 为什么 host offload 和 global cache 不能用同一个命中率解释？
10. 哪个源码事实证明旧文档的 Murmur hash 描述已经过时？
11. Kimi K3 为什么必须同时验证 latent KV、rope cache 和 LINEAR restore？
12. 如何证明 Prefix Cache 没有改变 logits，而不仅仅是返回 HTTP 200？

## 61. 推荐源码阅读顺序

~~~text
kv_cache_config.h/.cpp
  -> block_hasher.h/.cpp
  -> prefix_cache.h/.cpp
  -> block.h/.cpp
  -> sequence.h/.cpp + sequence_kv_state.h/.cpp
  -> block_manager_impl.h/.cpp
  -> linear_state_prefix_cache.*
  -> linear_state_block_manager.*
  -> composite_block_manager.*
  -> scheduler_policy.cpp / unified_policy.cpp
  -> batch_input_builder.cpp
  -> worker_impl.cpp
  -> model attention headers
  -> PD/CP/host transfer
  -> unit tests and request-level differential test
~~~

读到每个函数时都问五个问题：谁拥有 block、hash 的单位是什么、cursor 的单位是什么、设备读写哪一个字段、失败时如何回滚。

## 62. Cache identity 审计：当前 key 包含什么

从 BlockHasher 的真实输入可以列出当前身份字段：

|身份字段|是否直接进入 hash|说明|
|---|---|---|
|token id 序列|是|以 int32 字节参与 XXH3|
|前一个 block hash|是|构成完整 prefix chain|
|全局 XXH3 seed|是|由 KVCacheConfig 提供|
|MM schedule_data.key|MM 模式是|只加入与 block 重叠的 item|
|模型 id/revision|否|通常依赖 Engine 实例隔离|
|tokenizer/chat template id|否|只通过最终 token id 间接体现|
|LoRA/adapter id|否|相同 token 在不同 adapter 下可能产生不同 KV|
|RoPE scaling/position policy|否|位置通常由 token offset 推导，但配置不在 key|
|KV dtype/quant scale|否|依赖同一 Engine/cache geometry|
|租户/安全域|否|当前 map 本身不做租户隔离|

这张表的结论不是当前实现必然错误，而是它有一个前提：

> 一个 PrefixCache 实例的生命周期内，所有会改变 KV/state 的模型语义必须保持不变。

如果服务支持在线热切权重、每请求 adapter、动态 RoPE 配置或不同租户的强隔离，就要在更上层拆 cache namespace，或扩展 hash salt。

### 62.1 Tokenizer 不在 key 中为何通常仍安全

Prefix Cache 接收的是 token id，不是原始文本。不同文本只要产生相同 token id 和相同模型输入语义，就可以复用。

但是 tokenizer revision 变化后，同一 token id 的词表语义可能变化；如果模型 embedding 也同步变了，旧 cache 必须清空。

### 62.2 Adapter 是高风险项

LoRA 会改变 Q/K/V、MLP 或其他投影。即使 token id 完全相同，不同 adapter 产生的 KV 也可能不同。

当前 hash 没有 adapter id，因此部署时不能让不同 adapter 请求无隔离地共用同一 PrefixCache，除非源码另有每 adapter Engine/worker 隔离。

### 62.3 多租户安全

Prefix Cache 不返回原始 token，但跨租户时间差、命中指标或共享 HBM 状态仍可能形成侧信道。

高安全场景应考虑：

- 每租户/安全域独立 cache；
- 禁止公开细粒度命中信息；
- reset 时确保 block 内容不会被错误地址重读；
- 对远端 store 增加认证、加密和 namespace；
- 对 eviction 和 usage 指标做权限隔离。

## 63. Cache reset 与权重切换

ContinuousScheduler 暴露 reset_prefix_cache，并将操作转发给 KV cache manager，见 [continuous_scheduler.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/scheduler/continuous_scheduler.cpp:727)。

CompositeBlockManager 会逐个 reset leaf Prefix Cache，见 [composite_block_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/composite_block_manager.cpp:884)。

reset 的语义应是：

1. 删除 hash map 和 LRU owner；
2. 释放 cache 持有的 Block handle；
3. 活跃 Sequence 持有的 shared handle 仍按 refcount 生存；
4. 最后一个 owner 消失时 block 回 free list；
5. 新请求不能再由旧 hash 命中。

权重热切换时必须先阻止新请求进入旧 Engine，再等待/迁移活跃请求，最后 reset 或销毁 cache。直接在活跃 cache 上替换权重会让同一 hash 指向旧权重生成的 KV。

reset 还应在以下变化后执行：

- model checkpoint 或 revision 变化；
- adapter namespace 改变；
- tokenizer/vocabulary 与 embedding 变化；
- RoPE、attention mask、KV dtype、quant scale 变化；
- block size 或 cache geometry 变化；
- 多模态 encoder/preprocessor 版本变化。

## 64. 容量失败和事务式回滚

混合模型一次请求可能同时需要 KV block、LINEAR live slot、SWA/C4/C128 block。若只分配成功一部分就提交，后续 Batch 会在 capacity CHECK 失败，或者部分资源泄漏。

Composite allocate_sequence 使用 staged vector：

~~~text
for each leaf:
    allocate temporary blocks
    if any leaf fails:
        deallocate all staged blocks
        return false

verify every token-bearing leaf capacity
commit staged blocks into Sequence
~~~

实现位于 [composite_block_manager.cpp](/export/home/weinan5/penghengduo/CODE/xllm-KIMI-K3/xllm/xllm/core/framework/block/composite_block_manager.cpp:366)。

Prefix Cache 命中也需要类似的原子性：

- probe 所有 leaf；
- 计算共同 safe_hit；
- 把超出边界的临时 handle 还给 leaf；
- 设置 LINEAR restore source；
- 最后统一 mount。

如果中途异常，临时 probe handle 必须释放，不能留下一个只在 KV leaf 生效、LINEAR leaf 未生效的半命中状态。

### 64.1 容量指标为何不能只看 free KV blocks

实际 admission 的瓶颈可能是：

- KV free blocks；
- LINEAR state slots；
- C128 压缩 block；
- SWA window blocks；
- host transfer buffer；
- graph persistent metadata；
- pinned cache 无法 eviction。

因此日志要分 leaf 打印 total/free/used/shared/cache blocks，而不是只有一个总 free 数。

## 65. In-batch 模式的设备验证方案

In-batch Prefix Cache 是风险最高的优化之一，因为 consumer 可能在 producer forward 完成前就已把 block 视为 shared。

建议构造两个同 batch 请求：

~~~text
A = [公共前缀 2 blocks, suffix A]
B = [公共前缀 2 blocks, suffix B]
~~~

验证步骤：

1. 关闭 in-batch，保存 A/B logits 和每层输出；
2. 开启 in-batch，强制 A 先 admission、B 后 admission；
3. 确认 B 的 q_seq_len 只包含 suffix；
4. dump A/B block table，确认公共 block id 相同；
5. 在设备 timeline 标记公共 block 写入和 B attention 读取；
6. 确认写完成依赖先于读；
7. 对比每层 hidden、final logits、greedy token；
8. 交换 A/B admission 顺序再测；
9. 加入 graph capture、overlap stream 和多 TP rank 再测；
10. 做错误注入：延迟 producer stream，确认 event 能阻止 consumer 提前读。

只有步骤 5 至 7 才能把“调度级共享”提升为“设备级正确”。

## 66. 性能 benchmark 设计

一个有意义的 Prefix Cache benchmark 至少包含四组流量：

|流量组|公共前缀|目的|
|---|---:|---|
|Cold|0%|测 hash/cache 管理的纯开销|
|Short-hit|1 个 block|测低收益边界|
|Long-hit|50%-90% prompt|测典型 TTFT 收益|
|Churn|大量一次性长前缀|测 eviction 和 HBM 压力|

控制变量：

- 相同模型、权重、dtype、block size；
- 相同 prompt/output 长度分布；
- 相同并发和 arrival process；
- 相同 TP/DP/EP/CP/PD 拓扑；
- 相同 graph、chunked prefill、speculative 配置；
- 预热后再统计 steady state。

报告指标：

- TTFT P50/P90/P99；
- TPOT/ITL 和 TTLT；
- prompt token throughput、decode token throughput；
- token hit ratio、request hit ratio；
- Prefix Cache block resident/eviction/pinned；
- HBM peak、host memory、workspace、碎片；
- TP/EP collective bytes 和 duration；
- PD/host transfer bytes、bandwidth、wait；
- Scheduler admission delay、preemption、rollback；
- graph capture/replay/fallback 比例。

性能结论必须写清硬件、commit、模型、请求分布和置信区间。一个固定 system prompt 的 10% 收益不能外推到随机短 prompt。

## 67. 在线服务的 curl 验证方法

Prefix Cache 是内部状态，curl 本身通常不需要新增字段；用两个具有公共完整 block 的普通 Chat Completion 请求即可触发。

验证思路：

1. 启动服务并明确 enable_prefix_cache=true；
2. 选择长度超过两个 block 的固定 system prompt；
3. 发第一个 cold 请求，记录 usage、TTFT、服务日志；
4. 保持 system prompt token 完全相同，只改变 user suffix；
5. 发第二个 warm 请求；
6. 从日志/metrics 确认 cached token 数和 q_seq_len；
7. 用 enable_prefix_cache=false 的服务作为结果对照。

示意请求：

~~~bash
curl -sS http://127.0.0.1:13736/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kimi",
    "messages": [
      {"role": "system", "content": "<固定长 system prompt>"},
      {"role": "user", "content": "第一个 suffix"}
    ],
    "temperature": 0,
    "max_tokens": 2
  }'
~~~

第二次只把“第一个 suffix”改成“第二个 suffix”。要保证 system prompt 经 chat template 后仍位于相同 token 位置，并且长度跨过完整 block 边界。

curl 返回 HTTP 200 只证明服务链可达。必须联合查看 cached token usage、Batch q 长度、block table 和数值对照。

## 68. 术语表

|术语|本文含义|
|---|---|
|Prefix|从 token 0 开始连续相同的前缀|
|Solid prefix|第一个 miss 后立即停止，不允许中间空洞|
|Gap-tolerant|可保留 invalid entry 并寻找更深有效 checkpoint|
|Block hash|某完整 token block 加此前 hash 的 XXH3-128|
|Shared block|至少两个 owner 持有引用的物理 block|
|Cached token cursor|模型已经有可读历史状态的 token 数|
|Cache publication|把已完成 block/state 登记到 Prefix Cache|
|Probe|临时查询各 leaf 候选命中，不代表最终可跳过长度|
|Trim|按混合状态共同可恢复边界裁剪候选命中|
|Mount|把最终 shared handle 挂到 Sequence|
|Live slot|当前请求可写的线性 state 槽|
|Checkpoint slot|按 prefix hash 保存、可供新请求恢复的只读状态槽|
|Pinned|仍被活跃 owner 引用，不能 eviction|
|In-batch cache|同一 Scheduler batch admission 内的前缀共享|
|E4|请求/模型级数值正确性证据|
|E5|长时间性能和稳定性证据|

## 69. 最终总结

Prefix Cache 的本质不是一个 hash map，而是一套跨层状态协议：

~~~text
token/MM identity
  -> chained hash
  -> LRU block/state ownership
  -> scheduler safe admission
  -> batch suffix slicing
  -> model-specific KV/state restore
  -> backend paged metadata
  -> completed-block publication
~~~

普通 KV 解决的是复用已完成的分页 K/V；LINEAR 解决的是从可恢复 checkpoint 继续递推；SWA/C4/C128 解决的是多种窗口/压缩状态取共同安全边界；MLA 还要同时维护 latent 与 positional cache。通信、overlap、内存和精度收益都必须沿这条链计算，不能只看 PrefixCache::match 的返回长度。

对当前 xLLM，最可靠的判断方式是：先看源码组合与 CHECK，再跑 E2 单测，再做目标模型 E4 数值对照，最后才用真实流量评估 E5 性能。对 Kimi K3，通用链路已经清楚，Full MLA paged 执行和 KDA+MLA 联合命中仍是必须完成的专项验证。
