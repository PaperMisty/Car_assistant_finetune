"""
智能汽车客服助手 - 数据合成 Generator Prompt 模块 (v2 优化版)
用于将领域 Seed Schema 转化为高质量、多轮对齐、具备服务闭环的 SFT 训练数据
"""

import json
from typing import Any, Dict, List, Optional


class SFTDataGeneratorPrompt:
    """
    SFT 数据合成生成器 Prompt 管理类 (v2 增强版)
    针对对话自然性、内在逻辑消重、端到端服务闭环进行了重点优化
    """

    # 1. 生成器系统提示词 (System Prompt for Generator)
    GENERATOR_SYSTEM_PROMPT: str = """你是一个顶级的 AI 数据工程专家，精通大模型垂直领域 SFT（监督微调）合成数据的构造。
你的任务是：根据给定的【汽车售后客服业务种子规范（Seed Schema）】，合成一段高质量、自然逼真、逻辑连贯且具备**完整服务闭环**的多轮客服对话数据。

### 【核心生成原则与质量标准】
1. **真实自然的口语化交互（严禁“查户口”式连续盘问）**：
   - 客户（user）初次提问通常带有情绪或表述模糊；
   - 客服（assistant）在信息不足时，**单轮最多追问 1~2 个最关键的问题**，使用温和耐心的口吻，严禁一次性生硬抛出一长串问题清单。
2. **严格的上下文逻辑消重（严禁“复读机”与无视已知信息）**：
   - 客服在后续轮次中，**严禁再次追问客户前面已经交代过的信息**（例如客户已说明是“刚买的二手车首次绑定”，客服不得再反问“您是首次绑定还是解绑”）。
3. **完整的端到端服务闭环（严禁半截子对话）**：
   - 对话必须包含完整的生命周期：
     `客户提问` -> `客服追问/调工具` -> `客户补充` -> `客服给方案/执行指导` -> `客户反馈操作结果（如“连上了/还是不行”）` -> `客服收尾（致谢祝好/转人工预约）`。
4. **事实依据（required_facts）与安全红线（prohibited_actions）**：
   - 客服在解释原因或指导操作时，必须基于种子中的客观机理（required_facts）；
   - 严格禁止触犯 prohibited_actions（如索要密码、行驶中引导操作、未查系统擅自承诺等）。
5. **标准工具调用协议（Tool Calling）**：
   - 若 `tool_required` 为 `true`，必须包含规范的工具调用轮次：
     `assistant 输出 tool_calls` -> `tool 返回业务模拟数据` -> `assistant 结合查询结果给客户合理解释与方案`。

### 【输出格式规范】
必须严格输出合法的 JSON 对象，不包含任何外部 markdown 标记或闲聊，顶层结构如下：
{
  "seed_id": "<对应的seed_id>",
  "scenario_type": "clarification / tool_use / standard",
  "tools": [...], // 若 tool_required 为 true 则给出工具定义，否则为空列表 []
  "messages": [
    {"role": "system", "content": "你是智能汽车官方客服助手..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "name": "...", "tool_call_id": "...", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""

    # 2. 纯对话场景 Few-Shot 示例（具备完整闭环）
    FEW_SHOT_CONVERSATION_EXAMPLE: Dict[str, Any] = {
        "seed_id": "cat1_feature_021",
        "scenario_type": "clarification",
        "tools": [],
        "messages": [
            {
                "role": "system",
                "content": "你是智能汽车官方客服助手。在处理客户问题时，必须先明确关键事实并进行必要追问，分步骤给出排查建议；严禁向客户索要密码或隐私信息，在关键信息缺失时不得臆造确定结论。"
            },
            {
                "role": "user",
                "content": "你好，我刚坐进车里，为什么手机蓝牙一直搜不到车机啊？"
            },
            {
                "role": "assistant",
                "content": "您好，请先别着急。为了帮您快速排查，想先跟您确认一下：您当前使用的手机型号是什么？另外车机屏幕上的蓝牙列表里，能看到周围其他设备吗？"
            },
            {
                "role": "user",
                "content": "我是 iPhone 15 Pro，车机屏幕能看到旁边人的手机，就是我手机列表里没有车机名字。"
            },
            {
                "role": "assistant",
                "content": "明白。因为车机蓝牙首次配对时，需要车机处于广播状态且在车机屏幕上点击确认。这种情况通常是蓝牙广播未同步或有旧缓存引起的，建议您按以下两步操作：\n1. 在手机【设置】-【蓝牙】中，将蓝牙关闭 5 秒后重新打开；\n2. 在车机中控屏上点击【添加新设备】，使车机进入广播配对状态，然后用手机重新搜索并点击车机名称。\n\n您可以先尝试一下，看看能否搜到？"
            },
            {
                "role": "user",
                "content": "按你说的车机点了添加设备，手机上立马搜到了，配对码确认后已经连上了！"
            },
            {
                "role": "assistant",
                "content": "太好了，很高兴帮您成功连接！后续如果手机开启蓝牙，上车通常会自动回连。请问还有其他用车方面的问题需要协助吗？"
            },
            {
                "role": "user",
                "content": "没有了，谢谢你！"
            },
            {
                "role": "assistant",
                "content": "不客气，祝您用车愉快，一路平安！"
            }
        ]
    }

    # 3. 工具调用场景 Few-Shot 示例（具备完整闭环）
    FEW_SHOT_TOOL_EXAMPLE: Dict[str, Any] = {
        "seed_id": "cat1_feature_022",
        "scenario_type": "tool_use",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "vehicle_binding_query",
                    "description": "查询车辆当前在官方 App 的绑定状态及车主账号归属",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "vin": {"type": "string", "description": "17位车辆识别代号（VIN码）"}
                        },
                        "required": ["vin"]
                    }
                }
            }
        ],
        "messages": [
            {
                "role": "system",
                "content": "你是智能汽车官方客服助手。涉及车辆绑定、远程控制或权限变更等业务时，必须先调用系统工具核验状态与所有权，严禁绕过实名认证或在未查询前虚构业务结论。"
            },
            {
                "role": "user",
                "content": "你好，我刚买了一辆二手车，想把车绑定到我的官方App，但总提示绑定失败。车架号是 LSVAA123456789012。"
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_bind_check_001",
                        "type": "function",
                        "function": {
                            "name": "vehicle_binding_query",
                            "arguments": json.dumps({"vin": "LSVAA123456789012"}, ensure_ascii=False)
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "name": "vehicle_binding_query",
                "tool_call_id": "call_bind_check_001",
                "content": json.dumps({
                    "status": "already_bound",
                    "is_original_owner": True,
                    "bound_mobile_mask": "138****1234",
                    "unbind_required": True
                }, ensure_ascii=False)
            },
            {
                "role": "assistant",
                "content": "您好，系统查询显示该车辆目前仍绑定在原车主账号下，因此您暂时无法直接添加。\n\n因为二手车重新绑定需要核验车辆所有权以保障资产安全，建议您这样处理：\n1. 准备好过户后的**新车辆行驶证**和**二手车交易发票（或过户证明）**；\n2. 在官方 App 打开【我的】-【车辆管理】-【二手车认证申诉】入口上传凭证。\n\n人工审核通常在 1 个工作日内完成解绑并为您绑定新账号。请问您目前手头是否已拿到过户后的新行驶证？"
            },
            {
                "role": "user",
                "content": "有的，绿本和新行驶证都在手边，那我直接在App里上传申诉就行是吧？"
            },
            {
                "role": "assistant",
                "content": "是的，您直接在【二手车认证申诉】入口拍照上传清晰的新行驶证和证件即可。提交后系统会短信通知审核进度。如果在审核过程中有任何疑问，欢迎随时联系我们！"
            },
            {
                "role": "user",
                "content": "好的，非常清楚，谢谢！"
            },
            {
                "role": "assistant",
                "content": "不客气，恭喜您喜提爱车，祝您生活愉快！"
            }
        ]
    }

    # 5 组正交多样性扰动指令（用于单 Seed 扩充 5 种异构对话）
    ORTHOGONAL_VARIATIONS = [
        {
            "id": "var_0_anxious_fast",
            "name": "情绪焦急/时间紧迫型",
            "instruction": "【客户画像与风格要求】客户情绪较为急躁、担心行程耽误，语气较为催促；客服必须在第一轮给出极具安抚性的回应，并单刀直入提供最高优先级的核心排查步骤，语言精炼干练。"
        },
        {
            "id": "var_1_novice_detailed",
            "name": "新手小白/通俗引导型",
            "instruction": "【客户画像与风格要求】客户对车机功能或专业术语不熟悉，首轮描述较为口语化甚至有点模糊；客服必须使用通俗生动的日常语言（避免生硬工业术语），按照‘第一步、第二步’提供保姆级、手把手的循序渐进操作指引。"
        },
        {
            "id": "var_2_misoperation_correction",
            "name": "先入为主/误操作纠偏型",
            "instruction": "【客户画像与风格要求】客户首轮提到自己尝试过某种错误的操作（如盲目拔插硬件、误改了无关系统设置等）但依然失败，带有干扰性前置条件；客服需先温和礼貌地纠正客户的认知误区，然后引导客户回到正规的排障流程中完成闭环。"
        },
        {
            "id": "var_3_environmental_context",
            "name": "真实出行/环境情境化型",
            "instruction": "【客户画像与风格要求】对话需融入生动的真实用车场景细节（如地库弱信号、暴雨天气、高速服务区赶路、冬季寒冷、赶着接送孩子等具体情境）；客服在排障之余体现出人性化的场景关怀（如雨天注意防滑、地库注意照明安全等），对话极具生活气息。"
        },
        {
            "id": "var_4_tech_concise",
            "name": "专业极客/高效利落型",
            "instruction": "【客户画像与风格要求】客户表达言简意赅、逻辑清晰，首轮直接说明故障现象和车辆当前状态；客服展现出极高的专业素养，迅速精准匹配系统机理或工具调用，排查流程干脆利落、高效闭环收尾。"
        }
    ]

    @classmethod
    def build_user_prompt(cls, seed: Dict[str, Any], variation_idx: int = 0) -> str:
        """
        根据传入的单条 Seed 字典与正交扰动索引 (0~4)，构造完整的 Generator User Prompt
        """
        is_tool = seed.get("tool_required", False)
        example = cls.FEW_SHOT_TOOL_EXAMPLE if is_tool else cls.FEW_SHOT_CONVERSATION_EXAMPLE

        var_config = cls.ORTHOGONAL_VARIATIONS[variation_idx % len(cls.ORTHOGONAL_VARIATIONS)]

        prompt = f"""请根据以下提供的【业务种子（Seed）】，合成一条符合标准微调格式的 SFT 数据。

### 【输入业务种子信息】
```json
{json.dumps(seed, ensure_ascii=False, indent=2)}
```

### 【本次正交风格扰动要求（Diversity Injection）】
- 变体模式: 【{var_config['name']}】
- 具体指引: {var_config['instruction']}

### 【参考范例（Few-Shot）】
```json
{json.dumps(example, ensure_ascii=False, indent=2)}
```

### 【生成特别要求】
1. **体现正交风格**：必须充分体现上述【{var_config['name']}】的对话特征与交互逻辑，杜绝千篇一律的机械复读；
2. **自然克制追问**：单轮追问严禁超过 2 个问题，口吻温和自然，严禁像填表一样罗列问卷；
3. **上下文消重**：严禁追问客户在上一轮已经提及的信息；
4. **闭环完整性**：对话必须包含【提问 -> 追问/调工具 -> 方案指导 -> 客户反馈执行结果 -> 客服确认收尾】的完整闭环；
5. **守住红线**：严禁触犯 `prohibited_actions`；
6. 严格直接输出标准合法的 JSON 格式。
"""
        return prompt

    @classmethod
    def get_generator_payload(cls, seed: Dict[str, Any], variation_idx: int = 0) -> List[Dict[str, str]]:
        """
        获取直接可用于调用大模型 API 的 messages 结构（支持正交扰动索引 0~4）
        """
        return [
            {"role": "system", "content": cls.GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": cls.build_user_prompt(seed, variation_idx=variation_idx)},
        ]
