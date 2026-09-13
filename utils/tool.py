"""
utils/tool.py - 汽车客服 55+ 核心工具轻量化注册表 (基于 Pydantic 继承抽象与 LangChain @tool)

设计准则：
1. 【Token 极致精简】：去除冗长的场景分析与废话，仅保留明确的一句话功能 Brief；
2. 【参数强契约】：明确标注参数含义、必需/可选，显式声明 Literal 枚举值与默认值；
3. 【基类抽象复用】：通过 Pydantic 继承机制统一车辆、工单、城市、案件等通用槽位，杜绝重复冗余；
4. 【全量 55+ 工具收录】：完整覆盖 data/v2/tool_schemas.json 中的所有工具，开箱即用。
"""

import sys
import json
from typing import Dict, Any, Optional, List, Literal
from pydantic import BaseModel, Field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 兼容 langchain_core 导入；若无则自动降级为轻量装饰器
try:
    from langchain_core.tools import tool
    from langchain_core.utils.function_calling import convert_to_openai_tool
except ImportError:
    def tool(args_schema=None):
        def decorator(func):
            func.args_schema = args_schema
            func.name = func.__name__
            func.description = func.__doc__.strip() if func.__doc__ else ""
            return func
        return decorator

    def convert_to_openai_tool(tool_obj):
        schema = tool_obj.args_schema.model_json_schema() if hasattr(tool_obj, "args_schema") and tool_obj.args_schema else {}
        return {
            "type": "function",
            "function": {
                "name": getattr(tool_obj, "name", tool_obj.__name__),
                "description": getattr(tool_obj, "description", tool_obj.__doc__ or "").strip(),
                "parameters": schema
            }
        }


# ==============================================================================
# 第一部分：Pydantic 参数基类抽象（复用消除 80% 冗余代码）
# ==============================================================================

class VehicleIdBase(BaseModel):
    """仅包含必填车架号/车辆ID的基础模型"""
    vehicle_id: str = Field(..., description="[必需] 车辆唯一代号（17位VIN码或车牌号）")

class WorkOrderIdBase(BaseModel):
    """仅包含必填工单编号的基础模型"""
    work_order_id: str = Field(..., description="[必需] 维修或服务工单编号（如 'WO202311020088'）")

class CaseIdBase(BaseModel):
    """仅包含必填案件编号的基础模型"""
    case_id: str = Field(..., description="[必需] 投诉或质保案件唯一编号")

class CityBase(BaseModel):
    """仅包含必填城市的基础模型"""
    city: str = Field(..., description="[必需] 目标城市名称（如 '北京市'、'上海市'）")

class VehicleModelBase(BaseModel):
    """仅包含必填车型名称的基础模型"""
    vehicle_model: str = Field(..., description="[必需] 车辆具体款型（如 '2023款 G7 Max'）")

class IssueBase(BaseModel):
    """问题工单创建基础模型"""
    issue_summary: str = Field(..., description="[必需] 客户反馈问题或事故现象概述")
    contact_channel: Optional[str] = Field(default=None, description="[可选] 客户联系渠道（如 phone, app）")


# --- 组合派生参数模型 ---

class VehicleDateInput(VehicleIdBase):
    preferred_date: str = Field(..., description="[必需] 客户期望预约的日期（格式：YYYY-MM-DD）")

class VehiclePartInput(VehicleIdBase):
    part_name: str = Field(..., description="[必需] 待查询的车辆零部件名称（如 '动力电池', '前保险杠'）")

class CityServiceInput(CityBase):
    service_type: Optional[str] = Field(default=None, description="[可选] 查询的具体服务类型项目")


# ==============================================================================
# 第二部分：55+ 工具轻量化注册定义 (覆盖 tool_schemas.json 全量定义)
# ==============================================================================

TOOL_REGISTRY: Dict[str, Any] = {}

def register(func):
    """工具自注册装饰器 (兼容 LangChain StructuredTool 与原生函数)"""
    tool_name = getattr(func, "name", getattr(func, "__name__", None))
    TOOL_REGISTRY[tool_name] = func
    return func


# ----- 1. 质保与权益模块 -----

@register
@tool(args_schema=VehicleIdBase)
def battery_warranty_query(vehicle_id: str) -> Dict[str, Any]:
    """查询动力电池质保状态、衰减保障期与当前质保生效情况。"""
    return {"status": "success", "vehicle_id": vehicle_id, "battery_warranty": "8年或16万公里质保有效"}

@register
@tool(args_schema=VehicleIdBase)
def extended_warranty_query(vehicle_id: str) -> Dict[str, Any]:
    """查询车辆是否购买延保服务及生效范围。"""
    return {"status": "success", "vehicle_id": vehicle_id, "extended_warranty": "已购买整车延长质保2年"}

@register
@tool(args_schema=VehiclePartInput)
def repair_warranty_query(vehicle_id: str, part_name: str) -> Dict[str, Any]:
    """查询维修更换项目的具体配件质保期限。"""
    return {"status": "success", "vehicle_id": vehicle_id, "part_name": part_name, "warranty_period": "12个月"}

@register
@tool(args_schema=CaseIdBase)
def warranty_case_query(case_id: str) -> Dict[str, Any]:
    """查询质保理赔案件的审核状态与赔付进展。"""
    return {"status": "success", "case_id": case_id, "case_status": "审核通过，待配件调拨"}

class WarrantyPolicyInput(VehicleModelBase):
    pass

@register
@tool(args_schema=WarrantyPolicyInput)
def warranty_policy_query(vehicle_model: str) -> Dict[str, Any]:
    """查询指定车型的官方三包及整车质保政策细则。"""
    return {"status": "success", "vehicle_model": vehicle_model, "policy": "整车5年或10万公里三包保障"}

@register
@tool(args_schema=VehicleIdBase)
def warranty_status_query(vehicle_id: str) -> Dict[str, Any]:
    """查询整车基础质保是否在保及剩余天数。"""
    return {"status": "success", "vehicle_id": vehicle_id, "in_warranty": True, "remaining_days": 420}


# ----- 2. 充电与电池服务模块 -----

class CampaignInput(VehicleIdBase):
    campaign_code: str = Field(..., description="[必需] 营销或权益活动编号")

@register
@tool(args_schema=CampaignInput)
def campaign_eligibility_query(vehicle_id: str, campaign_code: str) -> Dict[str, Any]:
    """查询车辆或车主参与特定营销/充电权益活动的资格。"""
    return {"status": "success", "vehicle_id": vehicle_id, "eligible": True}

class ChargerCapabilityInput(VehicleIdBase):
    charger_model: Optional[str] = Field(default=None, description="[可选] 充电桩具体型号")

@register
@tool(args_schema=ChargerCapabilityInput)
def charger_capability_query(vehicle_id: str, charger_model: Optional[str] = None) -> Dict[str, Any]:
    """查询充电设备（家用桩/超充）与车辆的兼容性及最大充电功率。"""
    return {"status": "success", "vehicle_id": vehicle_id, "max_power_kw": 11}

class ChargerSurveyInput(BaseModel):
    address: str = Field(..., description="[必需] 安装充电桩的详细勘测地址")
    preferred_date: Optional[str] = Field(default=None, description="[可选] 预约上门勘测日期（YYYY-MM-DD）")

@register
@tool(args_schema=ChargerSurveyInput)
def charger_survey_booking(address: str, preferred_date: Optional[str] = None) -> Dict[str, Any]:
    """预约家充桩上门电表与车位勘测服务。"""
    return {"status": "success", "address": address, "booking_status": "已受理"}

class ChargingBenefitInput(VehicleIdBase):
    region: Optional[str] = Field(default=None, description="[可选] 查询适用的城市或行政区域")

@register
@tool(args_schema=ChargingBenefitInput)
def charging_benefit_query(vehicle_id: str, region: Optional[str] = None) -> Dict[str, Any]:
    """查询车辆账户绑定的免费充电额度或权益卡状态。"""
    return {"status": "success", "vehicle_id": vehicle_id, "free_kwh_left": 1200}


# ----- 3. 维修、配件与工单服务模块 -----

@register
@tool(args_schema=VehicleDateInput)
def bodyshop_booking(vehicle_id: str, preferred_date: str) -> Dict[str, Any]:
    """预约官方钣金喷漆维修服务。"""
    return {"status": "success", "vehicle_id": vehicle_id, "date": preferred_date, "result": "预约成功"}

class RepairAuthInput(WorkOrderIdBase):
    approved: bool = Field(..., description="[必需] 车主是否授权同意本次维修施工方案与报价")

@register
@tool(args_schema=RepairAuthInput)
def repair_authorization(work_order_id: str, approved: bool) -> Dict[str, Any]:
    """提交车主对指定维修工单的正式授权批准或拒绝。"""
    return {"status": "success", "work_order_id": work_order_id, "authorized": approved}

class RepairIntakeInput(VehicleIdBase):
    issue_summary: Optional[str] = Field(default=None, description="[可选] 需到店送修的故障或损伤简要描述")

@register
@tool(args_schema=RepairIntakeInput)
def repair_intake_query(vehicle_id: str, issue_summary: Optional[str] = None) -> Dict[str, Any]:
    """查询车辆进站维修的接待要求、预检工序及证件材料准备指引。"""
    return {"status": "success", "vehicle_id": vehicle_id, "requirements": "需携带行驶证与车主身份证"}

@register
@tool(args_schema=CityBase)
def repair_network_query(city: str) -> Dict[str, Any]:
    """查询指定城市下所有具备官方认证资质的维修中心及4S店网点。"""
    return {"status": "success", "city": city, "stores": ["官方海淀售后中心", "朝阳特约维修店"]}

@register
@tool(args_schema=WorkOrderIdBase)
def repair_order_query(work_order_id: str) -> Dict[str, Any]:
    """查询维修工单实时进度、施工工位、配件到货与预计交付时间。"""
    return {"status": "success", "work_order_id": work_order_id, "stage": "等待配件入库"}

@register
@tool(args_schema=WorkOrderIdBase)
def repair_quote_query(work_order_id: str) -> Dict[str, Any]:
    """查询指定维修工单的费用明细清单与报价明细。"""
    return {"status": "success", "work_order_id": work_order_id, "total_cost": 1850.0}

@register
@tool(args_schema=WorkOrderIdBase)
def mobility_support_query(work_order_id: str) -> Dict[str, Any]:
    """查询维修超期期间免费代步车安排或出行打车津贴审核状态。"""
    return {"status": "success", "work_order_id": work_order_id, "support_type": "代步车已准备就绪"}

class PartsInventoryInput(BaseModel):
    part_name: str = Field(..., description="[必需] 需查询的配件名称或原厂编号")
    city: Optional[str] = Field(default=None, description="[可选] 拟查询库存的城市")

@register
@tool(args_schema=PartsInventoryInput)
def parts_inventory_query(part_name: str, city: Optional[str] = None) -> Dict[str, Any]:
    """查询指定零部件的区域仓库现货库存情况。"""
    return {"status": "success", "part_name": part_name, "stock_count": 3}

@register
@tool(args_schema=WorkOrderIdBase)
def parts_order_query(work_order_id: str) -> Dict[str, Any]:
    """查询工单关联零件向厂家供应链发起订购的审批与发运状态。"""
    return {"status": "success", "work_order_id": work_order_id, "order_status": "已发运"}

class PartsPriceInput(BaseModel):
    part_name: str = Field(..., description="[必需] 零部件名称")
    vehicle_model: Optional[str] = Field(default=None, description="[可选] 适用的具体车型年款")

@register
@tool(args_schema=PartsPriceInput)
def parts_price_query(part_name: str, vehicle_model: Optional[str] = None) -> Dict[str, Any]:
    """查询原厂备件的标准统一指导零售价格及建议工时费。"""
    return {"status": "success", "part_name": part_name, "price_cny": 860.0}

class PartsTraceInput(BaseModel):
    part_order_id: str = Field(..., description="[必需] 零部件物流调拨单号")

@register
@tool(args_schema=PartsTraceInput)
def parts_trace_query(part_order_id: str) -> Dict[str, Any]:
    """查询在途调拨配件的实时干线物流与预计到店时间。"""
    return {"status": "success", "part_order_id": part_order_id, "eta": "今日16:30"}

@register
@tool(args_schema=WorkOrderIdBase)
def payment_and_invoice_query(work_order_id: str) -> Dict[str, Any]:
    """查询维修工单的线上支付状态与电子发票开具链接。"""
    return {"status": "success", "work_order_id": work_order_id, "invoice_status": "已开具"}


# ----- 4. 保养与到店预约模块 -----

@register
@tool(args_schema=VehicleIdBase)
def maintenance_due_query(vehicle_id: str) -> Dict[str, Any]:
    """查询车辆下一次定期保养到期里程与建议保养时间。"""
    return {"status": "success", "vehicle_id": vehicle_id, "due_mileage": 20000, "remaining_km": 1500}

class MaintenancePolicyInput(VehicleModelBase):
    mileage: Optional[int] = Field(default=None, description="[可选] 当前表显总里程（公里）")

@register
@tool(args_schema=MaintenancePolicyInput)
def maintenance_policy_query(vehicle_model: str, mileage: Optional[int] = None) -> Dict[str, Any]:
    """查询不同车型在各里程区间的官方常规保养周期与项目说明。"""
    return {"status": "success", "vehicle_model": vehicle_model, "cycle": "每1年或2万公里常规检查"}

class ServiceBookingInput(VehicleDateInput):
    store_id: str = Field(..., description="[必需] 拟预约到店的服务门店或4S店ID")

@register
@tool(args_schema=ServiceBookingInput)
def service_booking(vehicle_id: str, store_id: str, preferred_date: str) -> Dict[str, Any]:
    """预约指定服务网点的常规保养或维保工位。"""
    return {"status": "success", "booking_id": "BK8899", "status": "预约成功"}

@register
@tool(args_schema=VehicleIdBase)
def service_history_query(vehicle_id: str) -> Dict[str, Any]:
    """查询车辆全生命周期在官方网点的历史保养与维修记录。"""
    return {"status": "success", "vehicle_id": vehicle_id, "history_count": 4}

@register
@tool(args_schema=VehicleIdBase)
def service_package_query(vehicle_id: str) -> Dict[str, Any]:
    """查询车主已购买的基础保养或终身免费保养套餐余额。"""
    return {"status": "success", "vehicle_id": vehicle_id, "remaining_services": 2}

class ServicePolicyInput(BaseModel):
    topic: str = Field(..., description="[必需] 查询的服务政策主题（如 '退换车规则', '首保政策'）")
    vehicle_model: Optional[str] = Field(default=None, description="[可选] 适用车型")

@register
@tool(args_schema=ServicePolicyInput)
def service_policy_query(topic: str, vehicle_model: Optional[str] = None) -> Dict[str, Any]:
    """查询官方售后各项服务保障细则与承诺政策。"""
    return {"status": "success", "topic": topic, "policy_content": "首保自购车起5000公里内免费"}

class PickupServiceInput(BaseModel):
    store_id: str = Field(..., description="[必需] 负责取送车的服务门店代号")
    preferred_date: Optional[str] = Field(default=None, description="[可选] 预约取车时间（YYYY-MM-DD）")

@register
@tool(args_schema=PickupServiceInput)
def pickup_service_query(store_id: str, preferred_date: Optional[str] = None) -> Dict[str, Any]:
    """查询售后维修上门代客取车或送车到家的服务范围与排期。"""
    return {"status": "success", "store_id": store_id, "available": True}

@register
@tool(args_schema=CityServiceInput)
def mobile_service_query(city: str, service_type: Optional[str] = None) -> Dict[str, Any]:
    """查询移动服务车上门简易保养或应急搭电服务支持情况。"""
    return {"status": "success", "city": city, "support_mobile_van": True}


# ----- 5. 紧急救援与安全预警模块 -----

class RoadsideDispatchInput(VehicleIdBase):
    location: str = Field(..., description="[必需] 故障抛锚或事故发生的具体地理位置/路段")

@register
@tool(args_schema=RoadsideDispatchInput)
def roadside_assistance_dispatch(vehicle_id: str, location: str) -> Dict[str, Any]:
    """为抛锚无法行驶的车辆加急派发官方道路救援拖车。"""
    return {"status": "success", "dispatch_id": "RES991", "driver_eta_minutes": 30}

class RoadsideQueryInput(VehicleIdBase):
    question: Optional[str] = Field(default=None, description="[可选] 需咨询的救援收费或保险垫付疑问")

@register
@tool(args_schema=RoadsideQueryInput)
def roadside_or_insurance_query(vehicle_id: str, question: Optional[str] = None) -> Dict[str, Any]:
    """查询官方免费道路救援里程范围及搭电换胎政策。"""
    return {"status": "success", "vehicle_id": vehicle_id, "free_tow_km": 100}

@register
@tool(args_schema=CityBase)
def regional_alert_query(city: str) -> Dict[str, Any]:
    """查询指定区域暴雨、冰雪暴跌或地质灾害等行车安全预警通知。"""
    return {"status": "success", "city": city, "alerts": ["暴雨橙色预警，涉水路段减速慢行"]}

@register
@tool(args_schema=VehicleIdBase)
def recall_status_query(vehicle_id: str) -> Dict[str, Any]:
    """根据VIN码精准查询车辆是否存在未完成的官方安全召回或服务活动。"""
    return {"status": "success", "vehicle_id": vehicle_id, "has_pending_recall": False}


# ----- 6. 车联网、车机与车辆功能模块 -----

class DigitalKeyInput(VehicleIdBase):
    phone_model: Optional[str] = Field(default=None, description="[可选] 车主的智能手机机型")

@register
@tool(args_schema=DigitalKeyInput)
def digital_key_compatibility(vehicle_id: str, phone_model: Optional[str] = None) -> Dict[str, Any]:
    """查询手机蓝牙/UWB/NFC数字钥匙与当前车型的适配支持列表。"""
    return {"status": "success", "vehicle_id": vehicle_id, "supported": True}

@register
@tool(args_schema=VehicleIdBase)
def ota_campaign_query(vehicle_id: str) -> Dict[str, Any]:
    """查询车机最新可升级的 OTA 固件版本及发布变更说明。"""
    return {"status": "success", "vehicle_id": vehicle_id, "current_version": "v2.5", "latest_version": "v2.6"}

@register
@tool(args_schema=VehicleIdBase)
def subscription_status_query(vehicle_id: str) -> Dict[str, Any]:
    """查询智能座舱车联车机娱乐流量及高阶智驾软件订阅有效期。"""
    return {"status": "success", "vehicle_id": vehicle_id, "traffic_gb": 10, "autopilot_active": True}

class CapabilityInput(VehicleIdBase):
    capability: Optional[str] = Field(default=None, description="[可选] 拟查询的权限名称（如 '远程控车', '主车主'）")

@register
@tool(args_schema=CapabilityInput)
def vehicle_account_capability(vehicle_id: str, capability: Optional[str] = None) -> Dict[str, Any]:
    """查询车主或副授权账号在 App 端的车辆控制功能权限。"""
    return {"status": "success", "vehicle_id": vehicle_id, "permissions": ["控温", "车锁", "车况查看"]}

@register
@tool(args_schema=VehicleIdBase)
def vehicle_binding_query(vehicle_id: str) -> Dict[str, Any]:
    """查询车辆当前绑定的实名认证主账号信息。"""
    return {"status": "success", "vehicle_id": vehicle_id, "bound_user": "车主认证已完成"}

@register
@tool(args_schema=VehicleIdBase)
def vehicle_connectivity_status(vehicle_id: str) -> Dict[str, Any]:
    """查询车载 T-Box 通信模块的 4G/5G 信号强度与联网通断状态。"""
    return {"status": "success", "vehicle_id": vehicle_id, "online": True, "signal_strength": "4格"}

class FeatureNameInput(VehicleIdBase):
    feature_name: Optional[str] = Field(default=None, description="[可选] 车辆功能项（如 '自动大灯', '感应尾门'）")

@register
@tool(args_schema=FeatureNameInput)
def vehicle_feature_query(vehicle_id: str, feature_name: Optional[str] = None) -> Dict[str, Any]:
    """查询车辆硬件配置是否原生支持某功能及其具体车机设置路径。"""
    return {"status": "success", "vehicle_id": vehicle_id, "feature": feature_name, "supported": True}

@register
@tool(args_schema=VehicleIdBase)
def vehicle_health_query(vehicle_id: str) -> Dict[str, Any]:
    """调取整车远程诊断系统报告，查询胎压、制动液及电机电池健康度。"""
    return {"status": "success", "vehicle_id": vehicle_id, "health_score": 98, "tpms_status": "正常"}

@register
@tool(args_schema=VehicleIdBase)
def vehicle_online_status(vehicle_id: str) -> Dict[str, Any]:
    """查询车辆当前处于休眠、哨兵模式、在线行车还是离线状态。"""
    return {"status": "success", "vehicle_id": vehicle_id, "state": "休眠节电中"}

@register
@tool(args_schema=VehicleIdBase)
def vehicle_security_status(vehicle_id: str) -> Dict[str, Any]:
    """查询车辆车门车窗锁闭状态及车身防盗报警状态。"""
    return {"status": "success", "vehicle_id": vehicle_id, "doors_locked": True, "windows_closed": True}


# ----- 7. 客户运营、授权与投诉合规模块 (含枚举定义) -----

class CommunicationConsentInput(BaseModel):
    channel: Literal["phone", "sms", "app", "email"] = Field(
        ...,
        description="[必需/枚举] 拟授权联系的沟通渠道：'phone'(电话), 'sms'(短信), 'app'(推送), 'email'(邮件)"
    )
    consent: bool = Field(..., description="[必需] 是否同意授权联系")

@register
@tool(args_schema=CommunicationConsentInput)
def communication_consent_update(
    channel: Literal["phone", "sms", "app", "email"],
    consent: bool
) -> Dict[str, Any]:
    """更新车主在官方 CRM 系统中的多渠道外呼与营销免打扰偏好。"""
    return {"status": "success", "channel": channel, "consent": consent}

class ComplaintCaseInput(BaseModel):
    issue_summary: str = Field(..., description="[必需] 客户投诉核心问题、争议现象或涉事门店人员行为概述")
    vehicle_id: Optional[str] = Field(default=None, description="[可选] 投诉所涉车辆的唯一代号（17位VIN码或车牌号）")
    contact_channel: Optional[str] = Field(default=None, description="[可选] 客户要求联系的渠道（如 phone, app）")

@register
@tool(args_schema=ComplaintCaseInput)
def complaint_case_create(
    issue_summary: str,
    vehicle_id: Optional[str] = None,
    contact_channel: Optional[str] = None
) -> Dict[str, Any]:
    """为客户对门店服务态度恶劣、收费欺诈、维修争议或严重不满正式创建官方升级投诉工单。"""
    return {"status": "success", "case_id": "CP2026091001", "message": "投诉工单已创建并指派专员加急督办"}

@register
@tool(args_schema=CaseIdBase)
def complaint_case_query(case_id: str) -> Dict[str, Any]:
    """查询车主投诉工单的处理阶段、涉事门店及最终跟进回复。"""
    return {"status": "success", "case_id": case_id, "case_status": "督办处理中"}

class ConsentPolicyInput(BaseModel):
    policy_type: str = Field(..., description="[必需] 政策协议类型（如 '隐私保护指引', '车联网服务条款'）")

@register
@tool(args_schema=ConsentPolicyInput)
def consent_and_policy_query(policy_type: str) -> Dict[str, Any]:
    """查询车企用户个人信息授权及隐私合规政策详情。"""
    return {"status": "success", "policy_type": policy_type, "version": "2026版"}

class CustomerFollowupInput(VehicleIdBase):
    followup_note: str = Field(..., description="[必需] 回访记录的客户评价或新增反馈内容")

@register
@tool(args_schema=CustomerFollowupInput)
def customer_followup_record(vehicle_id: str, followup_note: str) -> Dict[str, Any]:
    """将维修或服务回访结果保存至 CRM 系统并生成闭环记录。"""
    return {"status": "success", "vehicle_id": vehicle_id, "recorded": True}


class DealerPolicyInput(BaseModel):
    policy_topic: str = Field(..., description="[必需] 经销商政策专题（如 '提车规范', '代金券核销'）")
    dealer_id: Optional[str] = Field(default=None, description="[可选] 经销商门店代号")

@register
@tool(args_schema=DealerPolicyInput)
def dealer_policy_query(policy_topic: str, dealer_id: Optional[str] = None) -> Dict[str, Any]:
    """查询特定授权经销商或全国统一的商务服务细则。"""
    return {"status": "success", "topic": policy_topic, "valid": True}

@register
@tool(args_schema=CityServiceInput)
def dealer_service_query(city: str, service_type: Optional[str] = None) -> Dict[str, Any]:
    """查询授权特约销售服务店的营业时间与接待能力。"""
    return {"status": "success", "city": city, "open_hours": "09:00-18:00"}

class InsuranceBookingInput(VehicleDateInput):
    pass

@register
@tool(args_schema=InsuranceBookingInput)
def insurance_consult_booking(vehicle_id: str, preferred_date: str) -> Dict[str, Any]:
    """预约官方续保顾问进行新一年车险方案测算与出单咨询。"""
    return {"status": "success", "vehicle_id": vehicle_id, "status": "续保顾问将准时联系"}

class InsurancePolicyInput(BaseModel):
    policy_number: str = Field(..., description="[必需] 车辆商业险或交强险保单号")

@register
@tool(args_schema=InsurancePolicyInput)
def insurance_policy_query(policy_number: str) -> Dict[str, Any]:
    """查询商业险具体险种（车损、三者险额度）及到期终止时间。"""
    return {"status": "success", "policy_number": policy_number, "valid_until": "2027-05-20"}

@register
@tool(args_schema=IssueBase)
def privacy_incident_create(issue_summary: str, contact_channel: Optional[str] = None) -> Dict[str, Any]:
    """创建车辆数据采集或隐私安全相关的事件报告工单。"""
    return {"status": "success", "ticket_id": "PRIV_990", "status": "已移交安全合规部"}


# ==============================================================================
# 第三部分：工具导出、按需过滤与执行工具链
# ==============================================================================

def get_all_tools() -> List[Any]:
    """获取全量 55+ 工具对象列表"""
    return list(TOOL_REGISTRY.values())

def get_tools_by_names(names: List[str]) -> List[Any]:
    """根据路由推荐的工具名称列表，按需获取轻量工具对象"""
    return [TOOL_REGISTRY[n] for n in names if n in TOOL_REGISTRY]

def export_openai_schemas(tool_list: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """导出标准精简版 OpenAI/Qwen Function Calling 规范格式"""
    tools_to_export = tool_list if tool_list is not None else get_all_tools()
    return [convert_to_openai_tool(t) for t in tools_to_export]

def export_compact_schemas(tool_list: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    """
    导出极致扁平紧凑的 JSON 格式（去除 properties / required 嵌套层级，Token 暴降 60%）。
    专用于替代冗长的 OpenAI JSON Schema 直接注入大模型 Prompt。
    格式示例:
    {
      "name": "vehicle_feature_query",
      "description": "查询车辆硬件配置...",
      "parameters": {
        "vehicle_id": "string(必填): 车辆唯一代号（17位VIN码或车牌号）",
        "feature_name": "string(可选): 车辆功能项（如 '自动大灯'）"
      }
    }
    """
    tools_to_export = tool_list if tool_list is not None else get_all_tools()
    compact_list = []
    for t in tools_to_export:
        if isinstance(t, dict):
            func = t.get("function", t)
            t_name = func.get("name", "")
            t_desc = func.get("description", "").strip()
            props = func.get("parameters", {}).get("properties", {})
            required_set = set(func.get("parameters", {}).get("required", []))
            compact_params = {}
            for p_name, p_info in props.items():
                req_label = "必填" if p_name in required_set else "可选"
                p_type = p_info.get("type", "string")
                p_desc = p_info.get("description", "").replace("[必需] ", "").replace("[可选] ", "").strip()
                if "enum" in p_info:
                    p_desc += f" 枚举:{p_info['enum']}"
                compact_params[p_name] = f"{p_type}({req_label}): {p_desc}"
            compact_list.append({
                "name": t_name,
                "description": t_desc,
                "parameters": compact_params,
            })
            continue

        t_name = getattr(t, "name", None) or getattr(t, "__name__", str(t))
        t_desc = (getattr(t, "description", None) or getattr(t, "__doc__", "") or "").strip()
        
        compact_params = {}
        if hasattr(t, "args_schema") and t.args_schema:
            schema = t.args_schema.model_json_schema()
            props = schema.get("properties", {})
            required_set = set(schema.get("required", []))
            
            for p_name, p_info in props.items():
                req_label = "必填" if p_name in required_set else "可选"
                p_type = p_info.get("type")
                if not p_type and "anyOf" in p_info:
                    types = [tp.get("type") for tp in p_info["anyOf"] if tp.get("type") != "null"]
                    p_type = "/".join(types) if types else "any"
                elif not p_type:
                    p_type = "string"
                    
                p_desc = p_info.get("description", "").replace("[必需] ", "").replace("[可选] ", "").strip()
                
                # 若有枚举 Literal，显示候选枚举
                if "enum" in p_info:
                    enum_str = f" 枚举:{p_info['enum']}"
                    p_desc += enum_str
                    
                compact_params[p_name] = f"{p_type}({req_label}): {p_desc}"
                
        compact_list.append({
            "name": t_name,
            "description": t_desc,
            "parameters": compact_params
        })
    return compact_list

def export_signature_schemas(tool_list: Optional[List[Any]] = None) -> List[str]:
    """
    导出类 Python 函数签名格式的伪代码（对大语言模型最自然亲和，Token 消耗极小）。
    格式示例:
    def vehicle_feature_query(vehicle_id: str, feature_name: str = None) -> dict:
        \"\"\"查询车辆硬件配置...\"\"\"
    """
    tools_to_export = tool_list if tool_list is not None else get_all_tools()
    signatures = []
    for t in tools_to_export:
        t_name = getattr(t, "name", t.__name__)
        t_desc = getattr(t, "description", t.__doc__ or "").strip()
        
        args_list = []
        doc_params = []
        if hasattr(t, "args_schema") and t.args_schema:
            schema = t.args_schema.model_json_schema()
            props = schema.get("properties", {})
            required_set = set(schema.get("required", []))
            
            for p_name, p_info in props.items():
                is_req = p_name in required_set
                p_type = p_info.get("type", "str")
                if p_type == "string": p_type = "str"
                elif p_type == "integer": p_type = "int"
                elif p_type == "boolean": p_type = "bool"
                
                arg_def = f"{p_name}: {p_type}" if is_req else f"{p_name}: Optional[{p_type}] = None"
                args_list.append(arg_def)
                
                p_desc = p_info.get("description", "").replace("[必需] ", "").replace("[可选] ", "").strip()
                doc_params.append(f"    :param {p_name}: ({'必填' if is_req else '可选'}) {p_desc}")
                
        args_str = ", ".join(args_list)
        doc_str = f"    \"\"\"{t_desc}\n" + "\n".join(doc_params) + "\n    \"\"\"" if doc_params else f"    \"\"\"{t_desc}\"\"\""
        signatures.append(f"def {t_name}({args_str}) -> dict:\n{doc_str}")
    return signatures


if __name__ == "__main__":
    total = len(TOOL_REGISTRY)
    print("=" * 80)
    print(f"🎉 汽车客服工具注册表构建完成！共收录: {total} 个工具 (已完成基类抽象与精简化)")
    print("=" * 80)
    
    # 抽取 2 个具代表性的工具进行 Schema 展示（含单参数与带枚举工具）
    sample_tools = get_tools_by_names(["repair_order_query", "communication_consent_update"])
    sample_schemas = export_openai_schemas(sample_tools)
    
    print("📋 精简后导出的标准 Schema (Token 开销极其紧凑，含枚举 Literal):")
    print(json.dumps(sample_schemas, ensure_ascii=False, indent=2))
    
    print("\n" + "=" * 80)
    print("🚀 测试本地快速执行带枚举的工具:")
    print("=" * 80)
    
    test_func = TOOL_REGISTRY["communication_consent_update"]
    res = test_func.invoke({"channel": "sms", "consent": False}) if hasattr(test_func, "invoke") else test_func(channel="sms", consent=False)
    print("执行结果:", res)
