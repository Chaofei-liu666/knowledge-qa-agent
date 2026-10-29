"""生成十万级规模的合成 chunk，用于验证向量索引与延迟。

产物不进仓库（.gitignore 掉 data/scale/），随时可以重新生成：
    python scripts/gen_scale_corpus.py --docs 10000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TOPICS = [
    ("差旅报销", ["交通费", "住宿费", "伙食补助", "市内交通费"], "报销"),
    ("采购管理", ["供应商准入", "比价", "合同用印", "验收"], "采购"),
    ("考勤休假", ["年假", "病假", "调休", "加班"], "考勤"),
    ("信息安全", ["数据分级", "账号口令", "外发审批", "终端管理"], "安全"),
    ("合同管理", ["用印", "归档", "履约", "变更"], "合同"),
    ("资产管理", ["领用", "盘点", "报废", "调拨"], "资产"),
    ("招聘录用", ["面试", "背景调查", "offer", "入职"], "招聘"),
    ("培训发展", ["课程", "学分", "外部培训", "认证"], "培训"),
    ("绩效管理", ["目标", "评估", "申诉", "校准"], "绩效"),
    ("费用管控", ["预算", "报销", "付款", "发票"], "费用"),
    # 以下是与验收题无关的干扰领域：真实知识库里大部分内容也是这种"别的部门的事"
    ("食堂管理", ["供餐时间", "餐卡充值", "临时加餐", "食品安全"], "食堂"),
    ("班车管理", ["线路", "发车时刻", "座位预约", "临时用车"], "班车"),
    ("停车管理", ["车位申请", "月卡", "访客车辆", "违停处理"], "停车"),
    ("工装发放", ["尺码统计", "换季发放", "工装回收", "特殊岗位"], "工装"),
    ("会议室", ["预约", "设备调试", "超时占用", "清洁"], "会议室"),
    ("快递收发", ["到件登记", "寄件", "保价", "异常件"], "快递"),
    ("健身房", ["开放时间", "器械维护", "课程报名", "安全须知"], "健身"),
    ("母婴室", ["位置", "清洁", "预约", "物品配置"], "母婴"),
    ("访客接待", ["登记", "陪同", "临时门禁", "接待标准"], "访客"),
    ("消防安全", ["演练", "器材检查", "疏散通道", "动火作业"], "消防"),
    ("绿化保洁", ["绿植养护", "保洁频次", "垃圾清运", "虫害防治"], "保洁"),
    ("车辆管理", ["派车", "油卡", "保养", "违章"], "车辆"),
    ("印章管理", ["刻制", "保管", "借用", "销毁"], "印章"),
    ("档案管理", ["归档范围", "借阅", "保管期限", "销毁"], "档案"),
    ("知识产权", ["专利申请", "商标注册", "软著登记", "侵权处理"], "知产"),
    ("审计配合", ["资料提供", "访谈", "整改", "底稿"], "审计"),
    ("税务申报", ["申报期", "发票认证", "税收优惠", "汇算清缴"], "税务"),
    ("结算管理", ["对账", "开票", "账期", "争议处理"], "结算"),
    ("汇率管理", ["结汇", "购汇", "汇率锁定", "外币账户"], "汇率"),
    ("商业保险", ["投保", "理赔", "续保", "受益人变更"], "保险"),
    ("员工体检", ["预约", "项目", "报告解读", "复查"], "体检"),
    ("离职管理", ["交接", "离职证明", "社保转移", "竞业限制"], "离职"),
]
SCOPES = ["集团", "华东区", "华北区", "华南区", "产品"]
DEPTS = ["财务", "人力", "行政", "法务", "采购", "信息安全", "研发", "销售"]

TEMPLATES = [
    "第{n}条　{d}负责{t}事项的初审与复核，单笔金额超过{amt}元的，须报{t}负责人批准。",
    "第{n}条　{t}相关材料应在事项发生后{days}个工作日内提交至{d}，逾期需书面说明。",
    "第{n}条　涉及{t}的操作须在系统中留痕，{d}每季度抽查一次，抽查比例不低于{pct}%。",
    "第{n}条　{d}应建立{t}台账，记录事项编号、经办人、审批人与完成时间。",
    "第{n}条　{t}的标准由{d}制定，报集团备案后执行；调整时提前{days}个工作日通知。",
    "第{n}条　违反{t}规定的，由{d}责令整改；情节较重的，按集团奖惩办法处理。",
    "第{n}条　{t}过程中形成的纸质材料，由{d}保存{days}个月，到期统一销毁。",
    "第{n}条　跨部门{t}事项，由主办部门{d}牵头，协办部门在{days}个工作日内反馈。",
]


def make_doc(i: int, rng: random.Random) -> list[dict]:
    topic, items, short = TOPICS[i % len(TOPICS)]
    dept = rng.choice(DEPTS)
    scope = rng.choice(SCOPES)
    doc_id = f"GEN-{i:05d}"
    title = f"{scope}{topic}实施细则（{doc_id}）"
    version = f"v{rng.randint(1, 6)}.{rng.randint(0, 9)}"
    year = rng.choice([2023, 2024, 2025])
    chunks = []
    for s in range(1, 11):
        item = items[s % len(items)]
        text = rng.choice(TEMPLATES).format(
            n=s, d=dept, t=item, amt=rng.choice([5000, 10000, 50000, 200000]),
            days=rng.choice([3, 5, 10, 15, 30]), pct=rng.choice([5, 10, 20]),
        )
        chunks.append({
            "chunk_id": f"{doc_id}#{s}",
            "doc_id": doc_id,
            "title": title,
            "version": version,
            "effective_date": f"{year}-0{rng.randint(1, 9)}-01",
            "scope": scope,
            "precedence": rng.choice([1, 2, 3]),
            "status": "active",
            "superseded_by": None,
            "topics": [topic, short],
            "slots": {},
            "section_id": str(s),
            "heading": f"{s} {item}",
            "text": text,
        })
    return chunks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=10000)
    parser.add_argument("--out", default=str(ROOT / "data" / "scale" / "chunks.jsonl"))
    args = parser.parse_args(argv)

    rng = random.Random(20251029)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w", encoding="utf-8") as fh:
        for i in range(args.docs):
            for chunk in make_doc(i, rng):
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total += 1
    print(f"生成 {args.docs} 篇 / {total} 个 chunk → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
