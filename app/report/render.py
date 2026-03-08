from __future__ import annotations
from app.report.schema import ReportIR

def render_markdown(ir: ReportIR) -> str:
    lines = []
    lines.append("# 舆情分析报告（MVP）\n")

    lines.append("## 话题")
    lines.append(f"- {ir.topic}\n")

    lines.append("## 摘要")
    lines.append(ir.summary or "（占位）当前为骨架阶段，尚未接入检索与多 Agent。\n")

    lines.append("## 关键要点")
    if ir.key_points:
        lines += [f"- {p}" for p in ir.key_points]
    else:
        lines.append("- （占位）")
    lines.append("")

    lines.append("## 参考来源")
    if ir.sources:
        for i, s in enumerate(ir.sources, 1):
            lines.append(f"{i}. {s.get('title','source')} {s.get('url','')}".strip())
    else:
        lines.append("- （占位）")
    lines.append("")
    return "\n".join(lines)