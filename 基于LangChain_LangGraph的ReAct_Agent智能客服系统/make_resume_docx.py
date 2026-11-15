# -*- coding: utf-8 -*-
r"""生成《简历项目经历 + 面试问答手册》Word 文档。

素材来源：本项目全部源码（agent/ rag/ model/ utils/ config/ prompts/）与 logs/ 下 4 天运行日志。
"""

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

FONT = "微软雅黑"
NAVY = RGBColor(0x1F, 0x38, 0x64)
BLUE = RGBColor(0x2E, 0x53, 0x95)
GRAY = RGBColor(0x5F, 0x5F, 0x5F)
INK = RGBColor(0x26, 0x26, 0x26)
RED = RGBColor(0xA6, 0x1B, 0x1B)

OUT = "简历项目经历_ReAct_Agent智能客服.docx"


# ---------------------------------------------------------------- 基础工具
def style_run(run, size=10.5, bold=False, color=INK, font=FONT, italic=False):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    for attr in ("w:eastAsia", "w:ascii", "w:hAnsi", "w:cs"):
        rFonts.set(qn(attr), font)


def para(doc, text="", size=10.5, bold=False, color=INK, align=None,
         before=0, after=6, line=1.45):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    if text:
        style_run(p.add_run(text), size=size, bold=bold, color=color)
    return p


def rich(doc, chunks, size=10.5, after=3, line=1.45, indent=None, before=0):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    if indent is not None:
        pf.left_indent = Cm(indent)
    for item in chunks:
        t, b, c = item
        style_run(p.add_run(t), size=size, bold=b, color=c)
    return p


def _bottom_border(p, color="1F3864", sz="8"):
    pPr = p._element.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), sz)
    bottom.set(qn("w:space"), "4")
    bottom.set(qn("w:color"), color)
    pbdr.append(bottom)
    pPr.append(pbdr)


def h1(doc, text):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(18)
    pf.space_after = Pt(9)
    pf.line_spacing = 1.2
    pf.keep_with_next = True
    style_run(p.add_run(text), size=14, bold=True, color=NAVY)
    _bottom_border(p)
    return p


def h2(doc, text, color=BLUE, size=11.5, before=11):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(4)
    pf.line_spacing = 1.3
    pf.keep_with_next = True
    style_run(p.add_run(text), size=size, bold=True, color=color)
    return p


def bullet(doc, lead, rest, size=10.5):
    p = doc.add_paragraph(style="List Bullet")
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(5)
    pf.line_spacing = 1.45
    pf.left_indent = Cm(0.75)
    pf.first_line_indent = Cm(-0.42)
    if lead:
        style_run(p.add_run(lead), size=size, bold=True, color=NAVY)
    style_run(p.add_run(rest), size=size, color=INK)
    return p


def shade_cell(cell, hexfill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hexfill)
    tcPr.append(shd)


def fill_cell(cell, text, size=9, bold=False, color=INK):
    cell.text = ""
    p = cell.paragraphs[0]
    pf = p.paragraph_format
    pf.space_before = Pt(2.5)
    pf.space_after = Pt(2.5)
    pf.line_spacing = 1.25
    style_run(p.add_run(text), size=size, bold=bold, color=color)


def make_table(doc, rows, widths):
    t = doc.add_table(rows=len(rows), cols=len(widths))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = t.cell(i, j)
            cell.width = Cm(widths[j])
            head = i == 0
            fill_cell(cell, val, size=9, bold=head, color=NAVY if head else INK)
            if head:
                shade_cell(cell, "DCE6F1")
    trPr = t.rows[0]._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)
    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        cs.set(qn("w:val"), "true")
        trPr.append(cs)
    return t


# ---------------------------------------------------------------- 文档初始化
doc = Document()
sec = doc.sections[0]
sec.page_width = Cm(21.0)
sec.page_height = Cm(29.7)
sec.left_margin = Cm(2.2)
sec.right_margin = Cm(2.2)
sec.top_margin = Cm(2.0)
sec.bottom_margin = Cm(2.0)

normal = doc.styles["Normal"]
normal.font.name = FONT
normal.font.size = Pt(10.5)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

core = doc.core_properties
core.title = "简历项目经历：基于 LangChain / LangGraph 的 ReAct Agent 智能客服系统"
core.author = "项目整理"
core.subject = "简历项目经历 + 面试问答手册"

# ---------------------------------------------------------------- 封面区
para(doc, "个人项目 · 简历素材", size=9, color=GRAY, after=2)
para(doc, "基于 LangChain / LangGraph 的 ReAct Agent 智能客服系统",
     size=19, bold=True, color=NAVY, after=2, line=1.25)
para(doc, "简历项目经历 + 面试问答手册", size=12, bold=True, color=BLUE, after=8)
rich(doc, [
    ("适用岗位：", True, GRAY), ("AI 应用开发 / AI Agent 开发 / 大模型应用工程师", False, GRAY),
    ("    整理时间：", True, GRAY), ("2026-11", False, GRAY),
], size=9, after=2)
rich(doc, [
    ("素材来源：", True, GRAY),
    ("本项目全部源码（agent / rag / model / utils / config / prompts）与 logs 下 4 天运行日志，"
     "问题清单均可在日志中回溯。", False, GRAY),
], size=9, after=10)

# ---------------------------------------------------------------- 一
h1(doc, "一、简历主版本（建议使用）")

h2(doc, "项目信息", before=4)
rich(doc, [("项目名称：", True, GRAY),
           ("基于 LangChain / LangGraph 的 ReAct Agent 智能客服系统", False, INK)], indent=0.0)
rich(doc, [("项目角色：", True, GRAY), ("独立开发（Agent 编排 / RAG 链路 / 中间件 / 前端）", False, INK)])
rich(doc, [("项目时间：", True, GRAY), ("2026.11", False, INK)])
rich(doc, [("技术栈：", True, GRAY),
           ("Python 3.10+ · LangChain 1.x · LangGraph · Chroma · 通义千问 qwen-plus / "
            "text-embedding-v1（DashScope）· Streamlit · PyPDF", False, INK)], after=8)

h2(doc, "项目描述")
para(doc, "面向扫地 / 扫拖一体机器人售后的智能客服系统。以 ReAct（Thought → Action → Observation）"
          "范式构建可自主决策的 Agent，融合 RAG 知识库检索、多工具编排与运行态提示词路由，"
          "将「专业知识问答 / 故障排查 / 个性化使用报告生成」三类异构场景收敛到同一入口，"
          "前端提供逐字流式对话与推理链路可视化。", after=8)

h2(doc, "主要工作")
bullet(doc, "Agent 主循环与工具编排：",
       "基于 create_agent 构建 ReAct 循环，注册 7 个 @tool（知识库检索、天气、用户定位、用户 ID、"
       "当前月份、外部使用记录、报告上下文注入），由模型自主完成多步推理与工具选择；"
       "在 System Prompt 中硬约束工具入参格式与 5 次调用上限，规避参数幻觉与工具死循环。")
bullet(doc, "中间件化横切设计（核心）：",
       "用 LangChain middleware 将可观测性与业务逻辑解耦——before_model 记录每轮模型调用的消息数与"
       "最新消息类型；wrap_tool_call 统一采集工具名 / 入参 / 执行结果并做异常兜底；"
       "dynamic_prompt 在每次生成 Prompt 前按 runtime.context 装载不同模板，主流程零侵入。")
bullet(doc, "运行态提示词路由：",
       "设计 fill_context_for_report 作为「语义信号工具」，Agent 识别出报告意图后先调用它，"
       "wrap_tool_call 在回调中置 runtime.context[\"report\"] = True，下一轮 dynamic_prompt "
       "自动切换到报告专用 Prompt；以「工具即状态机迁移信号」替代额外意图分类模型，"
       "零额外推理成本实现双场景人设切换。")
bullet(doc, "RAG 检索链路：",
       "统一加载 PDF + TXT 知识库，按中文标点（\\n\\n / \\n / 。/ ！/ ？/ ；）分块后写入 Chroma "
       "持久化向量库；检索 Top-3 召回，经 LCEL 链（PromptTemplate → 拼装参考资料 → ChatModel → "
       "StrOutputParser）生成基于原文的总结答案。")
bullet(doc, "幂等增量入库：",
       "对每个知识文件计算 MD5 摘要并维护入库清单，已入库文件跳过，避免重复 embedding 计费与"
       "向量库膨胀；单文件解析异常仅记日志并 continue，保证批量导入不被单点故障中断。")
bullet(doc, "提示词工程（三套模板）：",
       "RAG 总结模板强制「仅依据参考资料、多来源冲突时说明差异、无据可查直接声明未提及」；"
       "报告模板要求先按异常维度（耗材寿命 / 覆盖率 / 避障成功率）二次检索保养知识再给建议，"
       "解决「只罗列数据、不给结论」的报告通病。")
bullet(doc, "流式交互与状态保持：",
       "Streamlit session_state 单例缓存 Agent 与历史消息，避免每次 rerun 重建模型客户端与"
       "向量库连接；write_stream 逐字符输出并将 chunk 同步落缓存，实现打字机效果 + "
       "刷新不丢历史 + 推理过程可见。")
bullet(doc, "可观测性与配置治理：",
       "封装 logging 双 handler（控制台 INFO / 文件 DEBUG、按天滚动、exc_info 输出全栈）；"
       "模型名、分块参数、检索 Top-K、路径、提示词全部 YAML 外置，改配置不改代码。")

h2(doc, "项目成果")
bullet(doc, "", "单入口支撑 3 类业务场景，4 份文档共 234 个向量分片稳定召回；")
bullet(doc, "", "报告生成从「模型自由发挥」固化为 5 步工具链（用户 ID → 月份 → 上下文信号 → "
                "使用记录 → 分维度保养检索），产出结构化 Markdown 报告；")
bullet(doc, "", "全链路中间件日志使工具调用类问题可一次定位。")

# ---------------------------------------------------------------- 二
h1(doc, "二、简历精简版（篇幅受限时使用）")

para(doc, "ReAct Agent 智能客服系统｜个人项目｜Python / LangChain / LangGraph / Chroma / Streamlit",
     bold=True, color=NAVY, after=5)
bullet(doc, "", "基于 ReAct 范式搭建自主决策 Agent，中间件化实现工具监控、模型调用日志与运行态"
                "提示词动态路由，单入口覆盖知识问答、故障排查、报告生成三类场景。")
bullet(doc, "", "构建 Chroma + 通义 Embedding 的 RAG 链路（中文分块 200/20、Top-3 召回、LCEL "
                "总结链），设计 MD5 幂等增量入库避免重复 embedding。")
bullet(doc, "", "通过 Prompt 约束检索 query 自包含性与「历史知识不作事实依据」，解决多轮对话检索"
                "漂移与答案编造问题；以 5 次工具调用上限杜绝 Agent 死循环。")
bullet(doc, "", "Streamlit 实现逐字流式输出与 Agent 推理链路可视化，YAML 驱动全量参数配置。")

# ---------------------------------------------------------------- 三
h1(doc, "三、60 秒自我介绍（STAR 版）")

rich(doc, [("S 背景　", True, NAVY),
           ("扫地 / 扫拖机器人售后有两类高频需求——专业问答（故障、保养、选购）与个性化使用报告。"
            "前者依赖知识库准确性，后者依赖业务数据，用单一 Prompt 做不好。", False, INK)], after=6)
rich(doc, [("T 目标　", True, NAVY),
           ("把两类场景收敛到一个入口，让模型自主判断该查知识库、该调业务接口，还是该写报告。", False, INK)], after=6)
rich(doc, [("A 做法　", True, NAVY),
           ("用 ReAct 让模型自主多步推理；用中间件把「日志监控」与「提示词路由」从业务代码里剥离；"
            "用 RAG 保证事实来自知识库；用工具调用作为状态信号实现人设切换。", False, INK)], after=6)
rich(doc, [("R 结果　", True, NAVY),
           ("单入口覆盖三类场景，报告生成从自由发挥固化为 5 步工具链，全链路日志让工具类问题一次定位。", False, INK)], after=8)

# ---------------------------------------------------------------- 四
h1(doc, "四、技术栈与量化事实")

para(doc, "以下均为可直接核对的实现事实，面试报数字时优先用这些，而不是估算的百分比。", size=9.5, color=GRAY, after=6)

rows = [
    ("层级 / 模块", "技术选型", "在本项目中的作用"),
    ("Agent 编排", "LangChain 1.x create_agent + LangGraph Runtime",
     "ReAct 主循环、执行状态与运行态上下文"),
    ("中间件", "before_model / wrap_tool_call / dynamic_prompt",
     "工具监控、模型调用日志、运行态提示词路由"),
    ("对话模型", "通义千问 qwen-plus（DashScope ChatTongyi）", "推理、工具选择与答案生成"),
    ("向量模型", "DashScope text-embedding-v1", "知识文档与查询向量化"),
    ("向量库", "Chroma 持久化（collection = agent）", "234 个分片的相似度检索，Top-3 召回"),
    ("文档处理", "PyPDFLoader / TextLoader / RecursiveCharacterTextSplitter",
     "4 份文档解析（PDF + 3 份 TXT），chunk 200 / overlap 20，中文标点分隔符"),
    ("工具层", "7 个 @tool 函数", "知识检索、天气、定位、用户 ID、月份、外部记录、上下文信号"),
    ("配置与提示词", "4 个 YAML + 3 套 Prompt",
     "参数全外置；问答 / RAG 总结 / 报告三套人设"),
    ("前端", "Streamlit session_state + write_stream", "逐字流式输出、历史保持、推理过程可视化"),
    ("可观测性", "logging 双 handler（console INFO / file DEBUG）",
     "按天分文件、exc_info 全栈、工具链路追踪"),
]
make_table(doc, rows, [3.2, 5.6, 8.0])

# ---------------------------------------------------------------- 五
h1(doc, "五、问题排查与优化清单（面试深挖用）")

para(doc, "以下 7 项均来自真实调试过程，可在 logs/ 目录的日志中回溯对应时间点。", size=9.5, color=GRAY, after=6)

issues = [
    (r"""① TXT 知识文件入库全部失败""",
     r"""3 份 TXT 加载抛 RuntimeError: Error loading ...，知识库只剩 PDF 可用""",
     r"""langchain 的 TextLoader 未指定编码时按系统默认 GBK 读取，而文档为 UTF-8，
抛 UnicodeDecodeError: 'gbk' codec can't decode byte 0xab""",
     r"""txt_loader 显式传 encoding="utf-8"；同时在入库流程增加「文档为空 / 分片为空」校验，
并用单文件 try/except + continue 隔离单点失败""",
     r"""3 份文档（约 107KB）全部入库，日志由 3 条 ERROR 变为 3 条「加载成功」"""),

    (r"""② 知识库路径错乱""",
     r"""日志出现「data 不是文件夹」，紧接着出现「计算文件 txt 的 md5 失败」这种荒唐错误""",
     r"""配置项为相对路径、依赖进程 cwd；且路径校验失败的错误分支把 allowed_types（("txt","pdf")）
当结果返回，被上层当成文件路径遍历""",
     r"""抽 path_tool.get_abs_path()，基于 __file__ 反推项目根目录统一解析绝对路径；错误分支改为返回空集合""",
     r"""彻底消除 cwd 依赖，脚本从任意目录启动均可用；错误不再被静默放大成脏数据"""),

    (r"""③ 反复初始化导致重复入库""",
     r"""多次执行初始化脚本，向量库出现重复分片，embedding 接口被重复调用""",
     r"""每次 load_document 都全量执行 add_documents，无幂等控制""",
     r"""维护 MD5 摘要清单实现幂等增量导入，命中已入库摘要即跳过""",
     r"""实现增量入库、零重复计费；日志可见「内容已经存在于知识库内，跳过」"""),

    (r"""④ 双场景人设互相干扰""",
     r"""问答与报告共用一套 System Prompt，报告输出被客服口吻污染，反之亦然""",
     r"""Prompt 同时承担了「业务知识」与「运行时状态」两种职责""",
     r"""fill_context_for_report 作为语义信号工具 → wrap_tool_call 置 runtime.context["report"]=True
→ dynamic_prompt 在下一轮生成 Prompt 时切换模板""",
     r"""零额外推理成本完成人设切换，主流程无侵入、不需要额外意图分类模型"""),

    (r"""⑤ 多轮对话检索漂移""",
     r"""第二、三轮回答的召回相关性明显下降""",
     r"""模型把历史对话里的城市、上一轮关键词全拼进检索 query，引入检索噪音""",
     r"""Prompt 约束 query 必须自包含且只补全与当前问题直接相关的指代；RAG 总结模板区分
「历史个人信息可用 / 历史硬知识不可用、冲突以参考资料为准」""",
     r"""检索词聚焦当前问题，召回稳定性提升；历史个人信息仍可复用，不必重复追问"""),

    (r"""⑥ 答案编造与「只贴数据不给建议」""",
     r"""模型补全资料里没有的参数；使用报告只罗列数据、不给处置建议""",
     r"""RAG 侧无事实约束；报告侧无二次检索机制""",
     r"""RAG 模板强制仅依据参考资料、无据回答「未提及」、多来源冲突需说明差异；报告模板要求先按
异常维度（耗材 / 覆盖率 / 避障成功率）二次检索保养知识再成文""",
     r"""日志可见「参考资料中未提及…」的准确拒答，以及按 3 个维度分别检索后再成文"""),

    (r"""⑦ 排障靠 print""",
     r"""工具调错、参数传错时无从定位，只能逐行加打印""",
     r"""缺少链路级可观测性""",
     r"""middleware 统一采集工具名 / 入参 / 成败；模型调用前记录消息数与最新消息类型；
文件日志使用 exc_info=True 保留堆栈""",
     r"""工具调用类问题可在日志中一次定位，无需复现"""),
]

for title, symptom, cause, fix, result in issues:
    h2(doc, title, size=11, before=9)
    rich(doc, [("现象：", True, GRAY), (symptom, False, INK)], size=10, after=2, indent=0.3)
    rich(doc, [("根因：", True, GRAY), (cause, False, INK)], size=10, after=2, indent=0.3)
    rich(doc, [("解决：", True, GRAY), (fix, False, INK)], size=10, after=2, indent=0.3)
    rich(doc, [("结果：", True, GRAY), (result, False, INK)], size=10, after=2, indent=0.3)

# ---------------------------------------------------------------- 六
h1(doc, "六、面试追问 Q&A（参考回答）")

qa = [
    (r"""Q1  为什么用 Middleware 而不是 Callback？""",
     r"""层级不同。Callback 注册在事件层（on_llm_start / on_tool_end），只能看、不能改。Middleware
位于 Agent 执行图内部，能同时拿到 state 与 runtime，可以在模型调用前替换 Prompt、在工具调用前后
插入逻辑，属于「控制面」而不是「观察面」。我的场景需要在工具执行后写运行态上下文、在下一轮生成
Prompt 前读它——Callback 既拿不到 runtime，也改不了 System Prompt，所以必须用 Middleware。"""),

    (r"""Q2  提示词切换为什么用「工具调用」当信号，而不是意图分类？""",
     r"""三点。成本：分类器要多一次模型调用（延迟 + 费用），而工具信号复用 Agent 本来就要产出的
action，边际成本为零。稳定性：「是否调用 fill_context_for_report」是确定的结构化事件，比解析自然
语言意图可靠。维护：不需要维护标签体系和标注数据，只在 Prompt 里声明调用约束即可。
代价是依赖模型遵守约束，所以我在主 Prompt 里写成「报告场景必调前置工具、非报告场景严禁调用」，
并在报告 Prompt 的工具清单里显式保留该工具（否则切换后模型会忘记这一步）；兜底还可以在
fetch_external_data 被调用时同样置标记。"""),

    (r"""Q3  runtime.context 为什么能从 wrap_tool_call 传到 dynamic_prompt？""",
     r"""一次 invoke / stream 会创建一个 Runtime，它的 context 就是调用方传入的 dict（我用
context={"report": False}）。这个 Runtime 贯穿本次执行的所有 step，wrap_tool_call 的 request.runtime
与 dynamic_prompt 的 request.runtime 指向同一个对象，所以工具回调里写入的键，下一轮生成 Prompt
时能读到。要注意它只在本轮执行内有效、不跨轮持久化；要跨轮保留就得放进 state 或使用 checkpointer
的 thread 状态。"""),

    (r"""Q4  RAG 召回不准，你怎么排查？""",
     r"""先分清是「没检索到」还是「检索到但答偏」：直接打印 retriever.invoke 的 Top-K 原文。
前者查分块与检索参数（chunk 200 对条目式语料合适，但 PDF 跨页会切断；k=3 偏小、无相似度阈值、
无 MMR 导致结果同质化），后者查 query 与模板约束。我实际遇到的是后者——多轮把历史关键词拼进
query 造成漂移。改进顺序：加相似度阈值过滤 → MMR 提升多样性 → BM25 + 向量混合召回兜专业术语
（HEPA、胶刷等）→ rerank 重排。所有调整都要有评测集验证，不能凭感觉调参。"""),

    (r"""Q5  chunk_size=200 / overlap=20 是怎么定的？""",
     r"""由语料形态决定，不是拍脑袋。知识库是「200 条问答 / 条目」式短文本，单条 50–150 字，
200 字能覆盖 1–2 条且不跨主题；overlap 20 约 10%，防止答案被切断在分块边界。
分隔符特意从默认优先 "\n\n" 换成中文标点（。！？；），因为中文语料常以单换行加句末标点断句，
纯 "\n\n" 会切不开。如果语料换成说明书类长文档，应提升到 500–800，或改成按标题层级切分。"""),

    (r"""Q6  既然要检索，为什么还让 LLM 再总结一遍？""",
     r"""直接拼 Top-K 会有重复段落、元数据噪声和格式混乱，多文档冲突也没有裁决。让模型在「仅依据
参考资料」的约束下压缩成可读答案，输出更稳定，也能按用户问题的颗粒度取舍信息。
代价是引入幻觉风险，所以模板里加了硬约束：无据直说未提及、冲突要说明差异。
进一步可以做引用标注，把 source 元数据带回前端展示。"""),

    (r"""Q7  MD5 去重的边界在哪？""",
     r"""按文件内容摘要去重只能解决「同一文件重复导入」。文件内容被修改后摘要变化会重新入库，
但旧分片仍留在库里，出现新旧并存；删除文件也不会清理向量。生产上应把 source 写进 metadata，
更新时按 source 先 delete 再 add，或用 collection 版本号整体重建。这是我这套实现下一步要补的点。"""),

    (r"""Q8  多轮对话记忆现在是怎么做的？""",
     r"""坦白说没有真正做。execute_stream 每轮只传当轮 user message，Agent 侧不携带历史；
rag_service 里 history 也硬编码为空串。所以「多轮」目前只有前端展示层的连续，模型层面是单轮的。
改法两条：用 LangGraph checkpointer（MemorySaver / SqliteSaver）配 thread_id 自动累积 state；
或在前端把 session_state 的历史转成 messages 传入。RAG 侧再用最近 N 轮做 query 改写以解决指代。"""),

    (r"""Q9  Streamlit 里有哪些坑？流式是真流式吗？""",
     r"""三个坑：① 每次交互都会 rerun 整个脚本，模型客户端与向量库连接必须放 session_state 缓存，
否则每轮重建；② 流式结束后调用 st.rerun() 并向 session_state 追加消息，要保证只追加一次，
否则消息重复渲染；③ 模块级全局对象（如 agent_tools 里的 external_data 缓存）是进程级共享的，
多用户并发时会串数据。
另外要说实话：stream_mode="values" 是按 step 输出整段状态，不是 token 增量；前端逐字符 yield
只是模拟打字机，首字延迟并没有真正优化。要真流式得用 stream_mode="messages" 或 astream_events。"""),

    (r"""Q10  怎么防止死循环和越权？上线还要改什么？""",
     r"""死循环两层：Prompt 层限制 5 次工具调用；框架层设 LangGraph recursion_limit（默认 25 步）兜底。
越权更值得警惕：现在 fetch_external_data 的 user_id 由模型生成，理论上能读到其他用户数据。
正确做法是用户身份从 runtime.context 注入、工具不接受 LLM 传入的用户 ID——模型只负责「要不要查」，
不负责「查谁」。
上线还要补：Mock 工具换真实 API 并加超时 / 重试 / 熔断；DashScope 对话与 embedding 的 QPS 限流；
向量库连接管理；接 LangSmith / OpenTelemetry 记录 token 消耗与工具成功率；建 50–100 条 gold set
做回归，指标看召回率、答案忠实度、工具调用正确率、首字延迟。"""),
]

for q, a in qa:
    h2(doc, q, size=10.5, before=10)
    para(doc, a, size=10, after=2)

# ---------------------------------------------------------------- 七
h1(doc, "七、使用与填写提醒")

bullet(doc, "指标口径：", "如果没有评测集，就不要在简历里写「提升 X%」，用「机制 + 实测事实」表达"
                        "（4 份文档、234 个分片、Top-3、7 个工具、120 条使用记录）。")
bullet(doc, "Mock 工具：", "get_weather / get_user_location / get_user_id 都是 random.choice 假数据。"
                         "面试主动说明「业务工具为占位实现，链路与中间件是真实落地」，比被追问后露馅更有利。")
bullet(doc, "版本一致性：", "requirements.txt 仍锁 LangChain 0.3.7，但代码用的是 1.x 的 create_agent / "
                         "middleware API（实测环境 langchain 1.4.2）。建议改成实际验证过的版本，"
                         "否则别人 pip install 后直接跑不起来。")
bullet(doc, "项目性质：", "若为跟学 / 课程项目，建议写「独立完成 Agent 编排与 RAG 链路，并做了日志体系、"
                       "幂等入库、分维度检索等二次优化」，比写「从 0 设计」更稳，也更经得起追问。")
bullet(doc, "别写没做的：", "checkpointer 多轮记忆、多智能体协作、MCP 协议接入目前都还没落地，"
                         "写进技术栈会成为面试送命题。")

doc.save(OUT)
print("saved:", OUT)
