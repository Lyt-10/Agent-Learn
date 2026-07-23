"""Generate RAG explanation Word document and save to Desktop."""
import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

doc = Document()

# Style
style = doc.styles['Normal']
font = style.font
font.name = 'Arial'
font.size = Pt(11)

# Title
title = doc.add_heading('RAG 实现详解', level=0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph(
    '—— 基于 Agent K 项目的 ChromaDB + 智谱 GLM Embedding 方案',
).alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph('')

# ===== Part 1 =====
doc.add_heading('第一部分：总体理解', level=1)

doc.add_heading('一句话说清楚 RAG', level=2)
doc.add_paragraph(
    'RAG（检索增强生成）= 给 LLM 配一个私人图书管理员。\n\n'
    '你问 LLM 一个问题 → 图书管理员先去你的私人书库翻出相关章节 → '
    '把这些章节贴在问题下面 → 一起交给 LLM → LLM 基于真实资料回答，而不是瞎编。'
)

doc.add_heading('打个比方：考试场景', level=2)

doc.add_paragraph('没有 RAG（闭卷考试）：', style='List Bullet')
doc.add_paragraph(
    '你问“Agent K 项目的核心概念是什么？”\n'
    'LLM 靠记忆瞎猜 → 可能对、可能编造 ❌'
)

doc.add_paragraph('有了 RAG（开卷考试）：', style='List Bullet')
doc.add_paragraph(
    '你问“Agent K 项目的核心概念是什么？”\n'
    '→ 先去自己的笔记本里翻到相关页码\n'
    '→ 把那一页内容连同问题一起递给 LLM\n'
    '→ LLM 照着资料回答 ✅ 准确、可溯源'
)

doc.add_heading('这套 RAG 由哪几块组成？', level=2)
doc.add_paragraph('一共 3 个文件，各司其职：')

table = doc.add_table(rows=4, cols=3, style='Light Grid Accent 1')
table.cell(0, 0).text = '文件'
table.cell(0, 1).text = '类比'
table.cell(0, 2).text = '做了什么'
table.cell(1, 0).text = 'core/rag.py'
table.cell(1, 1).text = '书库 + 管理员'
table.cell(1, 2).text = '存文档、查文档、管理向量数据'
table.cell(2, 0).text = 'tools/builtins/rag.py'
table.cell(2, 1).text = '借书卡'
table.cell(2, 2).text = '把书库包装成标准化工具，LLM 看得懂怎么用'
table.cell(3, 0).text = 'examples/chatbot_with_rag/main.py'
table.cell(3, 1).text = '整个图书馆场景'
table.cell(3, 2).text = '启动时上书，对话时自动查书、引书回答'

doc.add_paragraph('')
doc.add_paragraph('数据流向：')
doc.add_paragraph(
    '你的文档 (.txt/.md)\n'
    '     |\n'
    '     v\n'
    '+- Phase 1: 索引 -+  <-- 启动时执行一次\n'
    '|  切成小块         |\n'
    '|  每块转成向量     |\n'
    '|  存入 ChromaDB   |\n'
    '+----------------+\n'
    '     |\n'
    '     v\n'
    '+- Phase 2: 对话 -+  <-- 每次提问都走这个循环\n'
    '|  用户提问         |\n'
    '|  -> 问题转成向量   |\n'
    '|  -> 搜索最相似片段 |\n'
    '|  -> 附在问题后面   |\n'
    '|  -> 发给 LLM 回答  |\n'
    '+----------------+'
)

# ===== Part 2 =====
doc.add_heading('第二部分：核心概念——什么是“向量”？', level=1)
doc.add_paragraph('这是理解 RAG 最关键的一步，讲完这个概念后面的代码就一目了然了。')

doc.add_heading('向量 = 把文字翻译成数字密码', level=2)
doc.add_paragraph('想象你有一个神奇的函数，叫 embedding()：')

p = doc.add_paragraph()
run = p.add_run(
    'embedding("猫")  ->  [0.8, 0.2, -0.5, 0.9, ...]   # 一串数字\n'
    'embedding("狗")  ->  [0.7, 0.3, -0.4, 0.8, ...]   # 和“猫”的数字很接近！\n'
    'embedding("电脑") -> [-0.3, 0.9, 0.6, -0.2, ...]  # 和“猫”的数字差异很大'
)
run.font.name = 'Consolas'
run.font.size = Pt(9)

doc.add_paragraph(
    '它的神奇之处在于：语义相近的词，向量数字也相近。\n\n'
    '• “猫”和“狗”都是宠物 → 向量距离近（比如距离 0.1）\n'
    '• “猫”和“电脑”毫无关系 → 向量距离远（比如距离 0.8）'
)

doc.add_paragraph(
    '这就意味着：你不需要精确匹配关键词，用意思相近的话就能搜到。\n\n'
    '比如你问「怎么养猫」，知识库里有篇「宠物饲养指南」——虽然字面上没有“猫”字，'
    '但向量距离很近，所以能被搜出来。这就是 RAG 比普通关键词搜索厉害的地方。'
)

# ===== Part 3 =====
doc.add_heading('第三部分：逐步拆解代码', level=1)
doc.add_paragraph('现在从启动到一次完整问答，一步步走一遍。')

# Step 1
doc.add_heading('第一步：引擎初始化（RAGEngine.__init__）', level=2)
doc.add_paragraph('代码位置：core/rag.py 第 39-87 行', style='List Bullet')
doc.add_paragraph('engine = RAGEngine(collection_name="tutorial_kb") 时，背后发生了三件事：')

doc.add_heading('1. 确定数据存哪里', level=3)
doc.add_paragraph(
    '默认存储路径为 ./rag_data/chroma_db/。和项目里的 chat_memory/ 一样——'
    '重启程序后之前索引的文档还在。'
)

doc.add_heading('2. 确定 embedding 用哪个服务（三级优先级）', level=3)
table2 = doc.add_table(rows=4, cols=3, style='Light Grid Accent 1')
table2.cell(0, 0).text = '优先级'
table2.cell(0, 1).text = '来源'
table2.cell(0, 2).text = '说明'
table2.cell(1, 0).text = '1'
table2.cell(1, 1).text = '函数参数'
table2.cell(1, 2).text = '代码里直接传的'
table2.cell(2, 0).text = '2'
table2.cell(2, 1).text = '专用环境变量'
table2.cell(2, 2).text = '.env 里的 EMBEDDING_*'
table2.cell(3, 0).text = '3'
table2.cell(3, 1).text = 'LLM 环境变量'
table2.cell(3, 2).text = '.env 里的 OPENAI_*（兜底）'

doc.add_paragraph('')
doc.add_paragraph(
    '本项目的配置：对话用 DeepSeek，embedding 用智谱 GLM。'
    '所以 EMBEDDING_BASE_URL 和 EMBEDDING_MODEL_ID 必须单独设，'
    '因为 DeepSeek 不提供 embedding 服务。'
)

doc.add_heading('3. 创建 ChromaDB 连接', level=3)
doc.add_paragraph('创建三个角色：')
table3 = doc.add_table(rows=4, cols=3, style='Light Grid Accent 1')
table3.cell(0, 0).text = '角色'
table3.cell(0, 1).text = '类型'
table3.cell(0, 2).text = '干什么'
table3.cell(1, 0).text = '_embedding_fn'
table3.cell(1, 1).text = 'OpenAIEmbeddingFunction'
table3.cell(1, 2).text = '调智谱 API 把文字转成向量'
table3.cell(2, 0).text = '_client'
table3.cell(2, 1).text = 'PersistentClient'
table3.cell(2, 2).text = '和磁盘上的 ChromaDB 数据库通信'
table3.cell(3, 0).text = '_collection'
table3.cell(3, 1).text = 'Collection'
table3.cell(3, 2).text = '书库里的一个“书架”，存放某个集合的所有文档'

doc.add_paragraph('')
doc.add_paragraph(
    'get_or_create_collection 的意思是：如果 tutorial_kb 这个书架已存在'
    '（上次运行时创建的），就用现成的；不存在就新建一个空的。'
    '这就是为什么重启程序后知识库还在——数据持久化在 ./rag_data/chroma_db/ 文件夹里。'
)

# Step 2
doc.add_heading('第二步：索引文档（启动时）', level=2)
doc.add_paragraph('代码位置：examples/chatbot_with_rag/main.py 第 88-107 行', style='List Bullet')

doc.add_paragraph('调用链路：')
p = doc.add_paragraph()
run = p.add_run(
    'index_knowledge_base()\n'
    '  -> engine.index_directory("knowledge/")\n'
    '    -> engine.index_file("knowledge/sample.md")\n'
    '      -> 读文件内容\n'
    '      -> _chunk_content() 切成小块\n'
    '      -> index_text() 每块存入 ChromaDB'
)
run.font.name = 'Consolas'
run.font.size = Pt(9)

doc.add_heading('为什么要分块？', level=3)
doc.add_paragraph(
    '因为 embedding 模型一次只能处理有限长度的文本（通常几百到几千字），'
    '把 100 页 PDF 整本塞进去是不行的。所以要把大文档切成合适大小的小块。'
)

doc.add_heading('分块策略', level=3)
p = doc.add_paragraph()
run = p.add_run('默认每个 chunk 上限 2000 字符')
run.bold = True
doc.add_paragraph(
    '• Markdown 文件：按 ## 二级标题切分。比如 sample.md 的三个章节各自成为一个 chunk。'
    '如果某个章节特别长，再按段落二次切分。',
    style='List Bullet'
)
doc.add_paragraph(
    '• 纯文本文件：按空行（\\n\\n）切分。尽量把相邻的段落合并到同一个 chunk，'
    '直到快超过 2000 字符才另起一个。',
    style='List Bullet'
)

doc.add_paragraph('以 sample.md 为例，分块后：')
p = doc.add_paragraph()
run = p.add_run(
    'Chunk 0: "## 什么是 Agent K？\\nAgent K 是一个从零开始学习..."\n'
    'Chunk 1: "## 核心概念\\n### Node（节点）\\nNode 是 Agent 的最小执行单元..."\n'
    'Chunk 2: "### Workflow = Node + Node\\n把两个 Node 串起来就是..."\n'
    'Chunk 3: "## 技术栈\\n- Python 3.13+..."\n'
    'Chunk 4: "## RAG（检索增强生成）\\nRAG = VectorDB..."'
)
run.font.name = 'Consolas'
run.font.size = Pt(9)

doc.add_heading('存入向量库（index_text）', level=3)
doc.add_paragraph(
    '调用 collection.add() 时，ChromaDB 在背后做了三件事：\n\n'
    '1. 用 embedding_fn 把文本转成一串 1024 维的向量数字\n'
    '2. 把原始文本和元数据（来源文件、第几个 chunk）存下来\n'
    '3. 把向量建好索引，方便后续快速搜索\n\n'
    '元数据保证了搜出来的结果能告诉你“这段文字来自哪个文件的第几块”，'
    'LLM 就可以据此引用来源。'
)

# Step 3
doc.add_heading('第三步：搜索文档（用户提问时）', level=2)

doc.add_heading('向量搜索（search 方法）', level=3)
doc.add_paragraph(
    '当用户问“什么是 Agent K？”，ChromaDB 内部做：\n\n'
    '1. 用同样的 embedding 模型把「什么是 Agent K？」转成向量\n'
    '2. 在库里找和这个向量距离最近的 5 个 chunk\n'
    '3. 返回原始文本、元数据和距离分数'
)

doc.add_paragraph('返回的数据结构示例：')
p = doc.add_paragraph()
run = p.add_run(
    '[\n'
    '  {\n'
    '    "id": "sample.md:0",\n'
    '    "text": "## 什么是 Agent K？\\nAgent K 是一个从零开始学习...",\n'
    '    "metadata": {"source": "knowledge/sample.md", "chunk_index": 0},\n'
    '    "distance": 0.363  <-- 越小越相关\n'
    '  },\n'
    '  {\n'
    '    "id": "sample.md:4",\n'
    '    "text": "## RAG（检索增强生成）\\nRAG = VectorDB...",\n'
    '    "metadata": {"source": "knowledge/sample.md", "chunk_index": 4},\n'
    '    "distance": 0.512  <-- 距离更大，相关性更低\n'
    '  },\n'
    '  ...\n'
    ']'
)
run.font.name = 'Consolas'
run.font.size = Pt(8.5)

doc.add_paragraph(
    'distance 是两个向量之间的“距离”。距离越小 = 语义越接近。'
    '0.363 比 0.512 更相关。'
)

doc.add_heading('格式化输出（query 方法）', level=3)
doc.add_paragraph(
    '把结构化结果转成人（和 LLM）能看懂的文字格式。最终发给 LLM 的内容长这样：'
)
p = doc.add_paragraph()
run = p.add_run(
    '[1] (相关度: 0.363) 来源: knowledge/sample.md\n'
    '## 什么是 Agent K？\n'
    'Agent K 是一个从零开始学习 Agent 开发的教程项目...\n\n'
    '[2] (相关度: 0.512) 来源: knowledge/sample.md\n'
    '## RAG（检索增强生成）\n'
    'RAG = VectorDB。核心思路很简单：\n'
    '1. 把文档切成小块...'
)
run.font.name = 'Consolas'
run.font.size = Pt(9)

# Step 4
doc.add_heading('第四步：把搜索包装成工具（create_rag_tool）', level=2)
doc.add_paragraph('代码位置：tools/builtins/rag.py', style='List Bullet')

doc.add_paragraph(
    '把 engine.query() 包一层，变成符合 Agent 工具协议的对象。'
    'LLM 看到这个工具的 description 和 parameters，就知道什么时候该调用它、传什么参数。'
    '这跟项目里已有的 search（网页搜索）、read（读文件）等工具格式完全一样。'
)

doc.add_heading('为什么用“闭包”模式？', level=3)
doc.add_paragraph(
    'rag_search 函数内部引用了外面的 engine 变量——这就是闭包。'
    '好处是 rag_search 天然绑定到某个特定的 RAGEngine 实例，不需要全局变量。'
)

# Step 5
doc.add_heading('第五步：完整对话流程', level=2)
doc.add_paragraph('代码位置：examples/chatbot_with_rag/main.py 第 110-163 行', style='List Bullet')

doc.add_paragraph('追踪一次完整对话——用户输入“Agent K 是什么？”：')

p = doc.add_paragraph()
run = p.add_run('第一轮 —— ChatNode')
run.bold = True
p.add_run(
    '\n把消息列表发给 LLM，附带所有工具定义（包括 rag_search）。'
    'LLM 读完你的问题，想了一下：“这个问题我应该查知识库”，返回 tool_call: rag_search。'
    '\n→ 路由到 ToolCallNode'
)

p = doc.add_paragraph()
run = p.add_run('ToolCallNode')
run.bold = True
p.add_run(
    '\n解析 LLM 的 tool_call → 调用 rag_search("Agent K 是什么？")'
    '\n→ engine.query() → ChromaDB 搜索 → 返回片段'
    '\n→ 把搜索结果作为“tool 消息”追加到对话历史'
    '\n→ 路由回 ChatNode'
)

p = doc.add_paragraph()
run = p.add_run('第二轮 —— ChatNode')
run.bold = True
p.add_run('\nLLM 再次被调用。这次对话历史变成了：')
p2 = doc.add_paragraph()
run2 = p2.add_run(
    '[用户: "Agent K是什么？"]\n'
    '[助手: 调用 rag_search]\n'
    '[工具结果: "[1] (相关度:0.363) 来源:sample.md\\n## 什么是Agent K？..."]'
)
run2.font.name = 'Consolas'
run2.font.size = Pt(9)

p3 = doc.add_paragraph()
p3.add_run('LLM 根据这段真实资料作答，返回文本 → 路由到 done，结束本轮对话。')

doc.add_paragraph('')
p = doc.add_paragraph()
run = p.add_run('关键点：')
run.bold = True
p.add_run(
    'LLM 不是“搜索了知识库”，而是它看到的对话历史上就有搜索结果。'
    'LLM 以为自己只是在和一个贴了资料的人聊天，然后基于资料回答——这就是 RAG 的本质。'
)

# ===== Part 4 =====
doc.add_heading('第四部分：总结', level=1)

doc.add_heading('一张图看完整流程', level=2)

p = doc.add_paragraph()
run = p.add_run(
    '程序启动\n'
    '  |\n'
    '  +- RAGEngine() -> 连接 ChromaDB + 配置 embedding 服务\n'
    '  +- index_directory("knowledge/") -> 扫描文件 -> 切块 -> 转向量 -> 存入 ChromaDB\n'
    '  +- create_rag_tool(engine) -> 把搜索功能包装成工具\n'
    '  |\n'
    '  +- 对话循环开始 --------------------+\n'
    '       |                               |\n'
    '       v                               |\n'
    '   用户输入                            |\n'
    '       |                               |\n'
    '       v                               |\n'
    '   ChatNode -> 发给 LLM（带 rag_search 工具）|\n'
    '       |                               |\n'
    '       +- LLM 回复文本 -> 打印 -> done   |\n'
    '       |                               |\n'
    '       +- LLM 调用 rag_search ---------+|\n'
    '             |                         ||\n'
    '             v                         ||\n'
    '          ToolCallNode                 ||\n'
    '             |                         ||\n'
    '             v                         ||\n'
    '          query -> search -> ChromaDB  ||\n'
    '             |        |               ||\n'
    '             |        +- 问题 -> 向量  ||\n'
    '             |        +- 向量相似度搜索||\n'
    '             |        +- 返回最相关chunk||\n'
    '             |                         ||\n'
    '             v                         ||\n'
    '          结果追加到对话历史           ||\n'
    '             |                         ||\n'
    '             +--> 回到 ChatNode ---------+|\n'
    '                                        |\n'
    '             <-- 循环继续 ----------------+\n'
)
run.font.name = 'Consolas'
run.font.size = Pt(8)

doc.add_heading('三个核心文件职责', level=2)
table4 = doc.add_table(rows=4, cols=3, style='Light Grid Accent 1')
table4.cell(0, 0).text = '文件'
table4.cell(0, 1).text = '类比'
table4.cell(0, 2).text = '做了什么'
table4.cell(1, 0).text = 'core/rag.py'
table4.cell(1, 1).text = '书库 + 管理员'
table4.cell(1, 2).text = '存文档、查文档、管理向量数据'
table4.cell(2, 0).text = 'tools/builtins/rag.py'
table4.cell(2, 1).text = '借书卡'
table4.cell(2, 2).text = '把书库包装成标准化工具，LLM 看得懂怎么用'
table4.cell(3, 0).text = 'examples/chatbot_with_rag/main.py'
table4.cell(3, 1).text = '整个图书馆场景'
table4.cell(3, 2).text = '启动时上书，对话时自动查书、引书回答'

doc.add_paragraph('')

doc.add_heading('如果你只记住一件事', level=2)
p = doc.add_paragraph()
run = p.add_run(
    'RAG 的本质不是魔法，就是三步：文档 → 向量 → 搜索。'
)
run.bold = True
p.add_run(
    'LLM 从来没“读过”你的文档，它只是在回答问题前，'
    '有人（ChromaDB）先帮它翻了书，把相关段落贴在问题下面。LLM 看了这些段落，自然就能答对。'
)

# Save
desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
filepath = os.path.join(desktop, 'RAG详解.docx')
doc.save(filepath)
print(f'Saved to: {filepath}')
