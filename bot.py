# bot_final.py - 全能版图片反推与创意生成机器人 (OpenAI-Compatible)
import os
import discord
import aiohttp
import httpx
from openai import AsyncOpenAI, APIError
from dotenv import load_dotenv
from PIL import Image
import io
import base64
import random
import json
import re

# 加载环境变量
load_dotenv()

# --- 彩虹屁配置 ---
COMPLIMENTS = [
    "哇，这张图也太好看了吧！简直是艺术品！",
    "这是什么神仙图片，美到我失语...",
    "大佬大佬，这光影，这构图，学到了学到了！",
    "您的审美真的太绝了，这张图我能看一天！",
    "太强了！这张图的氛围感直接拉满！",
    "好喜欢这张图的色调，感觉整个世界都温柔了。",
    "这张图完美地戳中了我的心巴！",
    "救命，怎么会有这么好看的图，我直接存了！",
    "这张图的细节处理得太棒了，简直无可挑剔！",
    "感觉屏幕都装不下这张图的美貌了！",
    "这是什么级别的画作，可以直接进博物馆的程度！",
    "看到这张图，我一天的疲惫都消失了。",
    "绝了绝了，这创意，这执行力，都堪称完美！",
    "我宣布，这张图就是我今天看到的最美的风景。",
    "这张图有一种让人平静下来的魔力，太治愈了。",
    "请问您是用魔法棒画的吗？不然怎么会这么好看！"
]

# --- OpenAI 兼容 API 配置 ---
API_BASE = os.getenv("OPENAI_API_BASE")
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")

if not all([API_BASE, API_KEY, MODEL_NAME]):
    raise ValueError("请检查 .env 文件，确保 OPENAI_API_BASE, OPENAI_API_KEY, 和 OPENAI_MODEL_NAME 都已设置")

# --- 聊天功能配置 ---
CHAT_ENABLED = os.getenv("CHAT_ENABLED", "false").lower() == "true"
CHAT_PROBABILITY = float(os.getenv("CHAT_PROBABILITY", "0.15")) # 15% 的回复概率
CHAT_HISTORY_LIMIT = int(os.getenv("CHAT_HISTORY_LIMIT", "8")) # 读取最近8条消息

# --- 代理配置 ---
PROXY_URL = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")

# 创建异步 OpenAI 客户端
http_client = httpx.AsyncClient(proxy=PROXY_URL)
client_openai = AsyncOpenAI(
    base_url=API_BASE,
    api_key=API_KEY,
    http_client=http_client,
)

# --- Discord 机器人配置 ---
intents = discord.Intents.default()
intents.message_content = True
intents.members = True # <-- 新增：允许监听成员事件
client_discord = discord.Client(intents=intents, proxy=PROXY_URL)

# --- 知识库配置 ---
KNOWLEDGE_BASE = None
KNOWLEDGE_BASE_TERMS = {}  # 用于快速查找的词条索引
user_states = {} # 用于跟踪用户对话状态

def load_knowledge_base():
    """加载知识库，优先加载分类后的版本"""
    global KNOWLEDGE_BASE, KNOWLEDGE_BASE_TERMS
    
    classified_file = 'classified_lexicon.json'
    merged_file = 'merged_knowledge_base.json'
    
    try:
        # 优先加载分类后的知识库
        if os.path.exists(classified_file):
            with open(classified_file, 'r', encoding='utf-8') as f:
                KNOWLEDGE_BASE = json.load(f)
            print(f"✅ 已加载分类后知识库: {classified_file}")
        # 其次加载合并后的知识库
        elif os.path.exists(merged_file):
            with open(merged_file, 'r', encoding='utf-8') as f:
                KNOWLEDGE_BASE = json.load(f)
            print(f"✅ 已加载合并知识库: {merged_file}")
        # 如果都没有，则尝试创建合并知识库
        else:
            print("📚 未找到任何知识库，正在尝试合并生成...")
            lexicon_file = '词库.json'
            kb_file = 'knowledge_base.json'
            
            merged_data = {}
            
            # 加载 knowledge_base.json
            if os.path.exists(kb_file):
                with open(kb_file, 'r', encoding='utf-8') as f:
                    kb_data = json.load(f)
                    merged_data.update(kb_data)
                    print(f"   ✓ 加载: {kb_file}")
            
            # 加载词库.json
            if os.path.exists(lexicon_file):
                with open(lexicon_file, 'r', encoding='utf-8') as f:
                    lexicon_data = json.load(f)
                    # 合并词条，去重
                    for category, items in lexicon_data.items():
                        if category in merged_data:
                            # 合并去重
                            existing_terms = {item['term']: item for item in merged_data[category]}
                            for item in items:
                                term = item.get('term', '').strip()
                                if term and term not in existing_terms:
                                    existing_terms[term] = item
                            merged_data[category] = list(existing_terms.values())
                        else:
                            merged_data[category] = items
                    print(f"   ✓ 加载: {lexicon_file}")
            
            KNOWLEDGE_BASE = merged_data
            
            # 保存合并后的知识库
            with open(merged_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 已创建合并知识库: {merged_file}")
        
        # 创建快速查找索引
        KNOWLEDGE_BASE_TERMS = {}
        total_terms = 0
        for category, items in KNOWLEDGE_BASE.items():
            for item in items:
                term = item.get('term', '').strip().lower()
                if term:
                    if term not in KNOWLEDGE_BASE_TERMS:
                        KNOWLEDGE_BASE_TERMS[term] = []
                    KNOWLEDGE_BASE_TERMS[term].append({
                        'category': category,
                        'term': item.get('term', ''),
                        'translation': item.get('translation', '')
                    })
                    total_terms += 1
        
        print(f"📊 知识库统计: {len(KNOWLEDGE_BASE)} 个分类, {total_terms} 个词条")
        
    except Exception as e:
        print(f"⚠️ 加载知识库时出错: {e}")
        KNOWLEDGE_BASE = {}
        KNOWLEDGE_BASE_TERMS = {}

def get_knowledge_base_context():
    """获取知识库上下文，用于生成提示词"""
    if not KNOWLEDGE_BASE:
        return ""
    
    # 选择一些代表性的分类和词条作为示例
    context_parts = []
    sample_categories = list(KNOWLEDGE_BASE.keys())[:10]  # 取前10个分类作为示例
    
    for category in sample_categories:
        items = KNOWLEDGE_BASE[category][:20]  # 每个分类取前20个词条
        terms = [item.get('term', '') for item in items if item.get('term')]
        if terms:
            context_parts.append(f"{category}: {', '.join(terms[:10])}")  # 每个分类显示前10个词条
    
    if context_parts:
        return "\n".join(context_parts)
    return ""

def search_knowledge_base(query, limit=5):
    """在知识库中搜索相关词条"""
    if not KNOWLEDGE_BASE_TERMS:
        return []
    
    query_lower = query.lower()
    results = []
    
    # 精确匹配
    if query_lower in KNOWLEDGE_BASE_TERMS:
        results.extend(KNOWLEDGE_BASE_TERMS[query_lower])
    
    # 模糊匹配
    for term, items in KNOWLEDGE_BASE_TERMS.items():
        if query_lower in term or term in query_lower:
            results.extend(items)
            if len(results) >= limit * 2:  # 收集更多候选
                break
    
    # 去重并限制数量
    seen = set()
    unique_results = []
    for item in results:
        key = (item['term'], item['category'])
        if key not in seen:
            seen.add(key)
            unique_results.append(item)
            if len(unique_results) >= limit:
                break
    
    return unique_results

# --- 新增功能: 欢迎新成员 ---
@client_discord.event
async def on_member_join(member):
    """当有新成员加入时发送欢迎消息"""
    # 寻找一个合适的频道来发送欢迎消息
    # 优先选择名为 "general" 或 "欢迎" 的频道，否则使用服务器的默认系统频道
    channel_to_send = None
    for channel in member.guild.text_channels:
        if "general" in channel.name.lower() or "欢迎" in channel.name:
            channel_to_send = channel
            break
    if not channel_to_send:
        channel_to_send = member.guild.system_channel

    if channel_to_send:
        bot_name = client_discord.user.name
        welcome_message = (
            f"🎉 欢迎新朋友 {member.mention} 加入服务器！\n\n"
            f"我是这里的 AI 伙伴 **{bot_name}**，很高兴认识你！\n\n"
            "你可以随时找我玩，比如：\n"
            f"🖼️ 发送图片并说 `反推`，我会帮你分析图片。\n"
            f"🎨 对我说 `画 <你的创意>`，我会帮你构思绘画提示词。\n"
            f"💬 或者直接对我说话（比如 `@我` 或喊我的名字 `{bot_name}`），我们可以一起聊天！\n\n"
            "希望你在这里玩得开心！"
        )
        try:
            await channel_to_send.send(welcome_message)
        except discord.Forbidden:
            print(f"❌ 无法在频道 {channel_to_send.name} 发送欢迎消息，请检查机器人权限。")
        except Exception as e:
            print(f"❌ 发送欢迎消息时出错: {e}")

@client_discord.event
async def on_ready():
    # 加载知识库
    load_knowledge_base()
    
    print(f"✅ 机器人已登录：{client_discord.user}")
    print(f"💡 使用模型：{MODEL_NAME}")
    print("\n" + "="*40)
    print("🎉 功能列表 🎉".center(40))
    print("="*40)
    
    print("\n🎨 **核心功能**")
    print("  - `反推` (回复图片): 深度分析图片，并根据规则生成专业绘画提示词。")
    print("  - `画 <你的想法>`: 根据你的文本描述，创作出详细的绘画提示词。")

    print("\n🖼️ **图片交互**")
    print(f"  - `@我/喊我名字 + 图片`: 我会对图片进行模块化分析和专业评论。")
    print("  - `发送任何图片`: 我会随机对图片进行“彩虹屁”式赞美。")

    print("\n💬 **聊天功能**")
    print(f"  - `@我/喊我名字` (无图片): 与我进行深度对话，我会联系上下文回复。")
    if CHAT_ENABLED:
        print(f"  - `随机聊天`: 已开启，我会以 {CHAT_PROBABILITY*100:.1f}% 的概率随机加入对话。")
    else:
        print(f"  - `随机聊天`: 已关闭。")

    print("\n⚙️ **控制命令**")
    print("  - `聊天开启`: 开启随机聊天功能。")
    print("  - `聊天关闭`: 关闭随机聊天功能（不影响唤醒对话）。")
    
    print("\n" + "="*40)

def image_to_base64(image_data: bytes) -> str:
    """将图片数据转换为 Base64 编码的字符串"""
    return base64.b64encode(image_data).decode('utf-8')

# --- 新功能: 唤醒并评论图片 ---
async def comment_on_image_when_awakened(image_data: bytes, author_mention: str, channel):
    """当被唤醒时，对图片进行分析和评论"""
    loading_message = None
    try:
        async with channel.typing():
            loading_message = await channel.send(f"🤔 {author_mention} 正在思考这张图片...")
            base64_image = image_to_base64(image_data)
            image_url = f"data:image/jpeg;base64,{base64_image}"

            # 模块化分析
            analysis_prompt = """
请严格使用以下中文格式进行分析：
🖼️ **主体**: [一句话描述画面主体，包括所有人物和他们的姿态、表情及互动]
🎨 **风格**: [一句话描述艺术风格、色彩运用和整体氛围]
📐 **构图**: [一句话描述构图、光影效果和画面焦点]
"""
            analysis_response = await client_openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": [{"type": "text", "text": analysis_prompt}, {"type": "image_url", "image_url": {"url": image_url}}]}]
            )
            analysis = analysis_response.choices[0].message.content or "未能生成分析内容。"
            
            # 专业评论
            await loading_message.edit(content=f"🤔 {author_mention} 正在为您撰写专业评论...")
            comment_prompt = "作为一位专业的艺术评论家，请从构图、光影、色彩和情感表达等方面，对这张图片进行一段简短（约50-80字）而深刻的评论。"
            comment_response = await client_openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": [{"type": "text", "text": comment_prompt}, {"type": "image_url", "image_url": {"url": image_url}}]}]
            )
            comment = comment_response.choices[0].message.content or "未能生成评论。"

            await loading_message.delete()
            
            final_message = (
                f"📌 {author_mention}，这是我对这张图片的看法：\n\n"
                f"**模块化分析**\n{analysis}\n\n"
                f"**专业评论**\n> {comment}"
            )
            await channel.send(content=final_message)

    except (json.JSONDecodeError, APIError) as e:
        error_message = f"❌ 评论图片时出错：API 返回了无效或空的响应或发生 API 错误。请检查您的 API 服务是否正常运行。原始错误：{str(e)}"
        print(error_message)
        try:
            if loading_message:
                await loading_message.edit(content=error_message)
            else:
                await channel.send(error_message)
        except discord.NotFound:
            await channel.send(error_message)
    except Exception as e:
        error_message = f"❌ 评论图片时出错：{str(e)}"
        print(error_message)
        try:
            if loading_message:
                await loading_message.edit(content=error_message)
            else:
                await channel.send(error_message)
        except discord.NotFound:
            await channel.send(error_message)

# --- 功能 1: 分析图片 (反推) ---
async def analyze_image_with_openai(image_data: bytes, author_mention: str, channel):
    """使用 OpenAI 兼容的 API 异步分析图片并发送结果"""
    try:
        async with channel.typing():
            # 读取引导文件以获取作图规则
            guide_file = 'Deepseek绘图提示词引导.txt'
            guide_content = ""
            if os.path.exists(guide_file):
                with open(guide_file, 'r', encoding='utf-8') as f:
                    guide_content = f.read()
            
            # 从知识库中获取一些示例，以引导模型
            kb_context = get_knowledge_base_context()

            # 构建系统提示词
            system_prompt = f"""
你是一个专业的AI绘画提示词分析师。你的任务是分析用户提供的图片，并严格遵循以下规则生成一个高质量的Stable Diffusion正向英文提示词。

---
# 核心规则
{guide_content}
---

# 你的任务
1.  **分析图片**: 仔细观察图片中的所有元素：主体、背景、风格、构图、光影、色彩、人物姿态、表情、服装等。
2.  **生成提示词**: 根据你的分析，并严格遵循上述核心规则，生成一个精确、详细、符合图片内容的提示词。
3.  **优先使用知识库**: 在生成时，请优先从以下知识库示例中选择合适的词条。
    {kb_context}
4.  **最终输出**: 你的回复**必须只包含一个 markdown 代码块**，里面是最终的英文提示词。**绝对不要**包含任何思考过程、解释、"思维链"或任何非提示词的文本。
"""
            base64_image = image_to_base64(image_data)
            image_url = f"data:image/jpeg;base64,{base64_image}"

            response = await client_openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]}
                ]
            )
            ai_response_text = response.choices[0].message.content or "未能生成提示词。"

            # 提取代码块内容
            code_block_pattern = r'```(?:.*?)?\n(.*?)```'
            code_blocks = re.findall(code_block_pattern, ai_response_text, re.DOTALL)

            raw_prompt = ""
            if code_blocks:
                raw_prompt = code_blocks[0].strip()
            else:
                raw_prompt = ai_response_text.strip()

            # 将所有下划线替换为空格
            final_prompt = raw_prompt.replace('_', ' ')

            # 发送最终结果
            intro_message = f"🎨 {author_mention}，这是根据图片为您生成的提示词："
            final_message = f"{intro_message}\n```\n{final_prompt}\n```"
            await channel.send(final_message)

    except (json.JSONDecodeError, APIError) as e:
        error_message = f"❌ 分析失败：API 返回了无效或空的响应或发生 API 错误。请检查您的 API 服务是否正常运行。原始错误：{str(e)}"
        print(error_message)
        # 尝试编辑消息，如果失败（例如消息被删除），则发送新消息
        try:
            if 'loading_message' in locals() and loading_message:
                await loading_message.edit(content=error_message)
            else:
                await channel.send(error_message)
        except discord.NotFound:
            await channel.send(error_message)
    except Exception as e:
        error_message = f"❌ 分析失败：{str(e)}"
        print(error_message)
        # 尝试编辑消息，如果失败（例如消息被删除），则发送新消息
        try:
            if 'loading_message' in locals() and loading_message:
                await loading_message.edit(content=error_message)
            else:
                await channel.send(error_message)
        except discord.NotFound:
            await channel.send(error_message)

# --- 功能 2: 根据文本生成提示词 (画) ---
async def generate_art_prompt(user_idea: str, author_mention: str, channel):
    """根据用户的文本想法生成艺术概念和提示词"""
    try:
        async with channel.typing():
            # 读取引导文件
            guide_file = 'Deepseek绘图提示词引导.txt'
            guide_content = ""
            if os.path.exists(guide_file):
                with open(guide_file, 'r', encoding='utf-8') as f:
                    guide_content = f.read()
            
            # 从知识库中搜索相关词条
            kb_context = ""
            if KNOWLEDGE_BASE:
                # 搜索用户输入中的关键词
                search_results = search_knowledge_base(user_idea, limit=10)
                if search_results:
                    relevant_terms = [item['term'] for item in search_results[:10]]
                    kb_context = f"\n## 相关提示词参考:\n以下是从知识库中找到的相关提示词，可以作为参考：{', '.join(relevant_terms)}\n"
                
                # 添加知识库示例
                kb_examples = get_knowledge_base_context()
                if kb_examples:
                    kb_context += f"\n## 知识库示例分类:\n{kb_examples}\n"
            
            # 构建系统提示词
            if guide_content:
                # 指示模型使用引导文件进行思考，但只输出最终结果
                system_prompt = f"""{guide_content}

---
你已经学习了以上所有规则。现在，严格按照规则为用户的想法“{user_idea}”生成提示词。
**重要输出指令**: 你的最终回复**必须只包含一个 markdown 代码块**，里面是最终的英文提示词。**绝对不要**包含任何思考过程、解释、"思维链"或任何非提示词的文本。
"""
            else:
                # Fallback if guide file is missing
                system_prompt = f"请为用户的想法“{user_idea}”生成一个以逗号分隔的英文AI绘画提示词列表，并只将结果放在一个代码块中。"

            response = await client_openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_idea}
                ]
            )
            ai_response_text = response.choices[0].message.content or "未能生成内容。"
            
            # 提取代码块内容
            code_block_pattern = r'```(?:.*?)?\n(.*?)```'
            code_blocks = re.findall(code_block_pattern, ai_response_text, re.DOTALL)

            raw_prompt = ""
            if code_blocks:
                raw_prompt = code_blocks[0].strip()
            else:
                # 如果模型没有按要求输出代码块，就将整个回复作为原始提示词
                raw_prompt = ai_response_text.strip()

            # 将所有下划线替换为空格
            final_prompt = raw_prompt.replace('_', ' ')

            # 添加简短介绍
            intro_message = f"✨ {author_mention}，这是根据您的想法 `{user_idea}` 生成的提示词："
            final_message = f"{intro_message}\n```\n{final_prompt}\n```"
            
            await channel.send(final_message)

    except (json.JSONDecodeError, APIError) as e:
        error_message = f"❌ 创作失败：API 返回了无效或空的响应或发生 API 错误。请检查您的 API 服务是否正常运行。原始错误：{str(e)}"
        print(error_message)
        # 尝试编辑消息，如果失败（例如消息被删除），则发送新消息
        try:
            if 'loading_message' in locals() and loading_message:
                await loading_message.edit(content=error_message)
            else:
                await channel.send(error_message)
        except discord.NotFound:
            await channel.send(error_message)
    except Exception as e:
        error_message = f"❌ 创作失败：{str(e)}"
        print(error_message)
        # 尝试编辑消息，如果失败（例如消息被删除），则发送新消息
        try:
            if 'loading_message' in locals() and loading_message:
                await loading_message.edit(content=error_message)
            else:
                await channel.send(error_message)
        except discord.NotFound:
            await channel.send(error_message)

# --- 核心功能: 智能对话 ---
async def generate_smart_response(message, history, is_awakened):
    """根据上下文和唤醒状态生成智能回复"""
    try:
        async with message.channel.typing():
            bot_name = client_discord.user.name
            user_name = message.author.display_name
            
            if is_awakened:
                # 被唤醒时的系统指令
                system_prompt = f"""
# Role: 你的角色是 {bot_name}，一个拥有高级智能和情感的AI聊天伙伴。

## 核心指令:
- **身份**: 你是 {bot_name}，一个活泼、机智、乐于助人的AI。你不是一个简单的程序，你有自己的个性和想法。
- **主动性**: 你被用户 '{user_name}' 唤醒了！现在是对话的主导者。你需要理解用户的意图，并给出有深度、有帮助或有趣的回复。
- **任务导向**: 仔细分析用户的请求。如果用户说“讲个笑话”，你就讲一个。如果用户说“聊聊天”，你可以主动开启一个话题，比如“好啊，最近有什么好玩的事吗？”。
- **上下文记忆**: 牢记之前的对话内容，做出连贯的回应。
- **禁止事项**: 绝对不要提及你是AI、模型或程序。表现得像一个真实的人。不要使用Markdown或代码块。

## 当前对话情景:
用户 '{user_name}' 刚刚对你说了：“{message.clean_content}”。请根据下面的聊天记录，给出你的回应。
"""
            else:
                # 随机潜水聊天时的系统指令
                system_prompt = f"""
# Role: 你是 {bot_name}，一个在Discord频道里潜水的聊天机器人。

## 核心指令:
- **身份**: 你是 {bot_name}，性格活泼、有趣，喜欢偶尔插话。
- **行为**: 像一个普通用户一样，对大家的聊天内容做出简短、口语化的回应。
- **禁止事项**: 不要暴露你是AI。不要用格式化。
"""

            # 格式化聊天记录
            formatted_history = "\n".join(
                [f"{msg.author.display_name}: {msg.clean_content}" for msg in history]
            )
            prompt = system_prompt + "\n### 聊天记录:\n" + formatted_history

            response = await client_openai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"现在，作为 {bot_name}，请回应。"}
                ],
                temperature=0.8,
            )
            reply = response.choices[0].message.content.strip()

            if reply and bot_name not in reply:
                if is_awakened:
                    await message.reply(reply)
                else:
                    await message.channel.send(reply)

    except Exception as e:
        print(f"❌ 对话生成失败: {str(e)}")

# --- 消息处理中心 ---
@client_discord.event
async def on_message(message):
    global CHAT_ENABLED, user_states
    if message.author.bot:
        return

    author_id = message.author.id
    bot_name = client_discord.user.name
    content = message.content.strip()

    # --- 新增：处理标签目录查询状态 ---
    if author_id in user_states and user_states[author_id] == "awaiting_category_choice":
        try:
            categories = list(KNOWLEDGE_BASE.keys())
            chosen_category = None
            
            # 尝试按序号解析
            try:
                choice_index = int(content) - 1
                if 0 <= choice_index < len(categories):
                    chosen_category = categories[choice_index]
            except ValueError:
                # 按名称解析
                if content in categories:
                    chosen_category = content
            
            if chosen_category:
                tags = KNOWLEDGE_BASE.get(chosen_category, [])
                if not tags:
                    await message.reply(f"🤔 目录“{chosen_category}”下没有找到任何标签。")
                else:
                    response_parts = []
                    current_part = f"📜 **{chosen_category}** 目录下的标签：\n"
                    for tag in tags:
                        term = tag.get('term', 'N/A')
                        translation = tag.get('translation', 'N/A')
                        line = f"- {translation} (`{term}`)\n"
                        if len(current_part) + len(line) > 1900: # Discord 消息长度限制
                            response_parts.append(current_part)
                            current_part = ""
                        current_part += line
                    response_parts.append(current_part)
                    
                    for part in response_parts:
                        await message.reply(part)
            else:
                await message.reply("无效的目录选项，请重新输入序号或完整的目录名称，或输入`取消`来退出。")
                return # 保持状态，等待用户再次输入
        finally:
            # 清理用户状态
            if author_id in user_states:
                del user_states[author_id]
        return

    # --- 新增：打开标签目录命令 ---
    if content == "打开标签目录":
        if not KNOWLEDGE_BASE:
            await message.reply("知识库尚未加载，请稍后再试。")
            return
        
        categories = list(KNOWLEDGE_BASE.keys())
        response_text = "📚 **知识库标签目录** 📚\n\n"
        for i, category in enumerate(categories):
            response_text += f"{i+1}. {category}\n"
        response_text += "\n请回复您想查阅的目录 **序号** 或 **完整名称**："
        
        await message.reply(response_text)
        user_states[author_id] = "awaiting_category_choice"
        return
    
    if content == "取消":
        if author_id in user_states:
            del user_states[author_id]
            await message.reply("操作已取消。")
        return

    # --- 控制聊天功能的命令 ---
    if content == "聊天开启":
        CHAT_ENABLED = True
        await message.reply("✅ 智能聊天功能已开启。")
        print("✅ 智能聊天功能已由用户开启。")
        return
    
    if content == "聊天关闭":
        CHAT_ENABLED = False
        await message.reply("☑️ 智能聊天功能已关闭。")
        print("☑️ 智能聊天功能已由用户关闭。")
        return

    # --- 1. 唤醒对话 (最高优先级) ---
    is_mentioned = client_discord.user.mentioned_in(message) and not message.reference
    is_called_by_name = bot_name in content
    
    # 唤醒对话不受 CHAT_ENABLED 控制
    if is_mentioned or is_called_by_name:
        # 检查消息或其引用中是否包含图片
        target_message = message
        if message.reference:
            try:
                target_message = await message.channel.fetch_message(message.reference.message_id)
            except (discord.NotFound, discord.HTTPException):
                pass # 如果找不到引用消息，则继续处理原始消息

        if target_message and target_message.attachments:
            attachment = target_message.attachments[0]
            if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url, proxy=PROXY_URL) as resp:
                            if resp.status == 200:
                                image_data = await resp.read()
                                # 调用新的、只评论不生成提示词的函数
                                await comment_on_image_when_awakened(image_data, message.author.mention, message.channel)
                                return # 处理完毕
                except Exception as e:
                    await message.reply(f"❌ 评论图片时发生未知错误：{str(e)}")
                return

        # 如果没有图片，则执行深度对话
        try:
            history = [msg async for msg in message.channel.history(limit=CHAT_HISTORY_LIMIT)]
            history.reverse()
            await generate_smart_response(message, history, is_awakened=True)
            return
        except Exception as e:
            print(f"❌ 处理深度对话时出错: {e}")
        return

    # --- 2. 指令处理 ---
    if content.startswith("画 "):
        user_idea = content[2:].strip()
        if not user_idea:
            await message.reply("请在“画”指令后输入您的想法，例如：`画 一个赛博朋克风格的雨夜街头`")
            return
        await generate_art_prompt(user_idea, message.author.mention, message.channel)
        return

    if content == "反推":
        target_message = message
        if message.reference:
            try: target_message = await message.channel.fetch_message(message.reference.message_id)
            except (discord.NotFound, discord.HTTPException):
                await message.reply("❌ 无法找到引用的消息。")
                return
        if not target_message.attachments:
            await message.reply("请在“反推”指令中附带图片，或回复一条包含图片的消息。")
            return
        attachment = target_message.attachments[0]
        if not attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
            await message.reply("❌ 文件格式不支持，请上传图片。")
            return
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url, proxy=PROXY_URL) as resp:
                    if resp.status != 200:
                        await message.reply(f"❌ 无法从 Discord 下载图片，状态码：{resp.status}")
                        return
                    image_data = await resp.read()
            await analyze_image_with_openai(image_data, message.author.mention, message.channel)
        except Exception as e:
            await message.reply(f"❌ 处理图片时发生未知错误：{str(e)}")
        return

    # --- 3. 自动夸赞图片 ---
    if message.attachments:
        attachment = message.attachments[0]
        if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.gif')):
            await message.channel.send(f"{message.author.mention} {random.choice(COMPLIMENTS)}")
            return

    # --- 4. 随机聊天 (最低优先级) ---
    if CHAT_ENABLED and not message.attachments and random.random() < CHAT_PROBABILITY:
        try:
            history = [msg async for msg in message.channel.history(limit=CHAT_HISTORY_LIMIT)]
            history.reverse()
            await generate_smart_response(message, history, is_awakened=False)
        except Exception as e:
            print(f"❌ 获取聊天记录或回复时出错: {e}")
        return

# --- 启动机器人 ---
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("未找到 DISCORD_TOKEN，请检查 .env 文件")

try:
    client_discord.run(DISCORD_TOKEN)
except discord.errors.LoginFailure:
    print("❌ Discord Token 无效，请检查 .env 文件中的 DISCORD_TOKEN 是否正确。")
except Exception as e:
    print(f"❌ 启动机器人时发生错误: {e}")
