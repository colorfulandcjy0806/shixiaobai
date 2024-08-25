from bs4 import BeautifulSoup
import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openxlab.model import download
import pickle
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import re
import os
import json
import requests
from lagent.actions.base_action import BaseAction, tool_api
from lagent.actions.parser import BaseParser, JsonParser
from lagent.schema import ActionReturn, ActionStatusCode
import nltk
Settings.llm = None  # 如果不需要LLM，可以禁用
# 修改 base_path，使用新的模型路径
base_path = './ShiXiaobai_history'

# 克隆并拉取模型文件
os.system('apt install git')
os.system('apt install git-lfs')
os.system(f'git clone https://code.openxlab.org.cn/chenjunyang/ShiXiaobai_history.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')

# 新增：将 search_baidu 替换进来
def search_baidu(keyword, limit=5):
    # 检查关键词是否为空
    if not keyword:
        return {"error": "未提供搜索内容"}

    # 构造百度搜索的URL
    search_url = "http://www.baidu.com/s"
    params = {'wd': keyword}

    # 发送HTTP GET请求
    response = requests.get(search_url, params=params)

    # 检查响应状态码是否为200
    if response.status_code == 200:
        # 使用BeautifulSoup解析HTML内容
        soup = BeautifulSoup(response.text, 'html.parser')
        # 获取摘要信息
        zongjie = soup.find('p', class_='cu-line-clamp-4').get_text().strip() if soup.find('p',
                                                                                           class_='cu-line-clamp-4') else ""

        # 设置计数器，限制最多返回3条结果
        count = 0

        # 遍历所有搜索结果的标题
        for result in soup.find_all('h3'):
            if count >= limit:
                break  # 超过限制则退出循环

            title = result.get_text().strip()
            # 获取搜索结果的链接
            link = result.find('a', href=True)['href'] if result.find('a', href=True) else ""
            # 返回搜索结果的摘要、标题和链接
            yield {
                "summary": zongjie,
                "title": title,
                "url": link
            }

            count += 1
    else:
        # 如果请求失败，返回错误信息
        yield {"error": "搜索失败"}

class MagicMaker(BaseAction):
    styles_option = [
        'dongman',  # 动漫
        'guofeng',  # 国风
        'xieshi',   # 写实
        'youhua',   # 油画
        'manghe',   # 盲盒
    ]
    aspect_ratio_options = [
        '16:9', '4:3', '3:2', '1:1',
        '2:3', '3:4', '9:16'
    ]

    def __init__(self,
                 style='guofeng',
                 aspect_ratio='4:3'):
        super().__init__()
        if style in self.styles_option:
            self.style = style
        else:
            raise ValueError(f'The style must be one of {self.styles_option}')
        
        if aspect_ratio in self.aspect_ratio_options:
            self.aspect_ratio = aspect_ratio
        else:
            raise ValueError(f'The aspect ratio must be one of {aspect_ratio}')
    
    @tool_api
    def generate_image(self, keywords: str) -> dict:
        try:
            response = requests.post(
                url='https://magicmaker.openxlab.org.cn/gw/edit-anything/api/v1/bff/sd/generate',
                data=json.dumps({
                    "official": True,
                    "prompt": keywords,
                    "style": self.style,
                    "poseT": False,
                    "aspectRatio": self.aspect_ratio
                }),
                headers={'content-type': 'application/json'}
            )
        except Exception as exc:
            return ActionReturn(
                errmsg=f'MagicMaker exception: {exc}',
                state=ActionStatusCode.HTTP_ERROR)
        image_url = response.json()['data']['imgUrl']
        return {'image': image_url}
    
# 初始化模型和索引
def init_models():
    index_file_path = os.path.join(base_path, "datasets/index.pkl")
    # 加载索引
    with open(index_file_path, "rb") as f:
        index = pickle.load(f)

    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_path + "/china_history", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_path + "/china_history", trust_remote_code=True, torch_dtype=torch.float16).cuda()
    
    return index, tokenizer, model


def chat(message, history, enable_rag, enable_baidu_search, enable_image_generation, index, tokenizer, model):
    image_html = ""  # 存储图片的 HTML
    image_generation_message = ""  # 存储图片生成的反馈消息

    # 默认提示消息
    final_response = "别着急，史老师正在思考……"

    if not history:
        system_prompt = "user: 你是史小白，一位聪明的中国历史老师，擅长循循善诱地教导学生学习。\nassistant: 明白，我是史小白，我将帮助你学习中国历史！"
        history = [[system_prompt, ""]]

    # 处理 RAG 检索
    if enable_rag:
        query_engine = index.as_query_engine()
        retrieval_result = query_engine.query(message)
        retrieved_content = retrieval_result.response
        retrieved_content = retrieved_content.replace("\n", "")
        message = f"{message}\n\n相关资料：{retrieved_content}"

    # 处理百度检索
    elif enable_baidu_search:
        search_summaries = search_baidu(message)
        retrieved_content = "\n\n".join([f"标题: {s['title']}\n摘要: {s['summary']}\n链接: {s['url']}" for s in search_summaries])
        message = f"{message}\n\n网络检索结果：{retrieved_content}"

    # 处理图片生成
    if enable_image_generation:
        result = guofeng_agent.generate_image(message)
        image_url = result.get('image', None)  # 获取生成图片的 URL
        if image_url:
            image_html = f'<img src="{image_url}" alt="生成的中国历史古风图" style="max-width: 100%;">'
            image_generation_message = "好呀，我已经通过我的画图工具帮你生成了你想要的中国历史古风图。"
            message = f"你已经成功利用工具，生成了关于{message}的图"
        else:
            image_generation_message = "生成图片失败"
    
    # 模型生成对话
    for response, history in model.stream_chat(tokenizer, message, history, max_new_tokens=2048, top_p=0.7, temperature=1):
        final_response = response
        history = history

    # 返回对话内容、图片和生成提示
    return final_response + "\n" + image_generation_message, history, image_html if enable_image_generation else ""

# 加载模型和索引
index, tokenizer, model = init_models()
guofeng_agent = MagicMaker(style='guofeng')


# 使用自定义 CSS 来增强古风效果
css = f"""
body {{
    background-size: cover;
    background-position: center;
    font-family: 'KaiTi', '楷体', serif; /* 使用楷体，增加古风感觉 */
    color: #4B2E2E; /* 更加柔和的深棕色字体，适合古风 */
    margin: 0;
    padding: 0;
    overflow-x: hidden;
}}

h1, h2 {{
    font-family: 'KaiTi', '楷体', serif; /* 标题使用楷体 */
    color: #8B4513; /* 棕色调标题 */
    text-align: center;
}}

.gr-button {{
    background-color: #A0522D; /* 棕色按钮背景 */
    border-radius: 8px;
    font-family: 'KaiTi', '楷体';
    color: #FFFFFF;
    border: 1px solid #8B4513; /* 棕色边框 */
    padding: 10px 20px;
    font-size: 18px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}}

.gr-button:hover {{
    background-color: #8B4513; /* 更深的棕色悬停效果 */
}}

textarea, input {{
    font-family: 'KaiTi', '楷体';  /* 输入框和文本区域使用楷体 */
    font-size: 16px;
    padding: 10px;
    border: 2px solid #8B4513; /* 棕色边框 */
    border-radius: 8px;
    width: 90%;
    max-width: 800px;
    margin: 10px auto;
    display: block;
    background-color: rgba(255, 248, 220, 0.8); /* 背景略带透明度 */
}}

.gradio-container {{
    padding: 20px;
    max-width: 1200px;
    margin: 0 auto;
    background-color: rgba(255, 248, 220, 0.7);  /* 背景半透明效果 */
    border: 1px solid #8B4513; /* 整体容器的棕色边框 */
    border-radius: 10px;  /* 圆角效果 */
}}

.description {{
    text-align: center;
    font-size: 18px;
    color: #4B2E2E;
}}
"""

# 定义 Gradio 界面
def gradio_interface(message, history, enable_rag, enable_baidu_search, enable_image_generation):
    final_response, updated_history, image_html = chat(message, history, enable_rag, enable_baidu_search, enable_image_generation, index, tokenizer, model)

    # 如果不启用图片生成，隐藏图片输出
    if not enable_image_generation:
        image_html = gr.update(visible=False)
    else:
        image_html = gr.HTML(value=image_html, visible=True)

    return final_response, updated_history, image_html

# 定义 Gradio 界面
gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="提问"), 
        gr.State(), 
        gr.Checkbox(label="RAG工具", value=False),
        gr.Checkbox(label="联网检索", value=False),
        gr.Checkbox(label="画图工具", value=False)
    ],
    outputs=[
        gr.Textbox(label="回答"),
        gr.State(),
        gr.HTML(label="生成的图片")  # 使用HTML组件展示图片
    ],
    title="史小白 - 中国历史智能助手",
    description="""
    <h2>欢迎使用史小白！</h2>
    <p>史小白致力于帮助用户学习和探索中国历史。具备<strong>历史问答</strong>、<strong>文言文翻译</strong>、<strong>中国古风图片生成</strong>等多项功能。</p>
    """,
    css=css,  # 引入自定义 CSS
).launch()