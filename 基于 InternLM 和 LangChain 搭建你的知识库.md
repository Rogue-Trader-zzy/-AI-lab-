> ## 环境配置

安装所需依赖langchain等，下载模型InternLM-7B。
> ## 知识库搭建

### 收集数据：
收集后缀为md和txt的文本数据。
### 加载数据：
使用 LangChain 提供的 FileLoader加载上述收集到的数据集，返回一个纯文本对象对应的列表docs。
### 构建向量数据库：
对文本分快，选择分块大小为 500，块重叠长度为 150；选用开源词向量模型 [Sentence Transformer](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) 来进行文本向量化；选择 Chroma 作为向量数据库。<br />![截屏2024-01-14 0.37.50.png](https://cdn.nlark.com/yuque/0/2024/png/40736923/1705163886152-ba38c29a-bf14-4879-9ecc-458d7c3654ff.png#averageHue=%23e8ead9&clientId=u6150470d-702d-4&from=paste&height=623&id=u68b2e0dc&originHeight=1246&originWidth=898&originalType=binary&ratio=2&rotation=0&showTitle=false&size=1087054&status=done&style=none&taskId=u5c449bb8-5b80-49e5-a0c4-b1882fcce2e&title=&width=449)![截屏2024-01-14 0.41.18.png](https://cdn.nlark.com/yuque/0/2024/png/40736923/1705164089094-b9730ced-8674-4f94-9a98-64a65fe7868a.png#averageHue=%23e7f0dc&clientId=u6150470d-702d-4&from=drop&id=u4b6b3639&originHeight=1286&originWidth=2470&originalType=binary&ratio=2&rotation=0&showTitle=false&size=4758146&status=done&style=none&taskId=u334fb82c-1501-495e-b57a-5fec07eb3dd&title=)<br />上图为切割的文本和向量化后的文本效果。
## InternLM 接入 LangChain
为便捷构建 LLM 应用，我们需要基于本地部署的 InternLM，继承 LangChain 的 LLM 类自定义一个 InternLM LLM 子类，从而实现将 InternLM 接入到 LangChain 框架中。我们分别重写了构造函数和 _call 函数：对于构造函数，我们在对象实例化的一开始加载本地部署的 InternLM 模型，从而避免每一次调用都需要重新加载模型带来的时间过长；_call 函数是 LLM 类的核心函数，LangChain 会调用该函数来调用 LLM，在该函数中，我们调用已实例化模型的 chat 方法，从而实现对模型的调用并返回调用结果。
## 构建检索问答链
将上文构建的向量数据库导入进来，我们可以直接通过 Chroma 以及上文定义的词向量模型来加载已构建的数据库。构建检索问答链，还需要构建一个 Prompt Template，该 Template 其实基于一个带变量的字符串，在检索之后，LangChain 会将检索到的相关文档片段填入到 Template 的变量中，从而实现带知识的 Prompt 构建。![截屏2024-01-14 0.51.28.png](https://cdn.nlark.com/yuque/0/2024/png/40736923/1705164695713-44951834-94f3-48fe-addc-894082444fcf.png#averageHue=%2373786e&clientId=u6150470d-702d-4&from=drop&id=u365c89b9&originHeight=518&originWidth=2672&originalType=binary&ratio=2&rotation=0&showTitle=false&size=759315&status=done&style=none&taskId=u8b24eeee-876b-4b11-bccb-64446bd91bf&title=)
## 部署 Web Demo
基于 Gradio 框架将其部署到 Web 网页，从而搭建一个小型 Demo。<br />我们首先将上文的代码内容封装为一个返回构建的检索问答链对象的函数，并在启动 Gradio 的第一时间调用该函数得到检索问答链对象，后续直接使用该对象进行问答对话，从而避免重复加载模型：
```python

from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from LLM import InternLM_LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def load_chain():
    # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")

    # 向量数据库持久化路径
    persist_directory = 'data_base/vector_db/chroma'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embeddings
    )

    # 加载自定义 LLM
    llm = InternLM_LLM(model_path = "/root/data/model/Shanghai_AI_Laboratory/internlm-chat-7b")

    # 定义一个 Prompt Template
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)

    # 运行 chain
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

    return qa_chain
```
接着我们定义一个类，该类负责加载并存储检索问答链，并响应 Web 界面里调用检索问答链进行回答的动作：
```python
class Model_center():
    """
    存储检索问答链的对象 
    """
    def __init__(self):
        # 构造函数，加载检索问答链
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append(
                (question, self.chain({"query": question})["result"]))
            # 将问答结果直接附加到问答历史中，Gradio 会将其展示出来
            return "", chat_history
        except Exception as e:
            return e, chat_history

```
然后我们只需按照 Gradio 的框架使用方法，实例化一个 Web 界面并将点击动作绑定到上述类的回答方法即可：
```python
import gradio as gr

# 实例化核心功能对象
model_center = Model_center()
# 创建一个 Web 界面
block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            # 展示的页面标题
            gr.Markdown("""<h1><center>InternLM</center></h1>
                <center>书生浦语</center>
                """)

    with gr.Row():
        with gr.Column(scale=4):
            # 创建一个聊天机器人对象
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")
                
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])

    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
gr.close_all()
# 直接启动
demo.launch()
```
将上述代码封装为 run_gradio.py 脚本，直接通过 python 命令运行，即可在本地启动知识库助手的 Web Demo，默认会在 7860 端口运行。<br />![截屏2024-01-14 1.00.51.png](https://cdn.nlark.com/yuque/0/2024/png/40736923/1705165268483-f9a016ce-336e-47ce-adb6-27c743f618a2.png#averageHue=%23f7f7f6&clientId=u6150470d-702d-4&from=drop&id=uef25e781&originHeight=1426&originWidth=2668&originalType=binary&ratio=2&rotation=0&showTitle=false&size=875017&status=done&style=none&taskId=ud7199334-270d-42da-b18a-46672421e4a&title=)<br />最后结果这样的，未找到原因。有谁知道的也可以告诉我下哈。
