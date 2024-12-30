import gradio as gr
import time

# 模拟训练日志更新函数
def update_log():
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return f"Training Log Updated at: {current_time}"

# 创建 Gradio 界面
with gr.Blocks() as demo:
    train_log = gr.Textbox(label="Training Log")
    
    # 添加定时任务来更新日志
    demo.load(fn=update_log, outputs=[train_log], every=1000)

# 启动 Gradio 界面
demo.launch()