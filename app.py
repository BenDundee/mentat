import gradio as gr
import requests

def chat_with_coach(message, history):
    response = requests.post("http://localhost:8000/chat",
                            json={"message": message, "history": history})
    return response.json()["response"]

if __name__ == "__main__":

    demo = gr.ChatInterface(
        fn=chat_with_coach,
        title="Executive Coach Bot",
        description="Your AI executive coach to help you achieve your goals.",
        theme="soft",
        examples=["Help me set a career goal", "I'm struggling with work-life balance"]
    )

    demo.launch()