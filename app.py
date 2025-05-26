import gradio as gr
import requests
import json

# Keep track of the chat history for the API
session_history = []

def chat_with_coach(message, history):
    # Format history for the API
    formatted_history = [[msg, resp] for msg, resp in history]
    
    # Call the API
    response = requests.post(
        "http://localhost:8000/chat",
        json={
            "message": message, 
            "history": formatted_history,
            "user_id": "gradio_user"  # You could make this configurable
        }
    )
    
    # Update session history
    session_history.append([message, response.json()["response"]])
    
    return response.json()["response"]

def clear_history():
    global session_history
    session_history = []
    return None

def save_conversation():
    with open("conversation_history.json", "w") as f:
        json.dump(session_history, f)
    return "Conversation saved to conversation_history.json"

def load_conversation():
    try:
        with open("conversation_history.json", "r") as f:
            loaded_history = json.load(f)
            global session_history
            session_history = loaded_history
            
            # Format for Gradio chatbot
            formatted_history = [(msg, resp) for msg, resp in loaded_history]
            return formatted_history
    except:
        return None

if __name__ == "__main__":
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Executive Coach Bot")
        gr.Markdown("Your AI executive coach to help you achieve your goals and improve your leadership skills.")
        
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    bubble_full_width=False,
                    height=600,
                    show_copy_button=True
                )
                
                msg = gr.Textbox(
                    label="Type your message here",
                    placeholder="Ask me about goal setting, leadership, or work-life balance...",
                    lines=3
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Conversation")
                
            with gr.Column(scale=1):
                gr.Markdown("### Options")
                save_btn = gr.Button("Save Conversation")
                load_btn = gr.Button("Load Conversation")
                
                save_status = gr.Textbox(label="Status", visible=True)
                
                gr.Markdown("### Example Prompts")
                example_prompts = gr.Examples(
                    examples=[
                        "Help me set a career goal",
                        "I'm struggling with work-life balance",
                        "How can I improve my leadership skills?",
                        "I need help with difficult conversations",
                        "Can you help me journal about my day?"
                    ],
                    inputs=msg
                )
        
        # Set up event handlers
        submit_btn.click(
            chat_with_coach, 
            inputs=[msg, chatbot], 
            outputs=[chatbot]
        ).then(
            lambda: "", 
            None, 
            msg
        )
        
        msg.submit(
            chat_with_coach, 
            inputs=[msg, chatbot], 
            outputs=[chatbot]
        ).then(
            lambda: "", 
            None, 
            msg
        )
        
        clear_btn.click(
            lambda: None, 
            None, 
            chatbot
        ).then(
            clear_history
        )
        
        save_btn.click(
            save_conversation,
            outputs=[save_status]
        )
        
        load_btn.click(
            load_conversation,
            outputs=[chatbot]
        )
        
    # Launch the demo
    demo.launch(
        share=False, server_name="0.0.0.0", server_port=7860
    )