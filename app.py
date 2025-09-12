import spaces
import gradio as gr
import re
import os 
hf_token = os.environ.get('HF_TOKEN')

from gradio_client import Client, handle_file

clipi_client = Client("fffiloni/CLIP-Interrogator-2")


import requests

HF_API_URL = "https://fffiloni-test-llama-api-debug.hf.space/api/predict"

def llama_gen_story(prompt):
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    payload = {"inputs": prompt}
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        print(f"API response status: {response.status_code}")
        print(f"API response text: {response.text}")
        if response.status_code == 200:
            result = response.json()
            # GPT-2 API returns a list of dicts with 'generated_text'
            output_text = None
            if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                output_text = result[0]["generated_text"]
            elif isinstance(result, dict) and "generated_text" in result:
                output_text = result["generated_text"]
            elif isinstance(result, str):
                output_text = result
            if output_text:
                return output_text.strip()
            else:
                return "No story generated. The model did not return any text."
        else:
            return f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        print(f"Exception during API call: {e}")
        return f"Exception during API call: {e}"

#client = Client("https://fffiloni-test-llama-api-debug.hf.space/", hf_token=hf_token)

clipi_client = Client("https://fffiloni-clip-interrogator-2.hf.space/")

@spaces.GPU
def llama_gen_story(prompt):
    """Generate a fictional story using the LLaMA 2 model based on a prompt.
    
    Args:
        prompt: A string prompt containing an image description and story generation instructions.
        
    Returns:
        A generated fictional story string with special formatting and tokens removed.
    """

    instruction = """[INST] <<SYS>>\nYou are a storyteller. You'll be given an image description and some keyword about the image. 
            For that given you'll be asked to generate a story that you think could fit very well with the image provided.
            Always answer with a cool story, while being safe as possible.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""

    
    prompt = instruction.format(prompt)
    
    # Now handled by API call above

def get_text_after_colon(input_text):
    # Find the first occurrence of ":"
    colon_index = input_text.find(":")
    
    # Check if ":" exists in the input_text
    if colon_index != -1:
        # Extract the text after the colon
        result_text = input_text[colon_index + 1:].strip()
        return result_text
    else:
        # Return the original text if ":" is not found
        return input_text

def infer(image_input, audience):
    """Generate a fictional story based on an image using CLIP Interrogator and LLaMA2.
    
    Args:
        image_input: A file path to the input image to analyze.
        audience: A string indicating the target audience, such as 'Children' or 'Adult'.
    
    Returns:
        A formatted, multi-paragraph fictional story string related to the image content.
        
    Steps:
        1. Use the CLIP Interrogator model to generate a semantic caption from the image.
        2. Format a prompt asking the LLaMA2 model to write a story based on the caption.
        3. Clean and format the story output for readability.
    """
    gr.Info('Calling CLIP Interrogator ...')

    clipi_result = clipi_client.predict(
		image=handle_file(image_input),
		mode="best",
		best_max_flavors=4,
		api_name="/clipi2"
    )
    print(clipi_result)
   

    instruction = """[INST] <<SYS>>\nYou are a storyteller. You'll be given an image description and some keyword about the image. \nFor that given you'll be asked to generate a story that you think could fit very well with the image provided.\nAlways answer with a cool story, while being safe as possible.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""
    llama_q = instruction.format(clipi_result)
    gr.Info('Calling Llama2 ...')
    result = llama_gen_story(llama_q)
    print(f"Llama2 result: {result}")
    if not result or not isinstance(result, str):
        return "Sorry, no story could be generated. Please try again or check your API settings."
    # Only process if result is a valid string
    result = get_text_after_colon(result)
    paragraphs = result.split('\n')
    formatted_text = '\n\n'.join(paragraphs)
    return formatted_text

css="""
#col-container {max-width: 910px; margin-left: auto; margin-right: auto;}


div#story textarea {
    font-size: 1.5em;
    line-height: 1.4em;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """
            <h1 style="text-align: center">Image to Story</h1>
            <p style="text-align: center">Upload an image, get a story made by Llama2 !</p>
            """
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(label="Image input", type="filepath", elem_id="image-in")
                audience = gr.Radio(label="Target Audience", choices=["Children", "Adult"], value="Children")
                submit_btn = gr.Button('Tell me a story')
            with gr.Column():
                #caption = gr.Textbox(label="Generated Caption")
                story = gr.Textbox(label="generated Story", elem_id="story")
        
        gr.Examples(examples=[["./examples/crabby.png", "Children"],["./examples/hopper.jpeg", "Adult"]],
                    fn=infer,
                    inputs=[image_in, audience],
                    outputs=[story],
                    cache_examples=False
                   )
        
    submit_btn.click(fn=infer, inputs=[image_in, audience], outputs=[story])

demo.queue(max_size=12).launch(ssr_mode=False, mcp_server=True)
