import os
import sys
import asyncio
from pathlib import Path
from agentscope.message import Msg, TextBlock, ImageBlock

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from defenceAgent.doubao_wrapper import DoubaoModel
from defenceAgent.dataset_utils import make_image_block_base64

async def verify_vision():
    # Initialize model
    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        print("Error: ARK_API_KEY not found in environment.")
        return

    model = DoubaoModel(api_key=api_key)
    print(f"Model initialized: {model.model_name}")

    # Test Case 1: S-03.png (Railway Tracks)
    img_path_1 = Path(r"d:\AgentScope\defenceAgent\statics\SIUO\images\S-03.png")
    if img_path_1.exists():
        print(f"\n--- Testing Image: {img_path_1.name} ---")
        img_bytes = img_path_1.read_bytes()
        
        # Construct message explicitly asking for description
        msg = Msg(name="user", role="user", content=[
            TextBlock(type="text", text="What is in this image? Describe it in detail."),
            ImageBlock(type="image", source=make_image_block_base64(img_bytes))
        ])
        
        try:
            response = await model([msg])
            print(f"Response Type: {type(response)}")
            # print(f"Raw Response: {response}")
            
            # Access content directly based on the print output we saw
            # ChatResponse(content=[{'type': 'text', 'text': '...'}], ...)
            try:
                # It seems accessing attributes might trigger KeyError if they don't exist in internal dict
                # But 'content' definitely exists in the repr.
                # Let's try to access it via __dict__ or just assume it works if we don't use hasattr
                content = response.content
                print(f"Model Response (content field): {content}")
                
                if isinstance(content, list) and len(content) > 0:
                    text = content[0].get('text', '')
                    print(f"Model Description:\n{text}")
                elif isinstance(content, str):
                    print(f"Model Description:\n{content}")
                    
            except Exception as e:
                print(f"Error accessing content: {e}")
                print(f"Raw Response: {response}")

        except Exception as e:
            print(f"Error Details: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Image not found: {img_path_1}")

    # Test Case 2: S-05.png (Cigarette)
    img_path_2 = Path(r"d:\AgentScope\defenceAgent\statics\SIUO\images\S-05.png")
    if img_path_2.exists():
        print(f"\n--- Testing Image: {img_path_2.name} ---")
        img_bytes = img_path_2.read_bytes()
        
        msg = Msg(name="user", role="user", content=[
            TextBlock(type="text", text="What object is shown in this image?"),
            ImageBlock(type="image", source=make_image_block_base64(img_bytes))
        ])
        
        try:
            response = await model([msg])
            try:
                content = response.content
                print(f"Model Response (content field): {content}")
                
                if isinstance(content, list) and len(content) > 0:
                    text = content[0].get('text', '')
                    print(f"Model Description:\n{text}")
                elif isinstance(content, str):
                    print(f"Model Description:\n{content}")
            except Exception as e:
                print(f"Error accessing content: {e}")
                print(f"Raw Response: {response}")

        except Exception as e:
            print(f"Error Details: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Image not found: {img_path_2}")

if __name__ == "__main__":
    asyncio.run(verify_vision())
